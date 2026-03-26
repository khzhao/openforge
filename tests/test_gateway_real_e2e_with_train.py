# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
import requests
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from _script_test_utils import start_test_ray_cluster

from test_gateway_real_e2e import (
    DEFAULT_MODEL,
    ROOT,
    chat_payload,
    get_free_port,
    make_artifact_dir,
    require_visible_gpus,
    request_json,
    resolve_model_path,
    wait_for_http,
    write_temp_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=3)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--server-start-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def build_start_session_payload(
    *,
    model_path: str,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    checkpoint_root: str,
) -> dict[str, Any]:
    return {
        "runtime": {
            "algo": {
                "name": "grpo",
                "clip_range": 0.2,
                "kl_coef": 0.0,
                "entropy_coef": 0.0,
                "normalize_group_std": False,
            },
            "model": {
                "model_name_or_path": model_path,
                "reference_model_name_or_path": model_path,
                "tokenizer_name_or_path": model_path,
                "attn_implementation": "sdpa",
            },
            "train": {
                "backend": "fsdp2",
                "config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": False,
                    "mixed_precision": {
                        "param_dtype": "bfloat16",
                        "reduce_dtype": "float32",
                    },
                    "offload": {"mode": "none", "pin_memory": False},
                    "amp": {
                        "enabled": False,
                        "precision": "float32",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 1.0e-1,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1.0e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 0,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                    },
                },
                "global_batch_size": 2,
                "mini_batch_size": 2,
                "micro_batch_size": 1,
                "max_rollout_policy_lag": 0,
                "checkpoints": checkpoint_root,
                "cpus_per_worker": 1,
                "parallel": {
                    "data_parallel_size": train_gpus,
                    "fsdp_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": 8,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                },
                "engine_groups": [
                    {
                        "name": "regular",
                        "worker_type": "regular",
                        "replicas": rollout_replicas,
                        "num_gpus_per_replica": gpus_per_replica,
                        "num_cpus_per_replica": 1,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": gpus_per_replica,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "enable_memory_saver": False,
                    }
                ],
            },
        },
    }


def record_event(
    path: Path,
    step: str,
    payload: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> None:
    event = {"step": step, "payload": payload}
    if events is not None:
        events.append(event)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def changed_parameter(
    *,
    model_path: str,
    checkpoint_dir: Path,
) -> tuple[str, float]:
    checkpoint_state = load_file(checkpoint_dir / "model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        trust_remote_code=True,
        dtype=torch.float32,
    )
    base_state = model.state_dict()
    for name, tensor in checkpoint_state.items():
        if name not in base_state:
            continue
        base_tensor = base_state[name].to(dtype=tensor.dtype, device=tensor.device)
        if not tensor.is_floating_point():
            continue
        diff = (tensor - base_tensor).abs().max().item()
        if diff > 0:
            return name, float(diff)
    raise RuntimeError("checkpoint weights did not change")


def trajectory_rows(sqlite_path: Path) -> list[tuple[str, str | None, str, float | None]]:
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = list(
            conn.execute(
                "select trajectory_id, group_id, status, final_reward from trajectories order by rowid"
            )
        )
    finally:
        conn.close()
    return rows


def wait_for_group_rows(
    *,
    sqlite_path: Path,
    group_id: str,
    expected_count: int,
    timeout: float,
) -> list[tuple[str, str | None, str, float | None]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = trajectory_rows(sqlite_path)
        group_rows = [row for row in rows if row[1] == group_id]
        if len(group_rows) == expected_count and all(
            row[2] == "trained" for row in group_rows
        ):
            return rows
        time.sleep(1.0)
    raise TimeoutError("timed out waiting for group trajectories to become trained")


def main() -> int:
    args = parse_args()
    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    gpu_ids = require_visible_gpus(total_requested_gpus)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    gateway_log_path = artifact_dir / "gateway_process.log"
    response_log_path = artifact_dir / "responses.jsonl"
    summary_path = artifact_dir / "summary.json"
    checkpoint_root = artifact_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config_path = write_temp_config(
        model_path=model_path,
        gateway_host=args.gateway_host,
        gateway_port=gateway_port,
        visible_gpus=len(gpu_ids),
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_node=args.cpus_per_node,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    env["RAY_ADDRESS"] = start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=args.cpus_per_node,
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openforge.cli.main",
            "gateway",
            "start",
            "--config",
            str(config_path),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        base_url = f"http://{args.gateway_host}:{gateway_port}"
        wait_for_http(f"{base_url}/health", timeout=args.server_start_timeout)

        started = request_json(
            "POST",
            f"{base_url}/start_session",
            build_start_session_payload(
                model_path=model_path,
                train_gpus=args.train_gpus,
                rollout_replicas=args.rollout_replicas,
                gpus_per_replica=args.gpus_per_replica,
                checkpoint_root=str(checkpoint_root),
            ),
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "start_session", started)
        session_id = str(started["session_id"])
        model_name = str(started["model"])

        parent = request_json(
            "POST",
            f"{base_url}/start_trajectory",
            {"session_id": session_id},
            timeout=60.0,
        )
        record_event(response_log_path, "start_parent", parent)
        parent_id = str(parent["trajectory_id"])

        parent_turn = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(
                session_id=session_id,
                trajectory_id=parent_id,
                model=model_name,
                content="Say hello in four words.",
            ),
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "generate_parent", parent_turn)

        child_group_id = "group-0"
        child_group = request_json(
            "POST",
            f"{base_url}/start_trajectory_groups",
            {
                "session_id": session_id,
                "counts": [2],
                "group_ids": [child_group_id],
            },
            timeout=60.0,
        )
        record_event(response_log_path, "start_child_group", child_group)
        child_ids = child_group["trajectory_ids"][0]
        child_a_id = str(child_ids[0])
        child_b_id = str(child_ids[1])

        child_a_turn = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(
                session_id=session_id,
                trajectory_id=child_a_id,
                model=model_name,
                content="Now answer: yes.",
            ),
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "generate_child_a", child_a_turn)

        child_b_turn = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(
                session_id=session_id,
                trajectory_id=child_b_id,
                model=model_name,
                content="Now answer: no.",
            ),
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "generate_child_b", child_b_turn)

        for trajectory_id, reward, step in (
            (parent_id, 0.0, "end_parent"),
            (child_a_id, 1.0, "end_child_a"),
            (child_b_id, -1.0, "end_child_b"),
        ):
            ended = request_json(
                "POST",
                f"{base_url}/end_trajectory",
                {
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "final_reward": reward,
                },
                timeout=60.0,
            )
            record_event(response_log_path, step, ended)

        sqlite_path = config_path.parent / "gateway.sqlite3"
        rows = wait_for_group_rows(
            sqlite_path=sqlite_path,
            group_id=child_group_id,
            expected_count=2,
            timeout=args.request_timeout,
        )
        post_train = request_json(
            "POST",
            f"{base_url}/start_trajectory",
            {"session_id": session_id},
            timeout=60.0,
        )
        record_event(response_log_path, "start_post_train", post_train)
        post_train_id = str(post_train["trajectory_id"])

        post_train_turn = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(
                session_id=session_id,
                trajectory_id=post_train_id,
                model=model_name,
                content="Say hello in four words.",
            ),
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "generate_post_train", post_train_turn)
        rollout_model_version = post_train_turn["metadata"]["rollout_model_version"]
        assert int(rollout_model_version) > 0

        ended_post_train = request_json(
            "POST",
            f"{base_url}/end_trajectory",
            {
                "session_id": session_id,
                "trajectory_id": post_train_id,
                "final_reward": 0.0,
            },
            timeout=60.0,
        )
        record_event(response_log_path, "end_post_train", ended_post_train)

        ended_session = request_json(
            "POST",
            f"{base_url}/end_session",
            {"session_id": session_id},
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "end_session", ended_session)

        rows = trajectory_rows(sqlite_path)
        child_rows = [row for row in rows if row[1] == child_group_id]
        assert len(child_rows) == 2
        assert all(row[2] == "trained" for row in child_rows)

        checkpoint_dir = None
        changed_name = None
        max_abs_diff = None
        checkpoint_dirs = sorted(
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.startswith("policy-")
        )
        if checkpoint_dirs:
            checkpoint_dir = checkpoint_dirs[-1]
            changed_name, max_abs_diff = changed_parameter(
                model_path=model_path,
                checkpoint_dir=checkpoint_dir,
            )

        summary = {
            "artifact_dir": str(artifact_dir),
            "config_path": str(config_path),
            "gateway_log_path": str(gateway_log_path),
            "response_log_path": str(response_log_path),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "sqlite_path": str(sqlite_path),
            "post_train_rollout_model_version": rollout_model_version,
            "changed_parameter": changed_name,
            "max_abs_diff": max_abs_diff,
            "trajectory_rows": rows,
            "status": "ok",
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("SUMMARY_PATH", summary_path, flush=True)
        print(
            "POST_TRAIN_ROLLOUT_MODEL_VERSION",
            rollout_model_version,
            flush=True,
        )
        if changed_name is not None:
            print("CHANGED_PARAMETER", changed_name, max_abs_diff, flush=True)
        print("GATEWAY_REAL_E2E_WITH_TRAIN_OK", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30.0)
        if proc.stdout is not None:
            output = proc.stdout.read()
            gateway_log_path.write_text(output, encoding="utf-8")
            if output.strip():
                print("GATEWAY_LOG_START", flush=True)
                print(output[-12000:], flush=True)
                print("GATEWAY_LOG_END", flush=True)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
