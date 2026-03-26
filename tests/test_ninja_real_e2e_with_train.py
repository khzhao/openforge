# Copyright 2026 openforge

"""Run a real openforge.ninja train smoke against a live gateway and model stack."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import openforge.ninja as ninja
import ray
from _script_test_utils import start_test_ray_cluster
from test_gateway_real_e2e import (
    DEFAULT_MODEL,
    ROOT,
    get_free_port,
    make_artifact_dir,
    request_json,
    require_visible_gpus,
    resolve_model_path,
    wait_for_http,
    write_temp_config,
)
from test_gateway_real_e2e_with_train import (
    build_start_session_payload,
    changed_parameter,
    record_event,
    trajectory_rows,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the live Ninja train smoke."""
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
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--algo-name", choices=("grpo", "grpo_tis"), default="grpo")
    parser.add_argument("--max-rollout-policy-lag", type=int, default=0)
    parser.add_argument("--tis-cap", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-1)
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def wait_for_trajectory_statuses(
    *,
    base_url: str,
    session_id: str,
    trajectory_ids: list[str],
    timeout: float,
) -> dict[str, str]:
    """Poll the gateway until all requested trajectories are trained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = request_json(
            "POST",
            f"{base_url}/trajectory_statuses",
            {
                "session_id": session_id,
                "trajectory_ids": trajectory_ids,
            },
            timeout=min(60.0, timeout),
        )
        statuses = {
            str(item["trajectory_id"]): str(item["status"])
            for item in payload["trajectories"]
        }
        if any(status in {"failed", "discarded"} for status in statuses.values()):
            raise RuntimeError(f"unexpected trajectory failure: {statuses}")
        if statuses and all(status == "trained" for status in statuses.values()):
            return statuses
        time.sleep(1.0)
    raise TimeoutError(
        f"timed out waiting for trajectories to become trained: {trajectory_ids}"
    )


def main() -> int:
    """Run the real Ninja train smoke."""
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("group_size must be > 0")
    if args.global_batch_size <= 0:
        raise ValueError("global_batch_size must be > 0")
    if args.global_batch_size % args.group_size != 0:
        raise ValueError("global_batch_size must be divisible by group_size")
    if args.algo_name == "grpo" and args.max_rollout_policy_lag != 0:
        raise ValueError("grpo requires max_rollout_policy_lag=0")
    if args.algo_name == "grpo_tis" and args.max_rollout_policy_lag <= 0:
        raise ValueError("grpo_tis requires max_rollout_policy_lag > 0")
    prompt_group_count = args.global_batch_size // args.group_size
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
    response_log_path.write_text("", encoding="utf-8")
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
    session_id: str | None = None

    try:
        base_url = f"http://{args.gateway_host}:{gateway_port}"
        wait_for_http(f"{base_url}/health", timeout=args.server_start_timeout)

        start_payload = build_start_session_payload(
            model_path=model_path,
            train_gpus=args.train_gpus,
            rollout_replicas=args.rollout_replicas,
            gpus_per_replica=args.gpus_per_replica,
            checkpoint_root=str(checkpoint_root),
        )
        algo_cfg = start_payload["runtime"]["algo"]
        algo_cfg["name"] = args.algo_name
        if args.algo_name == "grpo_tis":
            algo_cfg["tis_cap"] = args.tis_cap
        train_cfg = start_payload["runtime"]["train"]
        train_cfg["global_batch_size"] = args.global_batch_size
        train_cfg["mini_batch_size"] = args.mini_batch_size
        train_cfg["micro_batch_size"] = args.micro_batch_size
        train_cfg["max_rollout_policy_lag"] = args.max_rollout_policy_lag
        train_cfg["config"]["optim"]["lr"] = args.lr
        started = request_json(
            "POST",
            f"{base_url}/start_session",
            start_payload,
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "start_session", started)
        session_id = str(started["session_id"])

        train_trajectory_ids: list[str] = []
        train_reward_lock = threading.Lock()
        prompt_variant_counts: dict[str, int] = {}
        follow_ups = [
            "Now answer: yes.",
            "Now answer: no.",
            "Now answer: yes, formally.",
            "Now answer: no, formally.",
        ]

        @ninja.agent()
        def train_agent(client, *, prompt: str) -> float:
            with train_reward_lock:
                variant = prompt_variant_counts.get(prompt, 0)
                prompt_variant_counts[prompt] = variant + 1
            reward = 1.0 if variant % 2 == 0 else -1.0
            follow_up = follow_ups[variant % len(follow_ups)]
            response = client.chat.completions.create(
                model=model_path,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": follow_up},
                ],
                temperature=0.0,
                top_p=1.0,
                max_completion_tokens=8,
            )
            train_trajectory_ids.append(
                str(response.model_dump(mode="json")["metadata"]["trajectory_id"])
            )
            return reward

        probe_response: dict[str, Any] | None = None

        @ninja.agent()
        def probe_agent(client, *, prompt: str) -> float:
            nonlocal probe_response
            response = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_completion_tokens=8,
            )
            probe_response = response.model_dump(mode="json")
            return 0.0

        initial_policy_version = probe_agent.policy_version()
        assert initial_policy_version == 0
        record_event(
            response_log_path,
            "initial_policy_version",
            {"policy_version": initial_policy_version},
        )

        train_inputs = [
            {"prompt": f"Prompt {index + 1}: Say hello in four words."}
            for index in range(prompt_group_count)
        ]
        train_summary = ninja.train(
            train_agent,
            inputs=train_inputs,
            group_size=args.group_size,
            wait_timeout=args.request_timeout,
        )
        record_event(response_log_path, "ninja_train", train_summary)
        assert train_summary["group_size"] == args.group_size
        assert train_summary["prompt_groups"] == len(train_inputs)
        assert train_summary["samples"] == len(train_inputs) * args.group_size
        assert int(train_summary["final_policy_version"]) > int(
            train_summary["initial_policy_version"]
        )
        assert len(train_trajectory_ids) == len(train_inputs) * args.group_size
        train_statuses = wait_for_trajectory_statuses(
            base_url=base_url,
            session_id=session_id,
            trajectory_ids=train_trajectory_ids,
            timeout=args.request_timeout,
        )
        record_event(
            response_log_path,
            "train_trajectory_statuses",
            train_statuses,
        )

        probe_reward = probe_agent(prompt="Say hello in four words.")
        assert probe_reward == 0.0
        assert probe_response is not None
        record_event(response_log_path, "probe_after_train", probe_response)
        assert int(probe_response["metadata"]["rollout_model_version"]) > 0

        exported = train_agent.save()
        record_event(response_log_path, "export_checkpoint", exported)
        checkpoint_dir = Path(str(exported["checkpoint_path"]))
        changed_name, max_abs_diff = changed_parameter(
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
        )
        assert max_abs_diff > 0.0

        sqlite_path = config_path.parent / "gateway.sqlite3"
        rows = trajectory_rows(sqlite_path)
        tracked_rows = [
            row for row in rows if row[0] in set(train_trajectory_ids)
        ]
        assert len(tracked_rows) == len(train_trajectory_ids)
        assert all(row[2] == "trained" for row in tracked_rows)

        ended = request_json(
            "POST",
            f"{base_url}/end_session",
            {"session_id": session_id},
            timeout=args.request_timeout,
        )
        record_event(response_log_path, "end_session", ended)
        session_id = None

        summary = {
            "artifact_dir": str(artifact_dir),
            "config_path": str(config_path),
            "gateway_log_path": str(gateway_log_path),
            "response_log_path": str(response_log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "sqlite_path": str(sqlite_path),
            "group_size": args.group_size,
            "global_batch_size": args.global_batch_size,
            "mini_batch_size": args.mini_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "algo_name": args.algo_name,
            "max_rollout_policy_lag": args.max_rollout_policy_lag,
            "tis_cap": args.tis_cap,
            "lr": args.lr,
            "initial_policy_version": initial_policy_version,
            "train_summary": train_summary,
            "train_inputs": train_inputs,
            "train_trajectory_ids": train_trajectory_ids,
            "train_statuses": train_statuses,
            "post_train_rollout_model_version": probe_response["metadata"][
                "rollout_model_version"
            ],
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
        print("POST_TRAIN_ROLLOUT_MODEL_VERSION", probe_response["metadata"]["rollout_model_version"], flush=True)
        print("CHANGED_PARAMETER", changed_name, max_abs_diff, flush=True)
        print("NINJA_REAL_E2E_WITH_TRAIN_OK", flush=True)
        return 0
    finally:
        if session_id is not None:
            try:
                request_json(
                    "POST",
                    f"{base_url}/end_session",
                    {"session_id": session_id},
                    timeout=60.0,
                )
            except Exception:
                pass
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
