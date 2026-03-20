# Copyright 2026 openforge

"""Run a real gateway end-to-end smoke against live SGLang rollout runtime.

This script:
1. Resolves a local or Hub-backed model path.
2. Writes a temporary OpenForge config for one gateway, one train worker, and one
   rollout replica.
3. Starts ``openforge.gateway.main`` as a real subprocess.
4. Issues HTTP requests to the live gateway:
   ``/models``, ``/start_session``, ``/start_trajectory``, ``/generate``,
   ``/end_trajectory``, and ``/end_session``.

It is intentionally a standalone script, not a pytest test, so it can be used as
an actual end-to-end smoke in GPU environments.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "test_scripts" / "logs"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the gateway smoke script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--server-start-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for saving response and gateway logs.",
    )
    return parser.parse_args()


def require_visible_gpus(min_count: int) -> int:
    """Validate that enough GPUs are visible for the requested topology."""
    visible = torch.cuda.device_count()
    if visible < min_count:
        raise RuntimeError(
            f"Expected at least {min_count} visible GPUs, found {visible}"
        )
    return visible


def resolve_model_path(model_path_or_id: str) -> str:
    """Resolve a local model path or download/cache a Hub model snapshot."""
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    try:
        return snapshot_download(repo_id=model_path_or_id, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=model_path_or_id)


def get_free_port(host: str) -> int:
    """Return an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_http(url: str, *, timeout: float) -> None:
    """Poll an HTTP endpoint until it returns status 200."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error!r}")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Send one JSON HTTP request and return the parsed JSON response."""
    response = requests.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def write_temp_config(
    *,
    model_path: str,
    gateway_host: str,
    gateway_port: int,
    visible_gpus: int,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    cpus_per_node: int,
) -> Path:
    """Write a temporary gateway-server YAML config for the live smoke."""
    root = Path(tempfile.mkdtemp(prefix="openforge-gateway-e2e-"))
    sqlite_path = root / "gateway.sqlite3"
    config_path = root / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            data:
              path: {sqlite_path}
            gateway:
              host: {gateway_host}
              port: {gateway_port}
            cluster:
              num_nodes: 1
              gpus_per_node: {visible_gpus}
              cpus_per_node: {cpus_per_node}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def build_start_session_payload(
    *,
    model_path: str,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    checkpoint_root: str,
) -> dict[str, Any]:
    """Build the user-facing start_session payload."""
    return {
        "runtime": {
            "algo": {"kl_coef": 0.0},
            "model": {
                "model_name_or_path": model_path,
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
                        "lr": 1.0e-5,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1.0e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 1,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                    },
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
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
                    "spaces_between_words": True,
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


def make_artifact_dir(path: str | None) -> Path:
    """Create and return a directory for storing smoke artifacts."""
    if path is not None:
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gateway-real-e2e-", dir=LOG_ROOT))


def main() -> int:
    """Run the real gateway end-to-end smoke."""
    args = parse_args()
    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    visible_gpus = require_visible_gpus(total_requested_gpus)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    gateway_log_path = artifact_dir / "gateway_process.log"
    response_log_path = artifact_dir / "responses.jsonl"
    summary_path = artifact_dir / "summary.json"
    for path in (gateway_log_path, response_log_path, summary_path):
        if path.exists():
            path.unlink()
    checkpoint_root = str(artifact_dir / "checkpoints")
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    config_path = write_temp_config(
        model_path=model_path,
        gateway_host=args.gateway_host,
        gateway_port=gateway_port,
        visible_gpus=visible_gpus,
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_node=args.cpus_per_node,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [sys.executable, "-m", "openforge.gateway.main", "--config", str(config_path)]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    response_events: list[dict[str, Any]] = []

    def record_response(step: str, payload: dict[str, Any]) -> None:
        event = {"step": step, "payload": payload}
        response_events.append(event)
        with response_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    try:
        base_url = f"http://{args.gateway_host}:{gateway_port}"
        wait_for_http(
            f"{base_url}/health",
            timeout=float(args.server_start_timeout),
        )

        models = request_json(
            "GET",
            f"{base_url}/models",
            timeout=30.0,
        )
        record_response("models", models)
        print("MODELS", json.dumps(models, sort_keys=True), flush=True)

        started = request_json(
            "POST",
            f"{base_url}/start_session",
            build_start_session_payload(
                model_path=model_path,
                train_gpus=args.train_gpus,
                rollout_replicas=args.rollout_replicas,
                gpus_per_replica=args.gpus_per_replica,
                checkpoint_root=checkpoint_root,
            ),
            timeout=float(args.request_timeout),
        )
        record_response("start_session", started)
        print("START_SESSION", json.dumps(started, sort_keys=True), flush=True)
        session_id = str(started["session_id"])

        trajectory = request_json(
            "POST",
            f"{base_url}/start_trajectory",
            {"session_id": session_id},
            timeout=60.0,
        )
        record_response("start_trajectory", trajectory)
        print("START_TRAJECTORY", json.dumps(trajectory, sort_keys=True), flush=True)
        trajectory_id = str(trajectory["trajectory_id"])

        generated = request_json(
            "POST",
            f"{base_url}/generate",
            {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "messages": [
                    {"role": "user", "content": "Write a very short greeting."}
                ],
                "sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": 6,
                },
            },
            timeout=float(args.request_timeout),
        )
        record_response("generate", generated)
        print("GENERATE", json.dumps(generated, sort_keys=True), flush=True)
        assert str(generated["id"]).startswith("chatcmpl_")
        assert generated["object"] == "chat.completion"
        assert isinstance(generated["created"], int)
        assert isinstance(generated["model"], str) and generated["model"]
        assert generated["choices"]
        assert generated["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(generated["choices"][0]["message"]["content"], str)
        assert generated["metadata"]["session_id"] == session_id
        assert generated["metadata"]["trajectory_id"] == trajectory_id
        assert isinstance(generated["metadata"]["token_ids"], list)
        assert generated["metadata"]["rollout_model_version"]

        ended_trajectory = request_json(
            "POST",
            f"{base_url}/end_trajectory",
            {
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "final_reward": 1.0,
            },
            timeout=60.0,
        )
        record_response("end_trajectory", ended_trajectory)
        print(
            "END_TRAJECTORY", json.dumps(ended_trajectory, sort_keys=True), flush=True
        )

        ended_session = request_json(
            "POST",
            f"{base_url}/end_session",
            {"session_id": session_id},
            timeout=60.0,
        )
        record_response("end_session", ended_session)
        print("END_SESSION", json.dumps(ended_session, sort_keys=True), flush=True)

        final_models = request_json(
            "GET",
            f"{base_url}/models",
            timeout=30.0,
        )
        record_response("models_after", final_models)
        print("MODELS_AFTER", json.dumps(final_models, sort_keys=True), flush=True)
        summary = {
            "artifact_dir": str(artifact_dir),
            "config_path": str(config_path),
            "gateway_log_path": str(gateway_log_path),
            "response_log_path": str(response_log_path),
            "model_path": model_path,
            "base_url": base_url,
            "status": "ok",
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("ARTIFACT_DIR", str(artifact_dir), flush=True)
        print("SUMMARY_PATH", str(summary_path), flush=True)
        print("GATEWAY_REAL_E2E_OK", flush=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
