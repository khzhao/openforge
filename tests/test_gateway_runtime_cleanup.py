# Copyright 2026 openforge

"""Verify that ending a session tears down session-owned runtime processes."""

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

import psutil
import requests
from huggingface_hub import snapshot_download
import torch

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "test_scripts" / "logs"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the runtime-cleanup smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--server-start-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--cleanup-timeout", type=float, default=30.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--artifact-dir", default=None)
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
    gateway_host: str,
    gateway_port: int,
    visible_gpus: int,
    cpus_per_node: int,
) -> Path:
    """Write a temporary gateway-server YAML config for the live smoke."""
    root = Path(tempfile.mkdtemp(prefix="openforge-gateway-cleanup-"))
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
    return Path(tempfile.mkdtemp(prefix="gateway-runtime-cleanup-", dir=LOG_ROOT))


def runtime_process_rows(gateway_pid: int) -> list[dict[str, Any]]:
    """Return interesting or zombie descendants under the gateway process."""
    try:
        gateway = psutil.Process(gateway_pid)
    except psutil.NoSuchProcess:
        return []

    rows: list[dict[str, Any]] = []
    interesting_tokens = (
        "sglang::",
        "gcs_server",
        "raylet",
        "ray::DashboardAgent",
        "ray::RuntimeEnvAgent",
        "ray::TrainWorker",
        "ray::Engine",
        "ray::IDLE",
        "resource_tracker",
        "multiprocessing.spawn",
    )
    for child in gateway.children(recursive=True):
        try:
            status = child.status()
            cmd = " ".join(child.cmdline())
            name = child.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if status == psutil.STATUS_ZOMBIE or any(
            token in cmd for token in interesting_tokens
        ):
            rows.append(
                {
                    "pid": child.pid,
                    "ppid": child.ppid(),
                    "status": status,
                    "name": name,
                    "cmd": cmd,
                }
            )
    rows.sort(key=lambda row: row["pid"])
    return rows


def alive_process_rows(pids: set[int]) -> list[dict[str, Any]]:
    """Return still-alive processes for the provided pid set."""
    rows: list[dict[str, Any]] = []
    for pid in sorted(pids):
        if not psutil.pid_exists(pid):
            continue
        try:
            process = psutil.Process(pid)
            rows.append(
                {
                    "pid": pid,
                    "ppid": process.ppid(),
                    "status": process.status(),
                    "name": process.name(),
                    "cmd": " ".join(process.cmdline()),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return rows


def main() -> int:
    """Run the live runtime-cleanup check."""
    args = parse_args()
    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    visible_gpus = require_visible_gpus(total_requested_gpus)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    gateway_log_path = artifact_dir / "gateway_process.log"
    summary_path = artifact_dir / "summary.json"
    for path in (gateway_log_path, summary_path):
        if path.exists():
            path.unlink()
    checkpoint_root = str(artifact_dir / "checkpoints")
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    config_path = write_temp_config(
        gateway_host=args.gateway_host,
        gateway_port=gateway_port,
        visible_gpus=visible_gpus,
        cpus_per_node=args.cpus_per_node,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.Popen(
        [sys.executable, "-m", "openforge.gateway.main", "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    observed_runtime_pids: set[int] = set()

    try:
        base_url = f"http://{args.gateway_host}:{gateway_port}"
        wait_for_http(
            f"{base_url}/health",
            timeout=float(args.server_start_timeout),
        )

        for cycle in range(args.cycles):
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
            session_id = str(started["session_id"])
            trajectory = request_json(
                "POST",
                f"{base_url}/start_trajectory",
                {"session_id": session_id},
                timeout=60.0,
            )
            trajectory_id = str(trajectory["trajectory_id"])

            request_json(
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

            before_end = runtime_process_rows(proc.pid)
            print(
                f"RUNTIME_PROCESSES_BEFORE_END_CYCLE_{cycle}",
                json.dumps(before_end, indent=2),
                flush=True,
            )
            if not before_end:
                raise RuntimeError(
                    "expected session-owned runtime processes before ending the session"
                )
            observed_runtime_pids.update(row["pid"] for row in before_end)

            request_json(
                "POST",
                f"{base_url}/end_trajectory",
                {
                    "session_id": session_id,
                    "trajectory_id": trajectory_id,
                    "final_reward": 1.0,
                },
                timeout=60.0,
            )
            request_json(
                "POST",
                f"{base_url}/end_session",
                {"session_id": session_id},
                timeout=120.0,
            )

            deadline = time.monotonic() + float(args.cleanup_timeout)
            after_end = runtime_process_rows(proc.pid)
            while after_end and time.monotonic() < deadline:
                time.sleep(1.0)
                after_end = runtime_process_rows(proc.pid)

            print(
                f"RUNTIME_PROCESSES_AFTER_END_CYCLE_{cycle}",
                json.dumps(after_end, indent=2),
                flush=True,
            )
            if after_end:
                raise RuntimeError(
                    "session-owned runtime processes survived /end_session"
                )

        summary_path.write_text(
            json.dumps(
                {
                    "artifact_dir": str(artifact_dir),
                    "gateway_log_path": str(gateway_log_path),
                    "config_path": str(config_path),
                    "status": "ok",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print("GATEWAY_RUNTIME_CLEANUP_OK", flush=True)
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
            gateway_log_path.write_text(proc.stdout.read(), encoding="utf-8")
        deadline = time.monotonic() + float(args.cleanup_timeout)
        surviving = alive_process_rows(observed_runtime_pids)
        while surviving and time.monotonic() < deadline:
            time.sleep(1.0)
            surviving = alive_process_rows(observed_runtime_pids)
        if surviving:
            print(
                "RUNTIME_PROCESSES_AFTER_GATEWAY_EXIT",
                json.dumps(surviving, indent=2),
                flush=True,
            )
            raise RuntimeError("runtime processes survived gateway shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
