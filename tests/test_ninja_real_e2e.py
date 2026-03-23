# Copyright 2026 openforge

"""Run a real openforge.ninja smoke against a live gateway started by this script."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import openforge.ninja as ninja
import ray
from _script_test_utils import start_test_ray_cluster
from test_gateway_real_e2e import (
    ROOT,
    build_start_session_payload,
    get_free_port,
    make_artifact_dir,
    request_json,
    require_visible_gpus,
    resolve_model_path,
    wait_for_http,
    write_temp_config,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ninja smoke."""
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
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def main() -> int:
    """Run the real ninja smoke."""
    args = parse_args()
    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    gpu_ids = require_visible_gpus(total_requested_gpus)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    gateway_log_path = artifact_dir / "gateway_process.log"
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
        session_id = str(started["session_id"])

        responses: list[Any] = []

        @ninja.agent()
        def agent(client, *, prompt: str, reward: float) -> float:
            response = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_completion_tokens=8,
            )
            responses.append(response)
            return reward

        reward = agent(prompt="Say hello in four words.", reward=0.0)
        assert reward == 0.0
        response = responses[-1]
        assert response.metadata["session_id"] == session_id
        assert response.metadata["rollout_model_version"] == 0
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)

        policy_version = agent.policy_version()
        assert policy_version == 0

        request_json(
            "POST",
            f"{base_url}/end_session",
            {"session_id": session_id},
            timeout=120.0,
        )

        summary = {
            "artifact_dir": str(artifact_dir),
            "response": response,
            "policy_version": policy_version,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
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
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
