# Copyright 2026 openforge

"""Run a real openforge.ninja smoke against live Qwen + SGLang + FSDP2.

This script starts the real gateway through ``openforge.ninja.Gateway``, opens a
real session through ``openforge.ninja.Session``, and drives generation through a
registered ninja agent.

It validates the user-facing ninja API, not raw gateway HTTP calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from test_gateway_real_e2e import (
    DEFAULT_MODEL,
    get_free_port,
    make_artifact_dir,
    require_visible_gpus,
    resolve_model_path,
)

from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.gateway.types import RuntimeConfig
from openforge.ninja import register


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ninja smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def build_gateway_config(
    *,
    artifact_dir: Path,
    gateway_host: str,
    gateway_port: int,
    visible_gpus: int,
    cpus_per_node: int,
) -> GatewayServerConfig:
    """Build the gateway config passed to ``openforge.ninja.Gateway``."""
    return GatewayServerConfig(
        data=DataConfig(path=str(artifact_dir / "gateway.sqlite3")),
        gateway=GatewayConfig(host=gateway_host, port=gateway_port),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=visible_gpus,
            cpus_per_node=cpus_per_node,
        ),
    )


def build_runtime_config(
    *,
    model_path: str,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    checkpoint_root: Path,
) -> RuntimeConfig:
    """Build the runtime config passed to ``openforge.ninja.Session``."""
    return RuntimeConfig.model_validate(
        {
            "algo": {
                "name": "grpo",
                "clip_range": 0.2,
                "normalize_group_std": False,
                "entropy_coef": 0.0,
                "kl_coef": 0.0,
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
                        "lr": 1.0e-5,
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
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints": str(checkpoint_root),
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
        }
    )


def main() -> int:
    """Run the real ninja E2E smoke."""
    args = parse_args()
    total_requested_gpus = args.train_gpus + (
        args.rollout_replicas * args.gpus_per_replica
    )
    visible_gpus = require_visible_gpus(total_requested_gpus)
    model_path = resolve_model_path(args.model_path)
    gateway_port = args.gateway_port or get_free_port(args.gateway_host)
    artifact_dir = make_artifact_dir(args.artifact_dir)
    checkpoint_root = artifact_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"

    gateway_config = build_gateway_config(
        artifact_dir=artifact_dir,
        gateway_host=args.gateway_host,
        gateway_port=gateway_port,
        visible_gpus=visible_gpus,
        cpus_per_node=args.cpus_per_node,
    )
    runtime_config = build_runtime_config(
        model_path=model_path,
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        checkpoint_root=checkpoint_root,
    )

    responses: list[dict[str, Any]] = []

    @register(gateway_config, runtime_config)
    def agent(client, *, prompt: str, reward: float) -> float:
        response = client.generate(
            [{"role": "user", "content": prompt}],
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": 8,
            },
        )
        responses.append(response)
        return reward

    with agent.gateway():
        with agent.session():
            first_reward = agent(prompt="Say hello in four words.", reward=0.0)
            assert first_reward == 0.0
            first_response = responses[-1]
            first_version = str(first_response["metadata"]["rollout_model_version"])

            updated_response = None
            deadline = time.monotonic() + args.request_timeout
            while time.monotonic() < deadline:
                reward = agent(prompt="Say hello in four words.", reward=0.0)
                assert reward == 0.0
                candidate = responses[-1]
                version = str(candidate["metadata"]["rollout_model_version"])
                if version != first_version:
                    updated_response = candidate
                    break
                time.sleep(1.0)

            assert updated_response is not None, "rollout version never changed"

    summary = {
        "artifact_dir": str(artifact_dir),
        "first_response": first_response,
        "updated_response": updated_response,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
