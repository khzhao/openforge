# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import ray
import requests
from huggingface_hub import snapshot_download

from _script_test_utils import require_free_gpu_ids, start_test_ray_cluster
from openforge.configs.models import OpenForgeConfig
from openforge.rollout.manager import RolloutManager
from openforge.utils.ray import create_placement_groups

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def resolve_model_path(model_path_or_id: str) -> str:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model_path_or_id)


def build_cfg(
    *,
    model_path: str,
    visible_gpus: int,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    cpus_per_rollout_replica: int,
) -> OpenForgeConfig:
    return OpenForgeConfig.model_validate(
        {
            "data": {},
            "gateway": {"host": "127.0.0.1", "port": 0},
            "model": {
                "model_name_or_path": model_path,
                "tokenizer_name_or_path": model_path,
                "attn_implementation": "sdpa",
            },
            "algo": {"name": "grpo", "kl_coef": 0.0},
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": visible_gpus,
                "cpus_per_node": max(visible_gpus, 1),
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
                        "lr": 1e-5,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
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
                "ppo_epochs": 1,
                "max_rollout_policy_lag": 0,
                "checkpoints": tempfile.mkdtemp(
                    prefix="openforge-rollout-manager-live-"
                ),
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
                    "repetition_penalty": 1.0,
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
                        "num_cpus_per_replica": cpus_per_rollout_replica,
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


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    timeout: float,
) -> dict[str, object]:
    response = requests.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    assert isinstance(body, dict), body
    return body


def assert_completion(payload: dict[str, object], *, expected_tokens: int) -> int:
    output_ids = payload.get("output_ids")
    assert isinstance(output_ids, list), payload
    assert len(output_ids) == expected_tokens, payload
    meta_info = payload.get("meta_info")
    assert isinstance(meta_info, dict), payload
    weight_version = meta_info.get("weight_version")
    assert weight_version is not None, payload
    return int(weight_version)


def main() -> int:
    args = parse_args()
    total_requested = args.train_gpus + args.rollout_replicas * args.gpus_per_replica
    gpu_ids = require_free_gpu_ids(total_requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    print(f"visible_gpus={gpu_ids}", flush=True)
    model_path = resolve_model_path(args.model_path)
    print(f"model_path={model_path}", flush=True)

    cfg = build_cfg(
        model_path=model_path,
        visible_gpus=len(gpu_ids),
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
    )
    start_test_ray_cluster(gpu_ids=gpu_ids, num_cpus=max(total_requested, 1))

    placement_groups = create_placement_groups(cfg)
    actor_pg, _, _ = placement_groups["actor"]
    manager = RolloutManager(cfg, placement_groups)
    try:
        print("starting rollout manager", flush=True)
        manager.initialize(router_ip="127.0.0.1")
        print(f"router_url={manager.router_url}", flush=True)

        status = request_json(
            "GET",
            f"{manager.router_url}/status",
            timeout=10.0,
        )
        workers = status.get("workers")
        assert isinstance(workers, dict), status
        assert len(workers) == args.rollout_replicas, status
        assert all(bool(worker["healthy"]) for worker in workers.values()), status
        print(f"workers={list(workers)}", flush=True)

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
        }

        print("sending router generate request", flush=True)
        first = request_json(
            "POST",
            f"{manager.router_url}/generate",
            payload={
                "text": "Say hello in one short sentence.",
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            timeout=args.request_timeout,
        )
        version = assert_completion(first, expected_tokens=args.max_new_tokens)
        print(f"first_weight_version={version}", flush=True)

        trajectory_id = "traj-router-live"
        print("checking pinned trajectory routing", flush=True)
        pinned_first = request_json(
            "POST",
            f"{manager.router_url}/generate",
            payload={
                "trajectory_id": trajectory_id,
                "text": "Pinned trajectory turn one.",
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            timeout=args.request_timeout,
        )
        pinned_second = request_json(
            "POST",
            f"{manager.router_url}/generate",
            payload={
                "trajectory_id": trajectory_id,
                "text": "Pinned trajectory turn two.",
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            timeout=args.request_timeout,
        )
        pinned_first_version = assert_completion(
            pinned_first,
            expected_tokens=args.max_new_tokens,
        )
        pinned_second_version = assert_completion(
            pinned_second,
            expected_tokens=args.max_new_tokens,
        )
        assert pinned_first_version == pinned_second_version

        pinned_status = request_json(
            "GET",
            f"{manager.router_url}/status",
            timeout=10.0,
        )
        trajectories = pinned_status.get("trajectories")
        assert isinstance(trajectories, dict), pinned_status
        trajectory = trajectories.get(trajectory_id)
        assert isinstance(trajectory, dict), pinned_status
        assert int(trajectory["expected_version"]) == pinned_first_version
        assert isinstance(trajectory["worker_name"], str)

        request_json(
            "POST",
            f"{manager.router_url}/release_trajectories",
            payload={"trajectory_ids": [trajectory_id]},
            timeout=10.0,
        )
        final_status = request_json(
            "GET",
            f"{manager.router_url}/status",
            timeout=10.0,
        )
        final_trajectories = final_status.get("trajectories")
        assert isinstance(final_trajectories, dict), final_status
        assert trajectory_id not in final_trajectories

        print("PASS rollout manager router live smoke", flush=True)
        return 0
    finally:
        manager.shutdown()
        ray.util.remove_placement_group(actor_pg)
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
