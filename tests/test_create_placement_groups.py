# Copyright 2026 openforge

from __future__ import annotations

import argparse
import ray

from _script_test_utils import require_free_gpu_ids, start_test_ray_cluster
from openforge.configs.models import OpenForgeConfig
from openforge.utils.ray import CanaryWorker, create_placement_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=int, default=1)
    return parser.parse_args()
def build_cfg(*, visible_gpus: int, train_gpus: int, rollout_gpus: int) -> OpenForgeConfig:
    return OpenForgeConfig.model_validate(
        {
            "data": {},
            "gateway": {"host": "127.0.0.1", "port": 0},
            "model": {
                "model_name_or_path": "unused",
                "tokenizer_name_or_path": "unused",
                "attn_implementation": "sdpa",
            },
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
                "max_rollout_policy_lag": 0,
                "checkpoints": "/tmp/openforge-test-checkpoints",
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
                        "replicas": 1,
                        "num_gpus_per_replica": rollout_gpus,
                        "num_cpus_per_replica": 1,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": rollout_gpus,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "enable_memory_saver": False,
                    }
                ],
            },
        }
    )


def verify_mapping(name: str, pg, bundle_indices: list[int], expected_gpu_ids: list[int]) -> None:
    workers = []
    try:
        for bundle_index in bundle_indices:
            workers.append(
                CanaryWorker.options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index,
                ).remote()
            )
        placements = ray.get([worker.get_ip_and_gpu_id.remote() for worker in workers])
    finally:
        for worker in workers:
            ray.kill(worker)

    actual_gpu_ids = [gpu_id for _node_ip, gpu_id in placements]
    if actual_gpu_ids != expected_gpu_ids:
        raise RuntimeError(
            f"{name} GPU mapping mismatch: expected {expected_gpu_ids}, got {actual_gpu_ids}"
        )

    print(f"{name}: bundles={bundle_indices} gpus={actual_gpu_ids}")


def main() -> int:
    args = parse_args()
    if args.train_gpus <= 0 or args.rollout_gpus <= 0:
        raise ValueError("--train-gpus and --rollout-gpus must both be > 0")

    total_requested = args.train_gpus + args.rollout_gpus
    gpu_ids = require_free_gpu_ids(total_requested)
    cfg = build_cfg(
        visible_gpus=len(gpu_ids),
        train_gpus=args.train_gpus,
        rollout_gpus=args.rollout_gpus,
    )

    start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=max(len(gpu_ids), 1),
    )

    actor_pg = None
    try:
        placement_groups = create_placement_groups(cfg)
        actor_pg, actor_bundle_indices, actor_gpu_ids = placement_groups["actor"]
        rollout_pg, rollout_bundle_indices, rollout_gpu_ids = placement_groups["rollout"]

        if actor_pg != rollout_pg:
            raise RuntimeError("actor and rollout slices should share one placement group")
        if len(actor_bundle_indices) != args.train_gpus:
            raise RuntimeError("actor bundle count does not match requested train GPUs")
        if len(rollout_bundle_indices) != args.rollout_gpus:
            raise RuntimeError(
                "rollout bundle count does not match requested rollout GPUs"
            )
        if set(actor_bundle_indices) & set(rollout_bundle_indices):
            raise RuntimeError("actor and rollout bundle indices overlap")
        if len(set(actor_gpu_ids + rollout_gpu_ids)) != total_requested:
            raise RuntimeError("physical GPU ids are not unique across the placement group")

        verify_mapping("actor", actor_pg, actor_bundle_indices, actor_gpu_ids)
        verify_mapping("rollout", rollout_pg, rollout_bundle_indices, rollout_gpu_ids)

        print(
            f"SUCCESS train_gpus={args.train_gpus} rollout_gpus={args.rollout_gpus}"
        )
        return 0
    finally:
        if actor_pg is not None:
            ray.util.remove_placement_group(actor_pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
