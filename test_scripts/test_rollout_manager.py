# Copyright 2026 openforge

from __future__ import annotations

import argparse

import ray
import torch
from loguru import logger

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.manager import RolloutManager, start_sglang_engines
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.types import EngineAddr
from openforge.utils.ray import create_placement_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=2)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    return parser.parse_args()


def require_visible_gpus(min_count: int) -> int:
    visible = torch.cuda.device_count()
    if visible < min_count:
        raise RuntimeError(
            f"Expected at least {min_count} visible GPUs, found {visible}. "
            "Adjust CUDA_VISIBLE_DEVICES before running this script."
        )
    return visible


def build_cfg(
    *,
    visible_gpus: int,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    cpus_per_rollout_replica: int,
) -> OpenForgeConfig:
    rollout_gpus = rollout_replicas * gpus_per_replica
    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "dummy"},
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
                        "replicas": rollout_replicas,
                        "num_gpus_per_replica": gpus_per_replica,
                        "num_cpus_per_replica": cpus_per_rollout_replica,
                        "parallelism": {
                            "data_parallel_size": gpus_per_replica,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                    }
                ],
            },
        }
    )


def verify_engine_info(
    engine_info: dict[str, object],
    *,
    rollout_bundle_indices: list[int],
    rollout_gpu_ids: list[int],
    rollout_replicas: int,
    gpus_per_replica: int,
) -> None:
    engine_specs = list(engine_info["engine_specs"])
    engine_workers = list(engine_info["engine_workers"])
    engine_addrs = dict(engine_info["engine_addrs"])

    if len(engine_specs) != rollout_replicas:
        raise RuntimeError(
            f"Expected {rollout_replicas} engine specs, got {len(engine_specs)}"
        )
    if len(engine_workers) != rollout_replicas:
        raise RuntimeError(
            f"Expected {rollout_replicas} engine workers, got {len(engine_workers)}"
        )

    expected_names = [f"regular-{rank}" for rank in range(rollout_replicas)]
    actual_names = [spec.name for spec in engine_specs]
    if actual_names != expected_names:
        raise RuntimeError(f"Engine name mismatch: {actual_names} != {expected_names}")

    hosts = ray.get([worker.get_ip_addr.remote() for worker in engine_workers])
    post_init_specs = ray.get(
        [worker.get_post_init_spec.remote() for worker in engine_workers]
    )
    post_launch_addrs = ray.get(
        [worker.get_post_launch_addr.remote() for worker in engine_workers]
    )

    seen_ports = set()
    for rank, (spec, host, post_init_spec, post_launch_addr) in enumerate(
        zip(
            engine_specs,
            hosts,
            post_init_specs,
            post_launch_addrs,
            strict=True,
        )
    ):
        expected_offset = rank * gpus_per_replica
        expected_bundle_index = rollout_bundle_indices[expected_offset]
        expected_base_gpu_id = rollout_gpu_ids[expected_offset]
        addr = engine_addrs[spec.name]

        if spec.engine_rank != rank:
            raise RuntimeError(
                f"Engine rank mismatch for {spec.name}: "
                f"{spec.engine_rank} != {rank}"
            )
        if spec.gpu_rank_offset != expected_offset:
            raise RuntimeError(
                f"GPU rank offset mismatch for {spec.name}: "
                f"{spec.gpu_rank_offset} != {expected_offset}"
            )
        if spec.bundle_indices[spec.gpu_rank_offset] != expected_bundle_index:
            raise RuntimeError(
                f"Scheduled bundle mismatch for {spec.name}: "
                f"{spec.bundle_indices[spec.gpu_rank_offset]} != {expected_bundle_index}"
            )
        if spec.base_gpu_id != expected_base_gpu_id:
            raise RuntimeError(
                f"Base GPU mismatch for {spec.name}: "
                f"{spec.base_gpu_id} != {expected_base_gpu_id}"
            )
        if list(spec.bundle_indices) != rollout_bundle_indices:
            raise RuntimeError(
                f"Bundle slice mismatch for {spec.name}: "
                f"{spec.bundle_indices} != {rollout_bundle_indices}"
            )
        if list(spec.gpu_ids) != rollout_gpu_ids:
            raise RuntimeError(
                f"GPU id slice mismatch for {spec.name}: "
                f"{spec.gpu_ids} != {rollout_gpu_ids}"
            )
        if post_init_spec.name != spec.name:
            raise RuntimeError(
                f"Worker init spec mismatch for {spec.name}: "
                f"{post_init_spec.name} != {spec.name}"
            )
        if post_launch_addr != addr:
            raise RuntimeError(
                f"Worker launch addr mismatch for {spec.name}: "
                f"{post_launch_addr} != {addr}"
            )

        if not host:
            raise RuntimeError(f"Empty host for {spec.name}")
        if addr.port in seen_ports:
            raise RuntimeError(f"Duplicate engine port allocated: {addr.port}")
        seen_ports.add(addr.port)
        if addr.nccl_port != addr.port + 1:
            raise RuntimeError(
                f"NCCL port mismatch for {spec.name}: {addr.nccl_port} != {addr.port + 1}"
            )
        if addr.dist_init_addr != f"{host}:{addr.port + 2}":
            raise RuntimeError(
                f"dist_init_addr mismatch for {spec.name}: "
                f"{addr.dist_init_addr} != {host}:{addr.port + 2}"
            )


def verify_dummy_launch_log() -> None:
    engine = Engine()
    addr = EngineAddr(
        host="127.0.0.1",
        port=15000,
        nccl_port=15001,
        dist_init_addr="127.0.0.1:15002",
    )

    messages = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        engine.launch(addr)
    finally:
        logger.remove(sink_id)

    expected_message = f"Launching SGLang server on {addr}"
    normalized_messages = [message.rstrip("\n") for message in messages]
    if expected_message not in normalized_messages:
        raise RuntimeError(
            f"Expected launch log {expected_message!r}, got {normalized_messages!r}"
        )


def main() -> int:
    args = parse_args()
    if args.train_gpus <= 0:
        raise ValueError("--train-gpus must be > 0")
    if args.rollout_replicas <= 0:
        raise ValueError("--rollout-replicas must be > 0")
    if args.gpus_per_replica <= 0:
        raise ValueError("--gpus-per-replica must be > 0")
    if args.cpus_per_rollout_replica < 0:
        raise ValueError("--cpus-per-rollout-replica must be >= 0")

    rollout_gpus = args.rollout_replicas * args.gpus_per_replica
    total_requested = args.train_gpus + rollout_gpus
    visible_gpus = require_visible_gpus(total_requested)
    cfg = build_cfg(
        visible_gpus=visible_gpus,
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
    )

    manager = None
    direct_engine_info = None
    actor_pg = None
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True, include_dashboard=False)

        placement_groups = create_placement_groups(cfg)
        actor_pg, _, _ = placement_groups["actor"]
        rollout_pg, rollout_bundle_indices, rollout_gpu_ids = placement_groups["rollout"]
        if actor_pg != rollout_pg:
            raise RuntimeError("Expected train and rollout to share one placement group")

        manager = RolloutManager.remote(cfg, placement_groups)
        ray.get(manager.initialize.remote())
        ray.get(manager.shutdown.remote())

        direct_engine_info = start_sglang_engines(cfg, placement_groups)
        verify_engine_info(
            direct_engine_info,
            rollout_bundle_indices=rollout_bundle_indices,
            rollout_gpu_ids=rollout_gpu_ids,
            rollout_replicas=args.rollout_replicas,
            gpus_per_replica=args.gpus_per_replica,
        )
        verify_dummy_launch_log()

        print(
            "SUCCESS "
            f"replicas={args.rollout_replicas} "
            f"gpus_per_replica={args.gpus_per_replica} "
            f"engines={[spec.name for spec in direct_engine_info['engine_specs']]}"
        )
        return 0
    finally:
        if manager is not None:
            try:
                ray.get(manager.shutdown.remote())
            except Exception:
                pass
            try:
                ray.kill(manager)
            except Exception:
                pass
        if direct_engine_info is not None:
            for worker in direct_engine_info["engine_workers"]:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
        if actor_pg is not None:
            try:
                ray.util.remove_placement_group(actor_pg)
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
