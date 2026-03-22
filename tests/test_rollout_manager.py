# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import ray
from huggingface_hub import snapshot_download
from ray._private import worker as ray_worker

from _script_test_utils import require_free_gpu_ids, start_test_ray_cluster
from openforge.configs.models import OpenForgeConfig
from openforge.rollout.manager import RolloutManager, start_sglang_engines
from openforge.rollout.sglang.utils import get_local_gpu_id
from openforge.utils.ray import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    create_placement_groups,
    ray_noset_visible_devices,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=2)
    parser.add_argument("--gpus-per-replica", type=int, default=1)
    parser.add_argument("--cpus-per-rollout-replica", type=int, default=1)
    return parser.parse_args()
def resolve_model_path(model_path_or_id: str) -> str:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model_path_or_id, local_files_only=True)


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


def read_ray_logs(session_dir: str) -> str:
    logs_dir = Path(session_dir) / "logs"
    parts = []
    for pattern in ("*.out", "*.err"):
        for path in sorted(logs_dir.rglob(pattern)):
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def verify_engine_info(
    engine_info: dict[str, object],
    *,
    rollout_bundle_indices: list[int],
    rollout_gpu_ids: list[int],
    gpus_per_replica: int,
    session_dir: str,
) -> None:
    engine_specs = engine_info["engine_specs"]
    engine_workers = engine_info["engine_workers"]
    engine_addrs = engine_info["engine_addrs"]

    assert len(engine_specs) == len(engine_workers)

    post_specs = ray.get([worker.get_post_init_spec.remote() for worker in engine_workers])
    post_addrs = ray.get(
        [worker.get_post_launch_addr.remote() for worker in engine_workers]
    )
    hosts = ray.get([worker.get_ip_addr.remote() for worker in engine_workers])
    log_text = read_ray_logs(session_dir)

    ports = set()
    for rank, (spec, post_spec, post_addr, host) in enumerate(
        zip(engine_specs, post_specs, post_addrs, hosts, strict=True)
    ):
        offset = rank * gpus_per_replica
        addr = engine_addrs[spec.engine_name]

        assert spec.engine_name == f"regular-{rank}"
        assert spec.engine_rank == rank
        assert spec.gpu_rank_offset == offset
        assert spec.base_gpu_id == rollout_gpu_ids[offset]
        assert spec.bundle_indices == rollout_bundle_indices
        assert spec.gpu_ids == rollout_gpu_ids
        assert post_spec == spec
        assert post_addr == addr
        assert addr.port not in ports
        assert addr.nccl_port == addr.port + 1
        assert addr.dist_init_addr == f"{host}:{addr.port + 2}"
        assert f"Launching SGLang server on {addr}" in log_text
        ports.add(addr.port)


def assert_completion_length(payload: dict[str, object], *, expected_tokens: int) -> None:
    output_ids = payload.get("output_ids")
    if not isinstance(output_ids, list):
        raise RuntimeError(f"generate payload missing output_ids: {payload!r}")
    if len(output_ids) != expected_tokens:
        raise RuntimeError(
            f"expected {expected_tokens} completion tokens, got {len(output_ids)}: {payload!r}"
        )

    meta_info = payload.get("meta_info")
    if isinstance(meta_info, dict):
        completion_tokens = meta_info.get("completion_tokens")
        if completion_tokens is not None and int(completion_tokens) != expected_tokens:
            raise RuntimeError(
                "sampling_params.max_new_tokens was not respected: "
                f"expected {expected_tokens}, got completion_tokens={completion_tokens}"
            )


def validate_engine_sampling_params(engine_worker: object) -> None:
    prompt = "Validate sampling params on the direct engine path."
    for max_new_tokens in (1, 4):
        payload = ray.get(
            engine_worker.generate.remote(
                text=prompt,
                sampling_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                return_logprob=True,
            )
        )
        assert_completion_length(payload, expected_tokens=max_new_tokens)


def validate_manager_sampling_params(manager: RolloutManager) -> None:
    prompt = "Validate sampling params through the rollout manager."
    for max_new_tokens in (2, 5):
        payload = manager.router.generate(
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            text=prompt,
            return_logprob=True,
        )
        assert_completion_length(payload, expected_tokens=max_new_tokens)


def main() -> int:
    with patch.dict(
        os.environ,
        {env_var: "1" for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        clear=False,
    ):
        assert ray_noset_visible_devices()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, clear=False):
        assert get_local_gpu_id(4) == 0
        assert get_local_gpu_id(5) == 1
        assert get_local_gpu_id(7) == 3

    args = parse_args()
    total_requested = args.train_gpus + args.rollout_replicas * args.gpus_per_replica
    gpu_ids = require_free_gpu_ids(total_requested)
    model_path = resolve_model_path(args.model_path)
    cfg = build_cfg(
        model_path=model_path,
        visible_gpus=len(gpu_ids),
        train_gpus=args.train_gpus,
        rollout_replicas=args.rollout_replicas,
        gpus_per_replica=args.gpus_per_replica,
        cpus_per_rollout_replica=args.cpus_per_rollout_replica,
    )

    start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=max(len(gpu_ids), 1),
    )

    session_dir = ray_worker._global_node.get_session_dir_path()
    placement_groups = create_placement_groups(cfg)
    actor_pg, _, _ = placement_groups["actor"]
    _, rollout_bundle_indices, rollout_gpu_ids = placement_groups["rollout"]

    engine_info = start_sglang_engines(cfg, placement_groups)
    try:
        verify_engine_info(
            engine_info,
            rollout_bundle_indices=rollout_bundle_indices,
            rollout_gpu_ids=rollout_gpu_ids,
            gpus_per_replica=args.gpus_per_replica,
            session_dir=session_dir,
        )
        validate_engine_sampling_params(engine_info["engine_workers"][0])
    finally:
        for worker in engine_info["engine_workers"]:
            ray.kill(worker)

    manager = RolloutManager(cfg, placement_groups)
    manager.initialize(router_ip="127.0.0.1", policy="round_robin")
    validate_manager_sampling_params(manager)
    manager.shutdown()

    ray.util.remove_placement_group(actor_pg)
    ray.shutdown()

    print(
        "SUCCESS "
        f"replicas={args.rollout_replicas} "
        f"gpus_per_replica={args.gpus_per_replica} "
        f"engines={[spec.engine_name for spec in engine_info['engine_specs']]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
