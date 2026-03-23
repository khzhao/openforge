# Copyright 2026 openforge

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

# Keep every manager-sync entry point on the same CUDA/NCCL/FlashInfer setup.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import ray
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from _script_test_utils import require_free_gpu_ids, start_test_ray_cluster
from openforge.configs.models import OpenForgeConfig
from openforge.runtime import (
    create_rollout_manager,
    create_train_manager,
    register_rollout,
)
from openforge.utils.networking import get_free_port
from openforge.utils.ray import create_placement_groups

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYNC_MODES = ("disk", "distributed")
def resolve_local_model_path(model_path_or_id: str) -> str:
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
    checkpoint_root = tempfile.mkdtemp(prefix="openforge-fsdp2-manager-e2e-")
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
                        "lr": 1e-2,
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


def build_sample(model_path: str) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode(
        "OpenForge FSDP2 Manager end-to-end weight sync test.",
        add_special_tokens=True,
    )
    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or token_ids[-1]
    while len(token_ids) < 16:
        token_ids.append(pad_token_id)
    token_ids = token_ids[:32]

    tokens = torch.tensor(token_ids, dtype=torch.long)
    return {
        "tokens": tokens,
        "position_ids": torch.arange(tokens.numel(), dtype=torch.long),
        "advantages": torch.linspace(0.25, 1.0, tokens.numel(), dtype=torch.float32),
        "loss_mask": torch.ones(tokens.numel() - 1, dtype=torch.float32),
        "rollout_log_probs": torch.zeros(tokens.numel() - 1, dtype=torch.float32),
        "lengths": torch.tensor(tokens.numel(), dtype=torch.long),
    }


def assert_snapshot_success(
    responses: Sequence[dict[str, object]],
    *,
    context: str,
) -> None:
    for index, response in enumerate(responses):
        success = response.get("success")
        status = response.get("status")
        if success is True or status == "ok":
            continue
        raise RuntimeError(f"{context} failed on rollout engine {index}: {response}")


def assert_weights_changed(
    responses: Sequence[dict[str, object]],
    *,
    context: str,
) -> None:
    for index, response in enumerate(responses):
        if response.get("success") is True:
            raise RuntimeError(
                f"{context}: rollout engine {index} still matches its snapshot: {response}"
            )
        if response.get("success") is None and response.get("status") is None:
            raise RuntimeError(
                f"{context}: rollout engine {index} returned no success indicator: {response}"
            )


def assert_policy_versions(
    rollout_workers: Sequence[object],
    *,
    expected_version: int,
) -> None:
    versions = ray.get([worker.get_weight_version.remote() for worker in rollout_workers])
    expected = str(expected_version)
    for index, version in enumerate(versions):
        if version != expected:
            raise RuntimeError(
                f"rollout engine {index} expected policy_version={expected}, got {version!r}"
            )


def assert_trainer_weights_changed(
    before: list[list[tuple[str, torch.Tensor]]],
    after: list[list[tuple[str, torch.Tensor]]],
) -> None:
    after_tensors = {
        name: tensor
        for bucket in after
        for name, tensor in bucket
        if tensor.is_floating_point()
    }
    for bucket in before:
        for name, tensor in bucket:
            if not tensor.is_floating_point():
                continue
            updated = after_tensors.get(name)
            if updated is not None and not torch.equal(tensor, updated):
                return
    raise RuntimeError("trainer weights did not change after the training steps")


def run_weight_sync_e2e(
    *,
    model_path: str,
    train_gpus: int,
    rollout_replicas: int,
    gpus_per_replica: int,
    cpus_per_rollout_replica: int,
    base_policy_version: int,
    sync_modes: Sequence[str],
) -> int:
    normalized_sync_modes = tuple(sync_modes)
    invalid_modes = sorted(set(normalized_sync_modes) - set(SYNC_MODES))
    if invalid_modes:
        raise ValueError(f"unsupported sync modes: {invalid_modes}")

    total_requested = train_gpus + rollout_replicas * gpus_per_replica
    gpu_ids = require_free_gpu_ids(total_requested)
    resolved_model_path = resolve_local_model_path(model_path)
    cfg = build_cfg(
        model_path=resolved_model_path,
        visible_gpus=len(gpu_ids),
        train_gpus=train_gpus,
        rollout_replicas=rollout_replicas,
        gpus_per_replica=gpus_per_replica,
        cpus_per_rollout_replica=cpus_per_rollout_replica,
    )

    start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=max(len(gpu_ids), 1),
    )

    placement_groups = create_placement_groups(cfg)
    train_manager = None
    rollout_manager = None
    try:
        train_manager = create_train_manager(
            cfg,
            master_addr="127.0.0.1",
            master_port=get_free_port(start=20000),
            placement_groups=placement_groups,
        )

        rollout_manager = create_rollout_manager(
            cfg,
            placement_groups,
            router_ip="127.0.0.1",
            router_port=get_free_port(start=30000),
        )
        register_rollout(train_manager, rollout_manager)
        rollout_workers = rollout_manager.engine_workers
        if len(rollout_workers) != rollout_replicas:
            raise RuntimeError(
                "manager did not return the expected number of rollout engines: "
                f"{len(rollout_workers)} != {rollout_replicas}"
            )

        sample = build_sample(resolved_model_path)
        rank_minibatches = [
            {name: tensor.unsqueeze(0) for name, tensor in sample.items()}
            for _ in range(cfg.train.num_workers)
        ]
        global_step = 0
        for mode_index, mode in enumerate(normalized_sync_modes, start=1):
            snapshot_responses = ray.get(
                [worker.check_weights.remote(action="snapshot") for worker in rollout_workers]
            )
            assert_snapshot_success(snapshot_responses, context=f"{mode}/snapshot")

            pre_step_weights = train_manager.build_tensor_buckets(bucket_bytes=64 << 20)
            step_results = []
            for _ in range(3):
                global_step += 1
                step_results = train_manager.step(
                    rank_minibatches,
                    global_step=global_step,
                )
            post_step_weights = train_manager.build_tensor_buckets(
                bucket_bytes=64 << 20
            )
            assert_trainer_weights_changed(pre_step_weights, post_step_weights)

            policy_version = base_policy_version + mode_index
            train_manager.sync_rollout_weights(
                policy_version=policy_version,
                bucket_bytes=(4 << 20) if mode == "distributed" else (256 << 20),
                mode=mode,
            )

            assert_policy_versions(
                rollout_workers,
                expected_version=policy_version,
            )
            compare_responses = ray.get(
                [worker.check_weights.remote(action="compare") for worker in rollout_workers]
            )
            assert_weights_changed(compare_responses, context=f"{mode}/compare")
            print(
                f"[{mode}] SUCCESS engines={len(rollout_workers)} "
                f"train_metrics={step_results[0].metrics}"
            )

        print(
            "SUCCESS "
            f"modes={','.join(normalized_sync_modes)} "
            f"engines={len(rollout_workers)} "
            f"train_gpus={train_gpus} "
            f"rollout_tp={gpus_per_replica}"
        )
        return 0
    finally:
        if rollout_manager is not None:
            try:
                rollout_manager.shutdown()
            except Exception:
                pass
        try:
            if train_manager is not None:
                train_manager.shutdown()
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()
