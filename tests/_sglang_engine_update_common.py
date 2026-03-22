# Copyright 2026 openforge

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from _script_test_utils import require_free_gpu_ids, start_test_ray_cluster
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.manager import start_sglang_engines
from openforge.utils.distributed import init_custom_process_group
from openforge.utils.networking import get_free_port
from openforge.utils.ray import create_placement_groups

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def configure_nccl_env() -> None:
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def require_visible_gpus(min_count: int) -> int:
    return len(require_free_gpu_ids(min_count))


def resolve_local_model_path(model_path_or_id: str) -> str:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model_path_or_id, local_files_only=True)


def build_cfg(
    *,
    model_path: str,
    visible_gpus: int,
    train_gpus: int = 1,
) -> OpenForgeConfig:
    checkpoint_root = tempfile.mkdtemp(prefix="openforge-sglang-engine-update-")
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
                        "replicas": 1,
                        "num_gpus_per_replica": 1,
                        "num_cpus_per_replica": 1,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "fsdp_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "enable_memory_saver": False,
                    }
                ],
            },
        }
    )


def start_single_engine(
    *,
    model_path: str,
    train_gpus: int = 1,
) -> tuple[object, OpenForgeConfig, dict[str, tuple[object, list[int], list[int]]]]:
    gpu_ids = require_free_gpu_ids(train_gpus + 1)
    cfg = build_cfg(
        model_path=model_path,
        visible_gpus=len(gpu_ids),
        train_gpus=train_gpus,
    )
    start_test_ray_cluster(
        gpu_ids=gpu_ids,
        num_cpus=max(len(gpu_ids), 1),
    )
    placement_groups = create_placement_groups(cfg)
    engine_info = start_sglang_engines(cfg, placement_groups)
    return engine_info["engine_workers"][0], cfg, placement_groups


def shutdown_single_engine(
    worker: object,
    placement_groups: dict[str, tuple[object, list[int], list[int]]],
) -> None:
    try:
        ray.get(worker.stop.remote())
    finally:
        try:
            ray.kill(worker)
        except Exception:
            pass
        try:
            actor_pg, _, _ = placement_groups["actor"]
            ray.util.remove_placement_group(actor_pg)
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()


def load_source_model(model_path: str, *, device: torch.device) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=dtype,
    )
    model.eval()
    return model.to(device)


def select_update_tensor(
    model: AutoModelForCausalLM,
    *,
    max_numel: int = 1_000_000,
) -> tuple[str, torch.Tensor]:
    fallback: tuple[str, torch.Tensor] | None = None
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        if fallback is None:
            fallback = (name, tensor)
        if tensor.numel() > max_numel:
            continue
        updated = tensor.detach().clone().contiguous()
        updated.view(-1)[0] = updated.view(-1)[0] + updated.new_tensor(1e-3)
        return name, updated
    if fallback is None:
        raise RuntimeError("Could not find a floating-point tensor to update")
    name, tensor = fallback
    updated = tensor.detach().clone().contiguous()
    updated.view(-1)[0] = updated.view(-1)[0] + updated.new_tensor(1e-3)
    return name, updated


def torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".", maxsplit=1)[-1]


def assert_success(response: dict[str, Any], *, context: str) -> None:
    success = response.get("success")
    status = response.get("status")
    if success is True or status == "ok":
        return
    raise RuntimeError(f"{context} failed: {response}")


def assert_weight_version(worker: object, *, expected_version: int) -> None:
    actual = ray.get(worker.get_weight_version.remote())
    if actual != str(expected_version):
        raise RuntimeError(
            f"Expected weight_version={expected_version}, got {actual!r}"
        )


def snapshot_weights(worker: object) -> None:
    assert_success(
        ray.get(worker.check_weights.remote(action="snapshot")),
        context="weights_checker/snapshot",
    )


def assert_weights_changed(worker: object, *, context: str) -> None:
    response = ray.get(worker.check_weights.remote(action="compare"))
    if response.get("success") is True:
        raise RuntimeError(
            f"{context}: rollout engine still matches its snapshot: {response}"
        )


def init_trainer_update_group(
    *,
    master_port: int,
    group_name: str,
    device: torch.device,
) -> dist.ProcessGroup:
    if device.type != "cuda":
        raise RuntimeError("distributed weight update test requires CUDA")
    torch.cuda.set_device(device)
    return init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=2,
        rank=0,
        group_name=group_name,
    )


def find_free_port() -> int:
    return get_free_port(start=20000)
