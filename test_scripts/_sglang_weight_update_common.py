# Copyright 2026 openforge

from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import snapshot_download
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.common import init_custom_process_group
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from transformers import AutoModelForCausalLM

from openforge.configs.models import OpenForgeConfig

if TYPE_CHECKING:
    from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def require_visible_gpus(min_count: int) -> None:
    visible = torch.cuda.device_count()
    if visible < min_count:
        raise RuntimeError(
            f"Expected at least {min_count} visible GPUs, found {visible}. "
            "Adjust CUDA_VISIBLE_DEVICES before running this script."
        )


def resolve_local_model_path(model_path_or_id: str) -> str:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=model_path_or_id, local_files_only=True)


def build_cfg(model_path: str, *, train_total_gpus: int) -> OpenForgeConfig:
    checkpoint_root = tempfile.mkdtemp(prefix="openforge-test-checkpoints-")
    total_gpus = max(torch.cuda.device_count(), train_total_gpus + 1)
    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "dummy"},
            "gateway": {"host": "127.0.0.1", "port": 0},
            "model": {
                "model_name_or_path": model_path,
                "tokenizer_name_or_path": model_path,
                "attn_implementation": "sdpa",
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": total_gpus,
                "cpus_per_node": 8,
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
                    "data_parallel_size": train_total_gpus,
                    "fsdp_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
                "placement": {"strategy": "PACK"},
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
                        "role": "regular",
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
                        "placement": {"strategy": "PACK"},
                    }
                ],
            },
        }
    )


def start_runtime(
    *,
    model_path: str,
    name: str,
    colocated: bool,
    train_total_gpus: int,
) -> tuple["SGLangEngineRuntime", OpenForgeConfig]:
    try:
        from openforge.rollout.sglang.engine_runtime import SGLangEngineRuntime
    except Exception as exc:
        raise RuntimeError(
            "Importing SGLangEngineRuntime failed before server launch. "
            "This environment currently has a broken transformers/torchvision "
            "stack for SGLang imports."
        ) from exc
    cfg = build_cfg(model_path, train_total_gpus=train_total_gpus)
    runtime = SGLangEngineRuntime(
        name=name,
        host="127.0.0.1",
        port=find_free_port(),
        model_path=model_path,
        parallelism=cfg.rollout.engine_groups[0].parallelism,
        request_timeout_seconds=10.0,
    )
    runtime.start(
        cfg=cfg,
        engine_replica_index=0,
        num_nodes=1,
        node_rank=0,
        dist_init_addr=f"127.0.0.1:{find_free_port()}",
        nccl_port=find_free_port(),
        colocated=colocated,
    )
    return runtime, cfg


def load_source_model(model_path: str, *, device: torch.device) -> AutoModelForCausalLM:
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model.to(device)


def _updated_tensor_copy(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    updated = tensor.detach().clone().contiguous()
    updated.view(-1)[0] = updated.view(-1)[0] + updated.new_tensor(1e-3)
    return name, updated


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
        return _updated_tensor_copy(name, tensor)
    if fallback is not None:
        return _updated_tensor_copy(*fallback)
    raise RuntimeError("Could not find a floating-point tensor to update")


def serialize_flattened_bucket(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    payload = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
    return MultiprocessingSerializer.serialize(payload, output_str=True)


def torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".", maxsplit=1)[-1]


def assert_success(response: dict[str, Any], *, context: str) -> None:
    success = response.get("success")
    status = response.get("status")
    if success is True or status == "ok":
        return
    if success is None and status is None:
        raise RuntimeError(f"{context} returned no success indicator: {response}")
    if not response.get("success", False):
        raise RuntimeError(f"{context} failed: {response}")


def assert_weight_version(
    runtime: "SGLangEngineRuntime",
    *,
    expected_version: int,
) -> None:
    actual = runtime.get_weight_version()
    if actual != str(expected_version):
        raise RuntimeError(
            f"Expected weight_version={expected_version}, got {actual!r}"
        )


def init_trainer_update_group(
    *,
    master_port: int,
    group_name: str,
    device: torch.device,
):
    if device.type != "cuda":
        raise RuntimeError("Distributed weight update test requires CUDA")
    torch.cuda.set_device(device)
    return init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=2,
        rank=0,
        group_name=group_name,
    )
