# Copyright 2026 openforge

from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

MODEL_SHARDS = [
    "model.layers",  # For Qwen/Llama/Mistal type models on Huggingface
]


def create_device_mesh(
    dp_size: int,
    world_size: int,
    device_type: Literal["cuda", "cpu"] = "cuda",
) -> DeviceMesh:
    """Create a device mesh for FSDP2."""
    assert dist.is_initialized(), "Distributed process group must be initialized"
    assert world_size % dp_size == 0, (
        "World size must be divisible by data parallel size"
    )
    fsdp_size = world_size // dp_size
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(dp_size, fsdp_size),
        mesh_dim_names=("dp", "fsdp"),
    )
    return device_mesh


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Get the torch dtype from a string."""
    return getattr(torch, dtype)


def apply_fsdp2(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: OffloadPolicy,
    reshard_after_forward: bool,
) -> None:
    """Apply FSDP2 to the model."""
    for shard in MODEL_SHARDS:
        if not _has_attribute(model, shard):
            continue
        obj = _get_attribute(model, shard)
        fully_shard(
            obj,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
    fully_shard(
        model,
        mesh=device_mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )


def _get_attribute(obj: Any, attribute_path: str) -> Any:
    for attribute in attribute_path.split("."):
        obj = getattr(obj, attribute)
    return obj


def _has_attribute(obj: Any, attribute_path: str) -> bool:
    try:
        _get_attribute(obj, attribute_path)
        return True
    except AttributeError:
        return False
