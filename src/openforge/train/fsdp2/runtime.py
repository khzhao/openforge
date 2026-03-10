# Copyright 2026 openforge

from typing import Literal, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

_DEFAULT_BLOCK_ATTR_PATHS: Sequence[str] = (
    "model.layers",  # Llama/Qwen/Mistral-style HF models
    "transformer.h",  # GPT2-style HF models
    "gpt_neox.layers",  # GPT-NeoX-style HF models
    "model.decoder.layers",  # Encoder/decoder-style HF models
)


def create_device_mesh(
    dp_size: int,
    fsdp_size: int,
    world_size: int,
    device_type: Literal["cuda", "cpu"] = "cuda",
) -> DeviceMesh:
    """Create a device mesh for FSDP2."""
    assert dist.is_initialized(), "Distributed process group must be initialized"
    assert world_size == dp_size * fsdp_size, (
        "FSDP2 currently expects world_size to equal "
        "data_parallel_size * fsdp_parallel_size"
    )
    return init_device_mesh(
        device_type,
        mesh_shape=(dp_size, fsdp_size),
        mesh_dim_names=("dp", "fsdp"),
    )


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Get the torch dtype from a string."""
    return getattr(torch, dtype)


def apply_fsdp2(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = False,
    shard_modules: Sequence[str] | None = None,
) -> FSDPModule:
    """Apply FSDP2 to a model.

    We shard transformer blocks first and then shard the root module so tied
    embeddings remain in the root FSDP instance instead of being wrapped
    independently.
    """
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    if offload_policy is None:
        offload_policy = OffloadPolicy()

    for module in resolve_shard_modules(model, shard_modules):
        fully_shard(
            module,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
    return fully_shard(
        model,
        mesh=device_mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )


def _getattr_path(obj: object, attr_path: str) -> object:
    """Resolve a dotted attribute path from an object."""
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr)
    return current


def _as_module_list(value: object, attr_path: str) -> list[nn.Module]:
    """Normalize a resolved attribute value into a list of modules."""
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, nn.Module):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(module, nn.Module) for module in value
    ):
        return list(value)
    raise TypeError(
        "fsdp2.shard_modules path "
        f"{attr_path!r} must resolve to nn.Module, nn.ModuleList, or "
        "list/tuple of nn.Module."
    )


def resolve_shard_modules(
    model: nn.Module,
    shard_module_paths: Sequence[str] | None,
) -> list[nn.Module]:
    """Resolve configured module paths into concrete module objects."""
    attr_paths = shard_module_paths or _DEFAULT_BLOCK_ATTR_PATHS

    resolved: list[nn.Module] = []
    seen_ids: set[int] = set()
    for attr_path in attr_paths:
        try:
            value = _getattr_path(model, attr_path)
        except AttributeError as exc:
            if shard_module_paths is None:
                continue
            raise ValueError(
                f"Invalid fsdp2.shard_modules path {attr_path!r}: attribute not found."
            ) from exc

        for module in _as_module_list(value, attr_path):
            module_id = id(module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            resolved.append(module)
    return resolved
