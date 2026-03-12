# Copyright 2026 openforge

import torch


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Get the torch dtype from a string."""
    return getattr(torch, dtype)


def get_torch_dtype_name(dtype: torch.dtype) -> str:
    """Get the torch dtype name from a torch dtype."""
    return dtype.name
