# Copyright 2026 openforge

import torch


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Get the torch dtype from a string."""
    return getattr(torch, dtype)
