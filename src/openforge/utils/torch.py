# Copyright 2026 openforge

from contextlib import ContextManager, contextmanager

import torch


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Get the torch dtype from a string."""
    return getattr(torch, dtype)


def get_torch_dtype_name(dtype: torch.dtype) -> str:
    """Get the torch dtype name from a torch dtype."""
    return str(dtype).removeprefix("torch.")


@contextmanager
def set_torch_default_dtype(dtype: torch.dtype) -> ContextManager[None]:
    """Context manager for setting the default torch dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
