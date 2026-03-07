# Copyright 2026 openforge

import gc

import torch


def clear_memory(sync: bool = True) -> None:
    """Clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def get_curr_gpu_device_memory_profile() -> dict[str, float]:
    """Get the memory profile of the current GPU device."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    max_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    free = total - reserved

    return {
        "device": device,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
        "total_gb": total,
        "free_gb": free,
        "utilization": allocated / total if total > 0 else 0.0,
    }
