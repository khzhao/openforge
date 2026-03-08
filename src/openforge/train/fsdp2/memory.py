# Copyright 2026 openforge

import torch
from torch.distributed.fsdp import FSDPModule
from torch.optim import Optimizer


@torch.no_grad()
def offload_params(
    model: FSDPModule | torch.nn.Module,
    offload_grad: bool = False,
) -> None:
    """Offload model parameters to CPU."""
    model.to(device="cpu")
    if not offload_grad:
        return

    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None or grad.device.type == "cpu":
            continue
        parameter.grad = grad.detach().to("cpu", non_blocking=True)
        del grad


@torch.no_grad()
def onload_params(
    model: FSDPModule | torch.nn.Module,
    device: torch.device,
    onload_grad: bool = False,
) -> None:
    """Load model parameters from CPU to GPU."""
    model.to(device=device)
    if not onload_grad:
        return

    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None or grad.device == device:
            continue
        parameter.grad = grad.detach().to(device, non_blocking=True)
        del grad


@torch.no_grad()
def offload_optimizer(optimizer: Optimizer) -> None:
    """Offload optimizer state to CPU."""
    _move_optimizer_state(optimizer, torch.device("cpu"))


@torch.no_grad()
def onload_optimizer(optimizer: Optimizer, device: torch.device) -> None:
    """Load optimizer state from CPU to GPU."""
    _move_optimizer_state(optimizer, device)


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    """Move optimizer state to a given device."""
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or value.device == device:
                continue
            state[key] = value.detach().to(device, non_blocking=True)
            del value
    if torch.cuda.is_available():
        torch.cuda.synchronize()
