# Copyright (c) 2026 rlc

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def _constant_lr_lambda(current_step: int, num_warmup_steps: int) -> float:
    """Constant LR with linear warmup."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def _cosine_lr_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float,
) -> float:
    """Cosine LR with linear warmup and a minimum LR ratio."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine_decay


def _linear_lr_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_rate: float,
) -> float:
    """Linear LR decay with linear warmup and a minimum LR ratio."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))


def get_lr_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
    min_lr_rate: float | None = None,
    last_epoch: int = -1,
) -> LRScheduler:
    """Build a LambdaLR scheduler."""
    if min_lr_rate is None:
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr_rate = min_lr / base_lr if base_lr > 0 else 0.0

    if scheduler_type == "constant":
        lr_lambda = partial(
            _constant_lr_lambda,
            num_warmup_steps=num_warmup_steps,
        )
    elif scheduler_type == "cosine":
        lr_lambda = partial(
            _cosine_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=min_lr_rate,
        )
    elif scheduler_type == "linear":
        lr_lambda = partial(
            _linear_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )
    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. "
            "Supported types: constant, cosine, linear"
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
