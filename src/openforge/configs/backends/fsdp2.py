# Copyright 2026 openforge

"""Configuration models for the FSDP2 backend."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration for FSDP2."""

    param_dtype: Literal["float32", "float16", "bfloat16"]
    reduce_dtype: Literal["float32", "float16", "bfloat16"]


@dataclass
class OffloadConfig:
    """CPU offload configuration for FSDP2."""

    mode: Literal["none", "cpu"]
    pin_memory: bool


@dataclass
class AMPConfig:
    """Automatic Mixed Precision configuration."""

    enabled: bool
    precision: Literal["float32", "float16", "bfloat16"]
    use_grad_scaler: bool


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters."""

    lr: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    max_grad_norm: float


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""

    type: Literal["constant", "cosine", "linear"]
    warmup_steps: int
    min_lr: float
    num_cycles: float
    min_lr_rate: Optional[float] = None


@dataclass
class FSDP2Config:
    """FSDP2 backend configuration."""

    gradient_checkpointing: bool
    reshard_after_forward: bool
    mixed_precision: MixedPrecisionConfig
    offload: OffloadConfig
    amp: AMPConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
