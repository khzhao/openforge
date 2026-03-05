# Copyright 2026 openforge

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel


class MixedPrecisionConfig(OpenForgeBaseModel):
    """Mixed precision configuration for FSDP2."""

    param_dtype: Literal["float32", "float16", "bfloat16"]
    reduce_dtype: Literal["float32", "float16", "bfloat16"]


class OffloadConfig(OpenForgeBaseModel):
    """CPU offload configuration for FSDP2."""

    mode: Literal["none", "cpu"]
    pin_memory: bool


class AMPConfig(OpenForgeBaseModel):
    """Automatic Mixed Precision configuration."""

    enabled: bool
    precision: Literal["float32", "float16", "bfloat16"]
    use_grad_scaler: bool


class OptimizerConfig(OpenForgeBaseModel):
    """Optimizer hyperparameters."""

    lr: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    max_grad_norm: float


class SchedulerConfig(OpenForgeBaseModel):
    """Learning rate scheduler settings."""

    type: Literal["constant", "cosine", "linear"]
    warmup_steps: int
    min_lr: float
    num_cycles: float
    min_lr_rate: float | None = None


class FSDP2Config(OpenForgeBaseModel):
    """FSDP2 backend configuration."""

    gradient_checkpointing: bool
    reshard_after_forward: bool
    mixed_precision: MixedPrecisionConfig
    offload: OffloadConfig
    amp: AMPConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


class MegatronConfig(OpenForgeBaseModel):
    """Megatron backend configuration."""

    def model_post_init(self) -> None:
        raise NotImplementedError("Megatron backend is not implemented yet")


class TrainConfig(OpenForgeBaseModel):
    """Configuration for the training process."""

    # Backend selection and backend-specific settings.
    backend: Literal["fsdp2", "megatron"]
    backend_cfg: FSDP2Config | MegatronConfig
    use_critic: bool

    # Cluster resources.
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int
    num_gpus_per_engine: int
    num_cpus_per_engine: int

    # Distributed parallelism layout.
    data_parallel_size: int
    pipeline_parallel_size: int
    tensor_parallel_size: int

    # Batch sizing semantics.
    global_batch_size: int
    mini_batch_size: int
    micro_batch_size: int

    # Checkpointing.
    checkpoints_dir: str

    @model_validator(mode="after")
    def _validate_backend_settings(self) -> TrainConfig:
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be > 0")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if self.global_batch_size % self.mini_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by mini_batch_size")
        if self.mini_batch_size % self.micro_batch_size != 0:
            raise ValueError("mini_batch_size must be divisible by micro_batch_size")

        if self.backend == "fsdp2":
            if not isinstance(self.backend_cfg, FSDP2Config):
                raise ValueError(
                    "backend_cfg must be FSDP2Config when backend is fsdp2"
                )
            if self.pipeline_parallel_size > 1 or self.tensor_parallel_size > 1:
                raise ValueError(
                    "FSDP2 does not support pipeline parallelism or tensor parallelism"
                )
        if self.backend == "megatron" and not isinstance(
            self.backend_cfg, MegatronConfig
        ):
            raise ValueError(
                "backend_cfg must be MegatronConfig when backend is megatron"
            )
        return self

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(self.mini_batch_size / self.micro_batch_size)
