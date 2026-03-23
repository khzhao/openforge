# Copyright 2026 openforge

from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel
from .topology import ParallelismConfig


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
    shard_modules: list[str] | None = None
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

    backend: Literal["fsdp2", "megatron"]
    config: FSDP2Config | MegatronConfig

    global_batch_size: int
    mini_batch_size: int
    micro_batch_size: int
    ppo_epochs: int = 1
    max_rollout_policy_lag: int
    checkpoints: str

    cpus_per_worker: int
    parallel: ParallelismConfig

    @model_validator(mode="after")
    def _validate_train_config(self) -> "TrainConfig":
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be > 0")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be > 0")
        if self.max_rollout_policy_lag < 0:
            raise ValueError("max_rollout_policy_lag must be >= 0")
        if self.global_batch_size % self.mini_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by mini_batch_size")
        if self.mini_batch_size % self.micro_batch_size != 0:
            raise ValueError("mini_batch_size must be divisible by micro_batch_size")
        if self.cpus_per_worker < 0:
            raise ValueError("cpus_per_worker must be >= 0")

        if self.backend == "fsdp2":
            if not isinstance(self.config, FSDP2Config):
                raise ValueError(
                    "train.config must be FSDP2Config when backend is fsdp2"
                )
            if (
                self.parallel.pipeline_parallel_size > 1
                or self.parallel.tensor_parallel_size > 1
                or self.parallel.context_parallel_size > 1
                or self.parallel.expert_parallel_size > 1
            ):
                raise ValueError(
                    "FSDP2 does not support pipeline, tensor, context, or expert parallelism"
                )

        if self.backend == "megatron" and not isinstance(self.config, MegatronConfig):
            raise ValueError(
                "train.config must be MegatronConfig when backend is megatron"
            )
        return self

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(self.mini_batch_size / self.micro_batch_size)

    @property
    def num_workers(self) -> int:
        return self.parallel.world_size

    @property
    def total_gpus(self) -> int:
        return self.num_workers

    @property
    def total_cpus(self) -> int:
        return self.num_workers * self.cpus_per_worker
