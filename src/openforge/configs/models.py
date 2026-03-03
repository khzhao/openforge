# Copyright 2026 openforge

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Reward = list[float] | float | None


class OpenForgeBaseModel(BaseModel):
    """Base class for OpenForge config models."""

    model_config = ConfigDict(extra="forbid")


class RolloutDatum(OpenForgeBaseModel):
    """Single rollout item stored for training consumption."""

    sample_id: str
    prompt_id: str
    group_id: str
    turn_index: int
    rollout_model_version: int
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    old_logprobs: list[float]
    reward: Reward = None
    consumed: bool = False


class DataConfig(OpenForgeBaseModel):
    """Configuration for data source behavior."""

    backend: str = "memory"


class GatewayConfig(OpenForgeBaseModel):
    """Configuration for the Gateway server."""

    host: str = "0.0.0.0"
    port: int = 8000
    backend_url: str = "http://localhost:8080/v1"


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
    min_lr_rate: Optional[float] = None


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

    pass


class ModelConfig(OpenForgeBaseModel):
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str


class TrainConfig(OpenForgeBaseModel):
    """Configuration for the training process."""

    backend: Literal["fsdp2", "megatron"]
    backend_cfg: FSDP2Config | MegatronConfig
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int
    data_parallel_size: int
    pipeline_parallel_size: int
    tensor_parallel_size: int
    checkpoints_dir: str
    use_critic: bool

    @model_validator(mode="after")
    def _validate_backend_settings(self) -> TrainConfig:
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


class OpenForgeConfig(OpenForgeBaseModel):
    """Configuration for the OpenForge project."""

    data: DataConfig = Field(default_factory=DataConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenForgeConfig:
        """Load OpenForgeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)
