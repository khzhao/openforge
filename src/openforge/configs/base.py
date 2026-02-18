# Copyright 2026 openforge

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    train_batch_size: int = 256
    valid_batch_size: int = 128
    ppo_mini_batch_size_per_gpu: int = 32
    ppo_micro_batch_size_per_gpu: int = 8
    ppo_epochs: int = 1
    train_data: Optional[str] = None
    valid_data: Optional[str] = None
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B-base"
    tokenizer_name_or_path: str = "Qwen/Qwen3-0.6B"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    zero_stage: int = 0
    host: str = "127.0.0.1"
    port: int = 23333
    backend: str = "nccl"
    seed: int = 42
    gradient_checkpointing_enable: bool = True


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    lr: float = 5e-6
    min_lr: float = 0.0
    lr_warmup_steps_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: float = 1.0
    weight_decay: float = 1e-3
