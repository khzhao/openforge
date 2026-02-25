# Copyright 2026 openforge

from dataclasses import dataclass
from typing import Literal

from .backends import FSDP2Config


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str


@dataclass
class TrainConfig:
    """Configuration for the training process."""

    backend: Literal["fsdp2"]
    backend_cfg: FSDP2Config
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int
    data_parallel_size: int
    pipeline_parallel_size: int
    tensor_parallel_size: int
