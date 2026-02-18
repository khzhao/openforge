# Copyright 2026 openforge

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str


@dataclass
class TrainConfig:
    """Configuration for the training process."""

    pass
