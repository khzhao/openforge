# Copyright 2026 openforge

from __future__ import annotations

from pathlib import Path

import yaml

from .base import OpenForgeBaseModel
from .rollout import RolloutConfig
from .train import TrainConfig


class DataConfig(OpenForgeBaseModel):
    """Configuration for data source behavior."""

    backend: str


class GatewayConfig(OpenForgeBaseModel):
    """Configuration for the Gateway server."""

    host: str
    port: int
    backend_url: str


class ModelConfig(OpenForgeBaseModel):
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str


class OpenForgeConfig(OpenForgeBaseModel):
    """Configuration for the OpenForge project."""

    data: DataConfig
    gateway: GatewayConfig
    model: ModelConfig
    train: TrainConfig
    rollout: RolloutConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenForgeConfig:
        """Load OpenForgeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)
