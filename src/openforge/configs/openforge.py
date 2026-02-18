# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .data import DataConfig
from .gateway import GatewayConfig
from .train import ModelConfig, TrainConfig


@dataclass
class OpenForgeConfig:
    """Configuration for the OpenForge project."""

    data: DataConfig
    gateway: GatewayConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenForgeConfig:
        """Load OpenForgeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")

        return cls(
            data=DataConfig(**cls._section(raw, "data")),
            gateway=GatewayConfig(**cls._section(raw, "gateway")),
            model=ModelConfig(**cls._section(raw, "model")),
            train=TrainConfig(**cls._section(raw, "train")),
        )

    @staticmethod
    def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
        section = raw.get(name)
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")
        return section
