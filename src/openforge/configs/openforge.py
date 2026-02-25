# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .backends.fsdp2 import (
    AMPConfig,
    FSDP2Config,
    MixedPrecisionConfig,
    OffloadConfig,
    OptimizerConfig,
    SchedulerConfig,
)
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
            train=cls._train_config(raw),
        )

    @staticmethod
    def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
        section = raw.get(name)
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")
        return section

    @classmethod
    def _train_config(cls, raw: dict[str, Any]) -> TrainConfig:
        train = cls._section(raw, "train")
        backend = train.get("backend")
        if backend != "fsdp2":
            raise ValueError("train.backend must be set to 'fsdp2'")

        backend_cfg = cls._fsdp2_config(train["backend_cfg"])
        values = dict(train)
        values["backend_cfg"] = backend_cfg
        return TrainConfig(**values)

    @classmethod
    def _fsdp2_config(cls, raw: dict[str, Any]) -> FSDP2Config:
        return FSDP2Config(
            gradient_checkpointing=raw["gradient_checkpointing"],
            reshard_after_forward=raw["reshard_after_forward"],
            mixed_precision=MixedPrecisionConfig(**raw["mixed_precision"]),
            offload=OffloadConfig(**raw["offload"]),
            amp=AMPConfig(**raw["amp"]),
            optim=OptimizerConfig(**raw["optim"]),
            scheduler=SchedulerConfig(**raw["scheduler"]),
        )
