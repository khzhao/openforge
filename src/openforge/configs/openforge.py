# Copyright 2026 openforge

from dataclasses import dataclass

from .gateway import GatewayConfig
from .train import ModelConfig, TrainConfig


@dataclass
class OpenForgeConfig:
    """Configuration for the OpenForge project."""

    gateway: GatewayConfig
    model: ModelConfig
    train: TrainConfig
