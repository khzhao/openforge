# Copyright 2026 openforge

from dataclasses import dataclass

from openforge.configs.models import OpenForgeConfig

__all__ = [
    "TrainStepResult",
    "TrainWorkerSpec",
    "TrainWorkerState",
]


@dataclass(slots=True)
class TrainWorkerSpec:
    """Rank-local worker initialization parameters."""

    cfg: OpenForgeConfig
    rank: int
    world_size: int
    master_addr: str
    master_port: int


@dataclass(slots=True)
class TrainWorkerState:
    """Public train worker state snapshot."""

    rank: int
    world_size: int
    backend: str
    device: str
    sleeping: bool


@dataclass(slots=True)
class TrainStepResult:
    """Result of one optimizer step on one train rank."""

    rank: int
    global_step: int | None
    metrics: dict[str, float]
