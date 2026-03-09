# Copyright 2026 openforge

from dataclasses import dataclass

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import ResolvedRolloutEngine

__all__ = ["RolloutWorkerSpec"]


@dataclass(slots=True)
class RolloutWorkerSpec:
    """Initialization spec for one rollout worker."""

    cfg: OpenForgeConfig
    engine: ResolvedRolloutEngine
    host: str
    port: int
    bootstrap_port: int | None = None
    policy_version: int | None = None
