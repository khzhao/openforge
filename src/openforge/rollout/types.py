# Copyright 2026 openforge

from dataclasses import dataclass

from openforge.configs.models import OpenForgeConfig
from openforge.configs.topology import ParallelismConfig

__all__ = ["RolloutWorkerSpec"]


@dataclass(slots=True)
class RolloutWorkerSpec:
    """Initialization spec for one rollout worker."""

    cfg: OpenForgeConfig
    engine_replica_index: int
    group_name: str
    role: str
    parallelism: ParallelismConfig
    host: str
    port: int
    policy_version: int | None = None
