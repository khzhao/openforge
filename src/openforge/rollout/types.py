# Copyright 2026 openforge


from dataclasses import dataclass, field
from typing import Any

from ray.util.placement_group import PlacementGroup

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import RolloutWorkerType
from openforge.configs.topology import ParallelismConfig

__all__ = ["EngineAddr", "EngineSpec"]


@dataclass(slots=True)
class EngineAddr:
    """Resolved network addresses for one rollout engine."""

    host: str
    port: int
    nccl_port: int
    dist_init_addr: str


@dataclass(slots=True)
class EngineSpec:
    """Concrete launch contract for one rollout engine worker."""

    cfg: OpenForgeConfig
    name: str
    worker_type: RolloutWorkerType
    engine_rank: int
    gpu_rank_offset: int
    base_gpu_id: int
    num_gpus: int
    num_cpus: int
    parallelism: ParallelismConfig
    # Scheduling from ray for rollout
    pg: PlacementGroup
    bundle_indices: list[int]
    gpu_ids: list[int]
    # SGLang overrides
    sglang_server_overrides: dict[str, Any] = field(default_factory=dict)
