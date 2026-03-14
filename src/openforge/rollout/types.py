# Copyright 2026 openforge


from dataclasses import dataclass
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

    # Main
    cfg: OpenForgeConfig
    name: str
    worker_type: RolloutWorkerType

    # Distributed
    node_rank: int
    num_nodes: int
    engine_rank: int
    gpu_rank_offset: int  # The GPU rank w.r.t bundle_indices
    base_gpu_id: int  # The physical GPU id
    num_gpus: int
    num_cpus: int
    parallelism: ParallelismConfig

    # Scheduling from ray for rollout
    pg: PlacementGroup
    bundle_indices: list[int]
    gpu_ids: list[int]

    # SGLang overrides
    enable_memory_saver: bool
    sglang_server_overrides: dict[str, Any]
