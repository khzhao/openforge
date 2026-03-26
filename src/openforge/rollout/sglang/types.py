# Copyright 2026 openforge

from __future__ import annotations

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

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(slots=True)
class EngineSpec:
    """Concrete launch contract for one rollout engine worker."""

    cfg: OpenForgeConfig
    engine_name: str
    worker_type: RolloutWorkerType

    node_rank: int
    num_nodes: int
    engine_rank: int
    gpu_rank_offset: int
    base_gpu_id: int
    num_gpus: int
    num_cpus: int
    parallelism: ParallelismConfig

    pg: PlacementGroup
    bundle_indices: list[int]
    gpu_ids: list[int]

    enable_memory_saver: bool
    sglang_server_overrides: dict[str, Any]
