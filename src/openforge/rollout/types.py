# Copyright 2026 openforge


from dataclasses import dataclass
from typing import Any, Literal

from ray.util.placement_group import PlacementGroup

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import RolloutWorkerType
from openforge.configs.topology import ParallelismConfig

__all__ = ["EngineAddr", "EngineSpec", "RouterSpec"]


@dataclass(slots=True)
class RouterSpec:
    """Configuration parameters for the SGLang router."""

    # Core parameters
    router_name: str
    router_ip: str
    router_port: int
    policy: Literal["cache_aware", "round_robin"]
    worker_urls: list[str]

    # Router non-core parameters
    request_timeout_secs: int
    worker_startup_timeout_secs: int
    worker_startup_check_interval: int
    health_check_timeout_secs: int
    health_check_interval_secs: int
    log_level: str | None

    @property
    def url(self) -> str:
        """Canonical HTTP URL for the router."""
        return f"http://{self.router_ip}:{self.router_port}"

    @property
    def health_url(self) -> str:
        """Router readiness probe URL."""
        return f"{self.url}/health"


@dataclass(slots=True)
class EngineAddr:
    """Resolved network addresses for one rollout engine."""

    host: str
    port: int
    nccl_port: int
    dist_init_addr: str

    @property
    def url(self) -> str:
        """Canonical HTTP URL for the engine."""
        return f"http://{self.host}:{self.port}"


@dataclass(slots=True)
class EngineSpec:
    """Concrete launch contract for one rollout engine worker."""

    # Main
    cfg: OpenForgeConfig
    engine_name: str
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
