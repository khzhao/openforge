# Copyright 2026 openforge

from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.sglang.router import Router
from openforge.rollout.types import (
    EngineAddr,
    EngineSpec,
    RouterSpec,
)
from openforge.utils.networking import get_free_port, get_host_ip
from openforge.utils.ray import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

__all__ = [
    "RolloutManager",
    "EngineGroup",
    "start_sglang_engines",
    "allocate_engine_addrs",
]


class RolloutManager:
    """Manager for the rollout process."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ):
        """Initialize the rollout manager."""
        self.cfg = cfg
        self.placement_groups = placement_groups

    def initialize(
        self,
        *,
        router_ip: str | None = None,
        router_port: int | None = None,
        engine_addrs: dict[str, EngineAddr] | None = None,
        **router_kwargs: Any,
    ) -> None:
        """Initialize the rollout manager."""
        self.engine_group = EngineGroup(self.cfg, self.placement_groups)
        self.engine_group.initialize(engine_addrs=engine_addrs)

        main_router_kwargs = {
            "router_name": "openforge-router",
            "router_ip": router_ip or get_host_ip(),
            "router_port": router_port or get_free_port(start=30000),
            "policy": router_kwargs.pop("policy", "cache_aware"),
            "worker_urls": [
                addr.url for addr in self.engine_group.engine_addrs.values()
            ],
            "request_timeout_secs": router_kwargs.pop("request_timeout_secs", 300),
            "worker_startup_timeout_secs": router_kwargs.pop(
                "worker_startup_timeout_secs", 300
            ),
            "worker_startup_check_interval": router_kwargs.pop(
                "worker_startup_check_interval", 1
            ),
            "health_check_timeout_secs": router_kwargs.pop(
                "health_check_timeout_secs", 5
            ),
            "health_check_interval_secs": router_kwargs.pop(
                "health_check_interval_secs", 5
            ),
            "log_level": router_kwargs.pop("log_level", None),
        }
        main_router_kwargs.update(router_kwargs)

        self.router_spec = RouterSpec(**main_router_kwargs)
        self._router = Router()
        self._router.initialize(self.router_spec)
        self._router.launch()

    def shutdown(self) -> None:
        """Shutdown the rollout manager."""
        self.engine_group.shutdown()
        self.router.shutdown()

    @property
    def router(self) -> Router:
        return self._router

    @property
    def engine_workers(self) -> list[Engine]:
        return self.engine_group.engine_workers

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        return self.engine_group.engine_addrs

    @property
    def engine_specs(self) -> list[EngineSpec]:
        return self.engine_group.engine_specs


class EngineGroup:
    """Group for the SGLang engine group."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ):
        """Startup processes for the SGLang engine group."""
        self.cfg = cfg
        self.placement_groups = placement_groups

    def initialize(
        self,
        *,
        engine_addrs: dict[str, EngineAddr] | None = None,
    ) -> None:
        """Initialize the SGLang engine group."""
        self.engine_info = start_sglang_engines(
            self.cfg,
            self.placement_groups,
            engine_addrs=engine_addrs,
        )

    def shutdown(self) -> None:
        """Terminate all child engine actors started by this group."""
        workers = self.engine_workers
        try:
            if workers:
                ray.get([worker.stop.remote() for worker in workers])
        finally:
            for worker in workers:
                ray.kill(worker)
            self.engine_info = {}

    @property
    def engine_specs(self) -> list[EngineSpec]:
        return self.engine_info["engine_specs"]

    @property
    def engine_workers(self) -> list[Engine]:
        return self.engine_info["engine_workers"]

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        return self.engine_info["engine_addrs"]


def start_sglang_engines(
    cfg: OpenForgeConfig,
    placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    engine_addrs: dict[str, EngineAddr] | None = None,
) -> dict[str, Any]:
    """Build engine launch specs from exact placements supplied by the caller."""
    assert len(cfg.rollout.engine_groups) == 1, (
        "Only one engine group is supported for rollout right now"
    )
    engine_group_cfg = cfg.rollout.engine_groups[0]
    assert engine_group_cfg.worker_type == "regular", (
        "Only regular engine groups are supported for rollout"
    )

    # 0. Get the placement group and GPU IDs
    pg, bundle_indices, gpu_ids = placement_groups["rollout"]

    # 1. Create the engine workers and their specs
    EngineWorker = ray.remote(Engine)
    engine_specs_and_workers = []
    for engine_rank in range(engine_group_cfg.replicas):
        engine_name = f"{engine_group_cfg.name}-{engine_rank}"
        gpu_rank_offset = engine_rank * engine_group_cfg.num_gpus_per_replica
        base_gpu_id = gpu_ids[gpu_rank_offset]

        engine_spec = EngineSpec(
            cfg=cfg,
            engine_name=engine_name,
            worker_type=engine_group_cfg.worker_type,
            node_rank=0,
            num_nodes=1,
            engine_rank=engine_rank,
            gpu_rank_offset=gpu_rank_offset,
            base_gpu_id=base_gpu_id,
            num_gpus=engine_group_cfg.num_gpus_per_replica,
            num_cpus=engine_group_cfg.num_cpus_per_replica,
            parallelism=engine_group_cfg.parallelism,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            enable_memory_saver=engine_group_cfg.enable_memory_saver,
            sglang_server_overrides=engine_group_cfg.sglang_server_overrides,
        )
        engine_worker = EngineWorker.options(
            num_cpus=1,
            num_gpus=0.1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[gpu_rank_offset],
            ),
            runtime_env={
                "env_vars": {
                    **dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1"),
                }
            },
        ).remote()
        engine_specs_and_workers.append((engine_spec, engine_worker))

    # 2. Initialize the engine workers
    ray.get(
        [worker.initialize.remote(spec) for spec, worker in engine_specs_and_workers]
    )

    # 3. Allocate addresses and ports, then launch the servers
    if engine_addrs is None:
        engine_addrs = allocate_engine_addrs(engine_specs_and_workers)
    else:
        expected_names = {spec.engine_name for spec, _ in engine_specs_and_workers}
        assert set(engine_addrs) == expected_names, (
            "engine_addrs must match the rollout engine names exactly: "
            f"expected {sorted(expected_names)}, got {sorted(engine_addrs)}"
        )
    ray.get(
        [
            worker.launch.remote(engine_addrs[spec.engine_name])
            for spec, worker in engine_specs_and_workers
        ]
    )

    # 4. Prepare outputs
    engine_specs = [spec for spec, _ in engine_specs_and_workers]
    engine_workers = [worker for _, worker in engine_specs_and_workers]
    return {
        "engine_specs": engine_specs,
        "engine_workers": engine_workers,
        "engine_addrs": engine_addrs,
    }


def allocate_engine_addrs(
    engine_specs_and_workers: list[tuple[EngineSpec, Engine]],
    base_port: int = 15000,
) -> dict[str, EngineAddr]:
    """Allocate host/port tuples for every engine, avoiding collisions per node."""
    PORTS_PER_ENGINE = 3
    engine_specs = [spec for spec, _ in engine_specs_and_workers]
    engine_workers = [worker for _, worker in engine_specs_and_workers]
    hosts = ray.get([worker.get_ip_addr.remote() for worker in engine_workers])
    node_port_cursor = {}
    engine_addrs = {}
    for spec, worker, host in zip(engine_specs, engine_workers, hosts, strict=True):
        cursor = node_port_cursor.get(host, base_port)
        port = ray.get(
            worker.get_free_port.remote(start=cursor, block_size=PORTS_PER_ENGINE)
        )
        node_port_cursor[host] = port + PORTS_PER_ENGINE
        engine_addrs[spec.engine_name] = EngineAddr(
            host=host,
            port=port,
            nccl_port=port + 1,
            dist_init_addr=f"{host}:{port + 2}",
        )
    return engine_addrs
