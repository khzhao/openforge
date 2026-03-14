# Copyright 2026 openforge

from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.types import EngineAddr, EngineSpec

__all__ = ["RolloutManager"]


@ray.remote(num_gpus=0, num_cpus=1)
class RolloutManager:
    """Manager for the rollout process."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ):
        """Startup processes for the rollout manager."""
        self.cfg = cfg
        self.placement_groups = placement_groups

    def initialize(self) -> None:
        """Initialize the rollout manager."""
        # 1. Start the SGLang engines
        self.engine_info = start_sglang_engines(self.cfg, self.placement_groups)

    def shutdown(self) -> None:
        """Terminate all child engine actors started by this manager."""
        for worker in self.engine_info["engine_workers"]:
            ray.kill(worker)
        self.engine_info.clear()


def start_sglang_engines(
    cfg: OpenForgeConfig,
    placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
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
            name=engine_name,
            worker_type=engine_group_cfg.worker_type,
            engine_rank=engine_rank,
            gpu_rank_offset=gpu_rank_offset,
            base_gpu_id=base_gpu_id,
            num_gpus=engine_group_cfg.num_gpus_per_replica,
            num_cpus=engine_group_cfg.num_cpus_per_replica,
            parallelism=engine_group_cfg.parallelism,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
        )
        engine_worker = EngineWorker.options(
            num_cpus=engine_group_cfg.num_cpus_per_replica,
            num_gpus=engine_group_cfg.num_gpus_per_replica,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[gpu_rank_offset],
            ),
        ).remote()
        engine_specs_and_workers.append((engine_spec, engine_worker))

    # 2. Initialize the engine workers
    states = []
    for spec, worker in engine_specs_and_workers:
        state = worker.initialize.remote(spec)
        states.append(state)
    ray.get(states)

    # 3. Allocate addresses and ports, then launch the servers
    engine_addrs = allocate_engine_addrs(engine_specs_and_workers)
    launches = []
    for spec, worker in engine_specs_and_workers:
        addr = engine_addrs[spec.name]
        launch = worker.launch.remote(addr)
        launches.append(launch)
    ray.get(launches)

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
        engine_addrs[spec.name] = EngineAddr(
            host=host,
            port=port,
            nccl_port=port + 1,
            dist_init_addr=f"{host}:{port + 2}",
        )
    return engine_addrs
