# Copyright 2026 openforge

from enum import Enum
from typing import Literal

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs import OpenForgeConfig
from openforge.scheduler.workers import ActorCriticRefWorker


class PlacementStrategy(Enum):
    """PACK = fewest nodes; SPREAD = most nodes."""

    PACK = "STRICT_PACK"
    SPREAD = "STRICT_SPREAD"


class ActorCriticRefGroup:
    """Ray actor group for colocated actor-critic-ref workers.

    Args:
        cfg: OpenForge configuration.
        strategy: PACK (default) or SPREAD placement across nodes.
    """

    def __init__(
        self,
        cfg: OpenForgeConfig,
        master_addr: str,
        master_port: int,
        strategy: PlacementStrategy
        | Literal["PACK", "SPREAD"] = PlacementStrategy.PACK,
    ) -> None:
        self.cfg = cfg
        self._master_addr = master_addr
        self._master_port = master_port
        self._num_nodes = cfg.train.num_nodes
        self._num_gpus_per_node = cfg.train.num_gpus_per_node
        self._num_gpus_per_engine = cfg.train.num_gpus_per_engine
        self._num_cpus_per_engine = cfg.train.num_cpus_per_engine
        self._world_size = self._num_nodes * self._num_gpus_per_node
        self._workers: list[ray.actor.ActorHandle] = []

        if isinstance(strategy, str):
            strategy = PlacementStrategy[strategy]
        self._pg = self._create_placement_group(strategy)
        self._create_workers()

    def _create_placement_group(
        self,
        strategy: PlacementStrategy,
    ) -> ray.util.placement_group.PlacementGroup:
        """Reserve GPU+CPU bundles atomically via Ray."""
        bundles = [
            {"GPU": self._num_gpus_per_engine, "CPU": self._num_cpus_per_engine}
        ] * self._world_size
        pg = placement_group(bundles, strategy=strategy.value)
        ray.get(pg.ready())
        return pg

    def _create_workers(self) -> None:
        """Pin one remote worker per bundle in the placement group."""
        RemoteWorker = ray.remote(ActorCriticRefWorker)

        for rank in range(self._world_size):
            worker = RemoteWorker.options(
                num_cpus=self._num_cpus_per_engine,
                num_gpus=self._num_gpus_per_engine,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=rank,
                ),
            ).remote()
            self._workers.append(worker)

    def initialize(self) -> None:
        """Discover rendezvous endpoint from rank 0, then init all workers."""
        ray.get(
            [
                self._workers[rank].initialize.remote(
                    self.cfg,
                    rank=rank,
                    world_size=self._world_size,
                    master_addr=self._master_addr,
                    master_port=self._master_port,
                )
                for rank in range(self._world_size)
            ]
        )

    def shutdown(self) -> None:
        """Shut down workers and release the placement group."""
        ray.get([worker.shutdown.remote() for worker in self._workers])
        self._workers.clear()
        ray.util.remove_placement_group(self._pg)

    def async_train_step(
        self,
        batch,
        *,
        global_step: int | None = None,
    ) -> list[ray.ObjectRef]:
        """Dispatch a training step to all workers; returns ObjectRefs."""
        return [
            worker.train_step.remote(batch, global_step=global_step)
            for worker in self._workers
        ]

    def save_checkpoint(self, *, step: int, policy_version: int) -> list[str]:
        """Save a distributed checkpoint; returns saved paths."""
        return ray.get(
            [
                worker.save_checkpoint.remote(step=step, policy_version=policy_version)
                for worker in self._workers
            ]
        )

    def load_checkpoint(self) -> tuple[int, int] | None:
        """Returns (step, policy_version) from rank 0, or None."""
        results = ray.get([worker.load_checkpoint.remote() for worker in self._workers])
        return results[0]

    def sleep(self) -> None:
        """Offload all workers to CPU."""
        ray.get([worker.sleep.remote() for worker in self._workers])

    def wakeup(self) -> None:
        """Restore all workers to GPU."""
        ray.get([worker.wakeup.remote() for worker in self._workers])

    def clear_memory(self) -> None:
        """Release temporary GPU memory on all workers."""
        ray.get([worker.clear_memory.remote() for worker in self._workers])
