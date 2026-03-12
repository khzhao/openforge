# Copyright 2026 openforge

from typing import Sequence

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tensordict import TensorDict

from openforge.configs.models import OpenForgeConfig
from openforge.configs.topology import PlacementStrategy
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.train.worker import TrainWorker
from openforge.utils.ray import ray_placement_group_strategy

__all__ = ["TrainWorkerGroup"]


class TrainWorkerGroup:
    """Ray actor group for one train worker per rank."""

    def initialize(
        self,
        cfg: OpenForgeConfig,
        master_addr: str,
        master_port: int,
        strategy: PlacementStrategy | str = PlacementStrategy.PACK,
    ) -> list[TrainWorkerState]:
        self.cfg = cfg
        self.master_addr = master_addr
        self.master_port = master_port

        self.num_gpus_per_worker = 1
        self.num_cpus_per_worker = cfg.train.cpus_per_worker
        self.world_size = cfg.train.num_workers
        self.workers = []

        self.pg = self._create_placement_group(strategy)
        self._create_workers()
        specs = [
            TrainWorkerSpec(
                cfg=self.cfg,
                rank=rank,
                world_size=self.world_size,
                master_addr=self.master_addr,
                master_port=self.master_port,
            )
            for rank in range(self.world_size)
        ]
        return ray.get(
            [
                worker.initialize.remote(spec)
                for worker, spec in zip(self.workers, specs, strict=True)
            ]
        )

    def states(self) -> list[TrainWorkerState]:
        return ray.get([worker.status.remote() for worker in self.workers])

    def step(
        self,
        per_rank_microbatches: Sequence[Sequence[TensorDict]],
        *,
        global_step: int | None = None,
    ) -> list[TrainStepResult]:
        if len(per_rank_microbatches) != self.world_size:
            raise ValueError(
                "Expected one microbatch sequence per training rank: "
                f"{self.world_size} != {len(per_rank_microbatches)}"
            )
        return ray.get(
            [
                worker.step.remote(
                    per_rank_microbatches[rank],
                    global_step=global_step,
                )
                for rank, worker in enumerate(self.workers)
            ]
        )

    def sleep(self) -> None:
        ray.get([worker.sleep.remote() for worker in self.workers])

    def wakeup(self) -> None:
        ray.get([worker.wakeup.remote() for worker in self.workers])

    def shutdown(self) -> None:
        ray.get([worker.shutdown.remote() for worker in self.workers])
        self.workers.clear()
        ray.util.remove_placement_group(self.pg)

    def _create_placement_group(
        self,
        strategy: PlacementStrategy | str,
    ) -> PlacementGroup:
        bundles = [
            {
                "GPU": self.num_gpus_per_worker,
                "CPU": self.num_cpus_per_worker,
            }
        ] * self.world_size
        pg = placement_group(
            bundles,
            strategy=ray_placement_group_strategy(strategy),
        )
        ray.get(pg.ready())
        return pg

    def _create_workers(self) -> None:
        remote_worker = ray.remote(TrainWorker)
        for rank in range(self.world_size):
            worker = remote_worker.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg,
                    placement_group_bundle_index=rank,
                ),
            ).remote()
            self.workers.append(worker)
