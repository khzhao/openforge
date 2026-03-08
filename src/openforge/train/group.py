# Copyright 2026 openforge

from typing import Sequence

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tensordict import TensorDict

from openforge.configs.models import OpenForgeConfig
from openforge.configs.topology import PlacementStrategy
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.train.types import (
    CheckpointInfo,
    TrainStepResult,
    TrainWorkerSpec,
    TrainWorkerState,
)
from openforge.train.worker import TrainWorker
from openforge.utils.ray import ray_placement_group_strategy

__all__ = ["TrainWorkerGroup"]


class TrainWorkerGroup:
    """Ray actor group for one train worker per rank."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        master_addr: str,
        master_port: int,
        strategy: PlacementStrategy | str = PlacementStrategy.PACK,
    ) -> None:
        self.cfg = cfg
        self._master_addr = master_addr
        self._master_port = master_port

        resolved = cfg.train.resolve(cfg.cluster)
        self._num_nodes = resolved.num_nodes
        self._num_gpus_per_node = resolved.num_gpus_per_node
        self._num_gpus_per_worker = 1
        self._num_cpus_per_worker = resolved.cpus_per_worker
        self._world_size = resolved.world_size
        self._workers: list[ray.actor.ActorHandle] = []

        self._pg = self._create_placement_group(strategy)
        self._create_workers()

    def initialize(self) -> list[TrainWorkerState]:
        specs = [
            TrainWorkerSpec(
                cfg=self.cfg,
                rank=rank,
                world_size=self._world_size,
                master_addr=self._master_addr,
                master_port=self._master_port,
            )
            for rank in range(self._world_size)
        ]
        return ray.get(
            [
                worker.initialize.remote(spec)
                for worker, spec in zip(self._workers, specs, strict=True)
            ]
        )

    def states(self) -> list[TrainWorkerState]:
        return ray.get([worker.status.remote() for worker in self._workers])

    def step(
        self,
        per_rank_microbatches: Sequence[Sequence[TensorDict]],
        *,
        global_step: int | None = None,
    ) -> list[TrainStepResult]:
        if len(per_rank_microbatches) != self._world_size:
            raise ValueError(
                "Expected one microbatch sequence per training rank: "
                f"{self._world_size} != {len(per_rank_microbatches)}"
            )

        return ray.get(
            [
                worker.step.remote(
                    per_rank_microbatches[rank],
                    global_step=global_step,
                )
                for rank, worker in enumerate(self._workers)
            ]
        )

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> list[CheckpointInfo]:
        return ray.get(
            [
                worker.save_checkpoint.remote(
                    step=step,
                    policy_version=policy_version,
                    save_optimizer=save_optimizer,
                )
                for worker in self._workers
            ]
        )

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> CheckpointInfo | None:
        results = ray.get(
            [
                worker.load_checkpoint.remote(
                    latest=latest,
                    step=step,
                    load_optimizer=load_optimizer,
                )
                for worker in self._workers
            ]
        )
        return results[0]

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef:
        results = ray.get(
            [
                worker.export_policy_artifact.remote(
                    step=step,
                    policy_version=policy_version,
                )
                for worker in self._workers
            ]
        )
        for result in results:
            if result is not None:
                return result
        raise RuntimeError("training export did not return a policy artifact")

    def push_tensor_update(
        self,
        session: TensorUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        ray.get(
            [
                worker.push_tensor_update.remote(
                    session,
                    step=step,
                    policy_version=policy_version,
                )
                for worker in self._workers
            ]
        )

    def push_distributed_update(
        self,
        session: DistributedUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        ray.get(
            [
                worker.push_distributed_update.remote(
                    session,
                    step=step,
                    policy_version=policy_version,
                )
                for worker in self._workers
            ]
        )

    def sleep(self) -> None:
        ray.get([worker.sleep.remote() for worker in self._workers])

    def wakeup(self) -> None:
        ray.get([worker.wakeup.remote() for worker in self._workers])

    def clear_memory(self) -> None:
        ray.get([worker.clear_memory.remote() for worker in self._workers])

    def shutdown(self) -> None:
        ray.get([worker.shutdown.remote() for worker in self._workers])
        self._workers.clear()
        ray.util.remove_placement_group(self._pg)

    def _create_placement_group(
        self,
        strategy: PlacementStrategy | str,
    ) -> PlacementGroup:
        bundles = [
            {"GPU": self._num_gpus_per_worker, "CPU": self._num_cpus_per_worker}
        ] * self._world_size
        pg = placement_group(
            bundles,
            strategy=ray_placement_group_strategy(strategy),
        )
        ray.get(pg.ready())
        return pg

    def _create_workers(self) -> None:
        remote_worker = ray.remote(TrainWorker)
        for rank in range(self._world_size):
            worker = remote_worker.options(
                num_cpus=self._num_cpus_per_worker,
                num_gpus=self._num_gpus_per_worker,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=rank,
                ),
            ).remote()
            self._workers.append(worker)
