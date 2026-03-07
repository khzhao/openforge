# Copyright 2026 openforge

from __future__ import annotations

from typing import Sequence

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tensordict import TensorDict

from openforge.configs import (
    DistributedPolicyWeights,
    OpenForgeConfig,
    PlacementStrategy,
    SerializedPolicyWeights,
)
from openforge.scheduler.utils import ray_placement_group_strategy
from openforge.scheduler.workers import ActorRefWorker


class ActorRefGroup:
    """Ray actor group for colocated actor-ref workers.

    Args:
        cfg: OpenForge configuration.
        strategy: PACK (default) or SPREAD placement across nodes.
    """

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

    def _create_placement_group(
        self,
        strategy: PlacementStrategy | str,
    ) -> ray.util.placement_group.PlacementGroup:
        """Reserve GPU+CPU bundles atomically via Ray."""
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
        """Pin one remote worker per bundle in the placement group."""
        RemoteWorker = ray.remote(ActorRefWorker)

        for rank in range(self._world_size):
            worker = RemoteWorker.options(
                num_cpus=self._num_cpus_per_worker,
                num_gpus=self._num_gpus_per_worker,
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
        batches: Sequence[TensorDict],
        *,
        global_step: int | None = None,
    ) -> list[ray.ObjectRef]:
        """Dispatch one rank-local training batch to each worker."""
        if len(batches) != self._world_size:
            raise ValueError(
                f"Expected {self._world_size} rank-local batches, got {len(batches)}"
            )

        return [
            worker.train_step.remote(batches[rank], global_step=global_step)
            for rank, worker in enumerate(self._workers)
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

    def export_policy_weights_for_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> SerializedPolicyWeights:
        """Export live SGLang weight-update payload from the training world."""
        results = ray.get(
            [
                worker.export_policy_weights_for_rollout.remote(
                    step=step,
                    policy_version=policy_version,
                )
                for worker in self._workers
            ]
        )
        for result in results:
            if result is not None:
                return result
        raise RuntimeError("training export did not return rollout weight payloads")

    def prepare_policy_weights_for_distributed_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> DistributedPolicyWeights:
        """Prepare metadata-only payloads for distributed rollout sync."""
        results = ray.get(
            [
                worker.prepare_policy_weights_for_distributed_rollout.remote(
                    step=step,
                    policy_version=policy_version,
                )
                for worker in self._workers
            ]
        )
        for result in results:
            if result is not None:
                return result
        raise RuntimeError(
            "training export did not return distributed rollout weight payloads"
        )

    def init_policy_weights_update_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        ray.get(
            self._workers[0].init_policy_weights_update_group.remote(
                master_addr=master_addr,
                master_port=master_port,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )
        )

    def async_broadcast_prepared_policy_weights_bucket(
        self,
        *,
        bucket_index: int,
        group_name: str,
    ) -> ray.ObjectRef:
        return self._workers[0].broadcast_prepared_policy_weights_bucket.remote(
            bucket_index=bucket_index,
            group_name=group_name,
        )

    def destroy_policy_weights_update_group(self, *, group_name: str) -> None:
        ray.get(
            self._workers[0].destroy_policy_weights_update_group.remote(
                group_name=group_name,
            )
        )

    def clear_prepared_policy_weights_for_rollout(self) -> None:
        ray.get(self._workers[0].clear_prepared_policy_weights_for_rollout.remote())

    def allocate_free_port(self, *, start: int = 40000) -> int:
        return ray.get(self._workers[0].allocate_free_port.remote(start=start))

    def sync_policy_weights_to_rollout(
        self,
        *,
        rollout_workers: list[ray.actor.ActorHandle],
        rollout_engines: list[object],
        policy_version: int,
        sync_mode: str,
    ) -> None:
        ray.get(
            [
                worker.sync_policy_weights_to_rollout.remote(
                    rollout_workers=rollout_workers,
                    rollout_engines=rollout_engines,
                    policy_version=policy_version,
                    sync_mode=sync_mode,
                )
                for worker in self._workers
            ]
        )

    def sleep(self) -> None:
        """Offload all workers to CPU."""
        ray.get([worker.sleep.remote() for worker in self._workers])

    def wakeup(self) -> None:
        """Restore all workers to GPU."""
        ray.get([worker.wakeup.remote() for worker in self._workers])

    def clear_memory(self) -> None:
        """Release temporary GPU memory on all workers."""
        ray.get([worker.clear_memory.remote() for worker in self._workers])
