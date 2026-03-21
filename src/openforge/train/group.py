# Copyright 2026 openforge

from typing import Sequence

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.train.fsdp2.weight_updater import WeightUpdater
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.train.worker import RayTrainWorker

__all__ = ["TrainManager"]


class TrainManager:
    """Ray actor group for one train worker per rank."""

    def __init__(self) -> None:
        self.workers = []
        self.rollout_workers: Sequence[object] | None = None

    def initialize(
        self,
        cfg: OpenForgeConfig,
        master_addr: str,
        master_port: int,
        placement_group: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ) -> list[TrainWorkerState]:
        assert "actor" in placement_group, "actor placement group must be provided"

        self.cfg = cfg
        self.master_addr = master_addr
        self.master_port = master_port
        self.pg, self.reordered_bundle_indices, self.reordered_gpu_ids = (
            placement_group["actor"]
        )

        self.num_cpus_per_worker = cfg.train.cpus_per_worker
        self.num_gpus_per_worker = 1
        self.world_size = cfg.train.num_workers
        assert len(self.reordered_bundle_indices) == self.world_size, (
            f"expected one bundle index per training worker: "
            f"{self.world_size} != {len(self.reordered_bundle_indices)}"
        )
        assert len(self.reordered_gpu_ids) == self.world_size, (
            f"expected one GPU ID per training worker: "
            f"{self.world_size} != {len(self.reordered_gpu_ids)}"
        )

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
        self.workers = [
            RayTrainWorker.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg,
                    placement_group_bundle_index=bundle_index,
                ),
            ).remote()
            for bundle_index in self.reordered_bundle_indices
        ]
        states = ray.get(
            [
                worker.initialize.remote(spec)
                for worker, spec in zip(self.workers, specs, strict=True)
            ]
        )
        return states

    def states(self) -> list[TrainWorkerState]:
        return ray.get([worker.status.remote() for worker in self.workers])

    def step(
        self,
        rank_minibatches: Sequence[dict[str, torch.Tensor]],
        *,
        global_step: int | None = None,
    ) -> list[TrainStepResult]:
        assert len(rank_minibatches) == self.world_size, (
            "Expected one mini-batch per training rank: "
            f"{self.world_size} != {len(rank_minibatches)}"
        )
        return ray.get(
            [
                worker.step.remote(
                    rank_minibatches[rank],
                    global_step=global_step,
                )
                for rank, worker in enumerate(self.workers)
            ]
        )

    def step_update(
        self,
        rank_minibatches_per_update: Sequence[Sequence[dict[str, torch.Tensor]]],
        *,
        global_step: int | None = None,
    ) -> list[list[TrainStepResult]]:
        rank_updates = [[] for _ in range(self.world_size)]
        for rank_minibatches in rank_minibatches_per_update:
            assert len(rank_minibatches) == self.world_size, (
                "Expected one mini-batch per training rank: "
                f"{self.world_size} != {len(rank_minibatches)}"
            )
            for rank, mini_batch in enumerate(rank_minibatches):
                rank_updates[rank].append(mini_batch)
        return ray.get(
            [
                worker.step_update.remote(
                    rank_updates[rank],
                    global_step=global_step,
                )
                for rank, worker in enumerate(self.workers)
            ]
        )

    def sleep(self) -> None:
        ray.get([worker.sleep.remote() for worker in self.workers])

    def wakeup(self) -> None:
        ray.get([worker.wakeup.remote() for worker in self.workers])

    def build_tensor_buckets(
        self,
        *,
        bucket_bytes: int,
    ) -> list[list[tuple[str, torch.Tensor]]]:
        results = ray.get(
            [
                worker.build_tensor_buckets.remote(bucket_bytes=bucket_bytes)
                for worker in self.workers
            ]
        )
        result = results[0]
        assert result is not None, "publisher rank returned no tensor buckets"
        return result

    def push_weights_to_rollouts_from_tensor(
        self,
        *,
        rollout_workers: Sequence[object],
        policy_version: int,
        bucket_bytes: int,
    ) -> bool:
        if self.world_size <= 1 or not torch.cuda.is_available():
            return False

        rollout_world_sizes = ray.get(
            [worker.distributed_world_size.remote() for worker in rollout_workers]
        )
        if self.world_size != sum(int(size) for size in rollout_world_sizes):
            return False

        train_node_ips = ray.get(
            [worker.node_ip_address.remote() for worker in self.workers]
        )
        rollout_node_ips = ray.get(
            [worker.node_ip_address.remote() for worker in rollout_workers]
        )
        if (
            len(set(train_node_ips)) != 1
            or len(set(rollout_node_ips)) != 1
            or train_node_ips[0] != rollout_node_ips[0]
        ):
            return False

        ray.get(
            [
                worker.push_weights_to_rollouts_from_tensor.remote(
                    rollout_workers=rollout_workers,
                    rollout_world_sizes=rollout_world_sizes,
                    policy_version=policy_version,
                    bucket_bytes=bucket_bytes,
                )
                for worker in self.workers
            ]
        )
        return True

    def export_checkpoint(self, *, policy_version: int) -> str:
        results = ray.get(
            [
                worker.export_checkpoint.remote(policy_version=policy_version)
                for worker in self.workers
            ]
        )
        result = results[0]
        assert result is not None, "publisher rank returned no checkpoint path"
        return result

    def register_rollout(self, rollout_workers: Sequence[object]) -> None:
        self.rollout_workers = list(rollout_workers)

    def sync_rollout_weights(
        self,
        *,
        policy_version: int,
        mode: str = "auto",
        bucket_bytes: int = 256 << 20,
    ) -> None:
        if self.rollout_workers is None:
            raise RuntimeError("rollout must be registered before syncing weights")

        WeightUpdater(self, bucket_bytes=bucket_bytes).sync(
            self.rollout_workers,
            policy_version=policy_version,
            mode=mode,
        )

    def shutdown(self) -> None:
        workers = list(self.workers)
        try:
            if workers:
                ray.get([worker.shutdown.remote() for worker in workers])
        finally:
            for worker in workers:
                ray.kill(worker)
            self.workers.clear()
            ray.util.remove_placement_group(self.pg)
