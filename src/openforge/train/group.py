# Copyright 2026 openforge

from typing import Sequence

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tensordict import TensorDict

from openforge.configs.models import OpenForgeConfig
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.train.worker import TrainWorker

__all__ = ["TrainWorkerGroup"]


class TrainWorkerGroup:
    """Ray actor group for one train worker per rank."""

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
            TrainWorker.options(
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

    def shutdown(self) -> None:
        ray.get([worker.shutdown.remote() for worker in self.workers])
        self.workers.clear()
        ray.util.remove_placement_group(self.pg)
