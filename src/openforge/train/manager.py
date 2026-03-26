# Copyright 2026 openforge

import threading
from typing import Any, Sequence

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.router.client import RolloutRouterClient
from openforge.train.fsdp2.weight_updater import WeightUpdater
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.train.worker import RayTrainWorker
from openforge.utils.nccl import apply_nccl_env_defaults

__all__ = ["TrainManager"]


class TrainManager:
    """Ray actor group for one train worker per rank."""

    def __init__(self) -> None:
        self.workers = []
        self.rollout_router_client: RolloutRouterClient | None = None
        self._sync_lock = threading.RLock()
        self._latest_stable_policy_version = 0

    def initialize(
        self,
        cfg: OpenForgeConfig,
        master_addr: str,
        master_port: int,
        placement_group: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ) -> list[TrainWorkerState]:
        assert "actor" in placement_group, "actor placement group must be provided"
        nccl_env_vars = apply_nccl_env_defaults()

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
                runtime_env={"env_vars": dict(nccl_env_vars)},
            ).remote()
            for bundle_index in self.reordered_bundle_indices
        ]
        try:
            states = ray.get(
                [
                    worker.initialize.remote(spec)
                    for worker, spec in zip(self.workers, specs, strict=True)
                ]
            )
            return states
        except Exception:
            self.shutdown()
            raise

    def states(self) -> list[TrainWorkerState]:
        return ray.get([worker.status.remote() for worker in self.workers])

    def step_update(
        self,
        rank_minibatches_per_update: Sequence[Sequence[dict[str, torch.Tensor]]],
        *,
        global_step: int | None = None,
    ) -> list[list[TrainStepResult]]:
        rank_updates = self._build_rank_updates(rank_minibatches_per_update)
        with self._sync_lock:
            return self._step_update_locked(
                rank_updates,
                global_step=global_step,
            )

    def step_update_and_publish(
        self,
        rank_minibatches_per_update: Sequence[Sequence[dict[str, torch.Tensor]]],
        *,
        global_step: int | None = None,
        policy_version: int,
    ) -> list[list[TrainStepResult]]:
        rank_updates = self._build_rank_updates(rank_minibatches_per_update)
        with self._sync_lock:
            results = self._step_update_locked(
                rank_updates,
                global_step=global_step,
            )
            rollout_router_client = self._publish_rollout_policy_version_locked(
                policy_version
            )
        if rollout_router_client is None:
            return results
        rollout_router_client.receive_policy_version(
            policy_version=policy_version,
        )
        return results

    def sleep(self) -> None:
        with self._sync_lock:
            ray.get([worker.sleep.remote() for worker in self.workers])

    def wakeup(self) -> None:
        with self._sync_lock:
            ray.get([worker.wakeup.remote() for worker in self.workers])

    def build_tensor_buckets(
        self,
        *,
        bucket_bytes: int,
    ) -> list[list[tuple[str, torch.Tensor]]]:
        with self._sync_lock:
            results = ray.get(
                [
                    worker.build_tensor_buckets.remote(bucket_bytes=bucket_bytes)
                    for worker in self.workers
                ]
            )
        result = results[0]
        assert result is not None, "publisher rank returned no tensor buckets"
        return result

    def export_checkpoint(self, *, policy_version: int) -> str:
        with self._sync_lock:
            results = ray.get(
                [
                    worker.export_checkpoint.remote(policy_version=policy_version)
                    for worker in self.workers
                ]
            )
        result = results[0]
        assert result is not None, "publisher rank returned no checkpoint path"
        return result

    def register_rollout(
        self,
        rollout_router_url: str,
        *,
        train_server_url: str,
    ) -> None:
        self.rollout_router_client = RolloutRouterClient(rollout_router_url)
        self.rollout_router_client.register_train_server(
            train_server_url=train_server_url,
        )

    def publish_rollout_policy_version(self, policy_version: int) -> None:
        with self._sync_lock:
            rollout_router_client = self._publish_rollout_policy_version_locked(
                policy_version
            )
        if rollout_router_client is None:
            return
        rollout_router_client.receive_policy_version(policy_version=policy_version)

    @property
    def latest_stable_policy_version(self) -> int:
        return self._latest_stable_policy_version

    def respond_to_weight_sync_request(
        self,
        *,
        workers: Sequence[tuple[str, int]],
        target_version: int,
        mode: str = "distributed",
    ) -> dict[str, Any]:
        normalized_workers = [
            (str(worker_url).rstrip("/"), int(world_size))
            for worker_url, world_size in workers
        ]
        if not normalized_workers:
            return {
                "target_version": target_version,
                "latest_stable_policy_version": self._latest_stable_policy_version,
                "results": [],
            }

        if not self._sync_lock.acquire(blocking=False):
            return {
                "target_version": target_version,
                "latest_stable_policy_version": self._latest_stable_policy_version,
                "results": [
                    {
                        "worker_url": worker_url,
                        "ok": False,
                        "error": "training_in_progress",
                    }
                    for worker_url, _ in normalized_workers
                ],
            }

        try:
            latest_stable_version = self._latest_stable_policy_version
            if target_version != latest_stable_version:
                return {
                    "target_version": target_version,
                    "latest_stable_policy_version": latest_stable_version,
                    "results": [
                        {
                            "worker_url": worker_url,
                            "ok": False,
                            "error": (
                                "target_version does not match the latest stable "
                                f"version: {target_version} != {latest_stable_version}"
                            ),
                        }
                        for worker_url, _ in normalized_workers
                    ],
                }

            try:
                WeightUpdater(self).sync(
                    normalized_workers,
                    policy_version=target_version,
                    mode=mode,
                )
            except Exception as exc:
                results = [
                    {
                        "worker_url": worker_url,
                        "ok": False,
                        "error": str(exc),
                    }
                    for worker_url, _ in normalized_workers
                ]
            else:
                results = [
                    {
                        "worker_url": worker_url,
                        "ok": True,
                    }
                    for worker_url, _ in normalized_workers
                ]

            return {
                "target_version": target_version,
                "latest_stable_policy_version": latest_stable_version,
                "results": results,
            }
        finally:
            self._sync_lock.release()

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

    def _build_rank_updates(
        self,
        rank_minibatches_per_update: Sequence[Sequence[dict[str, torch.Tensor]]],
    ) -> list[list[dict[str, torch.Tensor]]]:
        rank_updates = [[] for _ in range(self.world_size)]
        for rank_minibatches in rank_minibatches_per_update:
            assert len(rank_minibatches) == self.world_size, (
                "Expected one mini-batch per training rank: "
                f"{self.world_size} != {len(rank_minibatches)}"
            )
            for rank, mini_batch in enumerate(rank_minibatches):
                rank_updates[rank].append(mini_batch)
        return rank_updates

    def _step_update_locked(
        self,
        rank_updates: Sequence[Sequence[dict[str, torch.Tensor]]],
        *,
        global_step: int | None = None,
    ) -> list[list[TrainStepResult]]:
        return ray.get(
            [
                worker.step_update.remote(
                    rank_updates[rank],
                    global_step=global_step,
                )
                for rank, worker in enumerate(self.workers)
            ]
        )

    def _publish_rollout_policy_version_locked(
        self,
        policy_version: int,
    ) -> RolloutRouterClient | None:
        assert policy_version >= 0
        assert policy_version >= self._latest_stable_policy_version
        self._latest_stable_policy_version = policy_version
        return self.rollout_router_client
