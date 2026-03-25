# Copyright 2026 openforge

from __future__ import annotations

import uuid
from typing import Any, Literal, Sequence

import ray
import torch
import torch.distributed as dist

from openforge.utils.distributed import init_custom_process_group
from openforge.utils.networking import get_free_port
from openforge.utils.packed import (
    build_tensor_bucket_meta,
    flatten_tensor_bucket,
)

__all__ = ["WeightUpdater"]

WeightUpdateMode = Literal["auto", "disk", "distributed"]


class WeightUpdater:
    """Coordinate weight syncs between train workers and rollout workers."""

    def __init__(
        self,
        train_group: Any,
        *,
        bucket_bytes: int = 256 << 20,
    ) -> None:
        self.train_group = train_group
        self.bucket_bytes = bucket_bytes

    def sync(
        self,
        rollout_workers: Sequence[Any],
        *,
        policy_version: int,
        mode: WeightUpdateMode = "auto",
    ) -> None:
        success = False
        ray.get([worker.pause_generation.remote() for worker in rollout_workers])
        try:
            last_error: Exception | None = None
            for candidate_mode in self._modes(mode):
                try:
                    if candidate_mode == "distributed":
                        self._sync_distributed(
                            rollout_workers,
                            policy_version=policy_version,
                        )
                    else:
                        self._sync_disk(
                            rollout_workers,
                            policy_version=policy_version,
                        )
                    self._assert_version(
                        rollout_workers,
                        policy_version=policy_version,
                    )
                    success = True
                    return
                except Exception as exc:
                    last_error = exc
            raise RuntimeError("weight sync failed") from last_error
        finally:
            if success:
                ray.get(
                    [worker.continue_generation.remote() for worker in rollout_workers]
                )

    def _sync_disk(
        self,
        rollout_workers: Sequence[Any],
        *,
        policy_version: int,
    ) -> None:
        checkpoint_path = self.train_group.export_checkpoint(
            policy_version=policy_version
        )
        ray.get(
            [
                worker.update_weights_from_disk.remote(
                    model_path=checkpoint_path,
                    policy_version=policy_version,
                )
                for worker in rollout_workers
            ]
        )

    def _sync_distributed(
        self,
        rollout_workers: Sequence[Any],
        *,
        policy_version: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed weight sync requires CUDA")

        rollout_world_sizes = ray.get(
            [worker.distributed_world_size.remote() for worker in rollout_workers]
        )
        named_buckets = self.train_group.build_tensor_buckets(
            bucket_bytes=self.bucket_bytes,
        )
        trainer_slept = False
        self.train_group.sleep()
        trainer_slept = True
        num_buckets = len(named_buckets)
        bucket_metas = [build_tensor_bucket_meta(bucket) for bucket in named_buckets]
        source_device = torch.device("cuda", 0)
        torch.cuda.empty_cache()
        torch.cuda.set_device(source_device)

        group_name = f"openforge-weight-update-{uuid.uuid4().hex[:10]}"
        master_address = "127.0.0.1"
        master_port = get_free_port(start=20000)
        world_size = 1 + sum(rollout_world_sizes)
        rank_offset = 1
        ray.get(
            [
                worker.begin_init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=start_rank,
                    world_size=world_size,
                    group_name=group_name,
                    backend="nccl",
                )
                for worker, start_rank in zip(
                    rollout_workers,
                    self._rollout_rank_offsets(rollout_world_sizes, start=rank_offset),
                    strict=True,
                )
            ]
        )

        process_group = None
        sync_error = None
        cleanup_error = None
        try:
            process_group = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
            )
            ray.get(
                [
                    worker.wait_pending_runtime_call.remote()
                    for worker in rollout_workers
                ]
            )
            for bucket_index, (meta, bucket) in enumerate(
                zip(bucket_metas, named_buckets, strict=True)
            ):
                bucket_on_device = [
                    (name, tensor.to(source_device).contiguous())
                    for name, tensor in bucket
                ]
                flattened_bucket = flatten_tensor_bucket(bucket_on_device)
                flush_cache = bucket_index == num_buckets - 1
                ray.get(
                    [
                        worker.begin_update_weights_from_distributed.remote(
                            names=meta["names"],
                            dtypes=meta["dtypes"],
                            shapes=meta["shapes"],
                            group_name=group_name,
                            policy_version=policy_version,
                            load_format="flattened_bucket",
                            flush_cache=flush_cache,
                        )
                        for worker in rollout_workers
                    ]
                )
                dist.broadcast(flattened_bucket, src=0, group=process_group)
                del bucket_on_device
                del flattened_bucket
                torch.cuda.empty_cache()
                ray.get(
                    [
                        worker.wait_pending_runtime_call.remote()
                        for worker in rollout_workers
                    ]
                )
        except Exception as exc:
            sync_error = exc
            raise
        finally:
            if process_group is not None:
                try:
                    dist.destroy_process_group(process_group)
                except Exception as exc:
                    cleanup_error = exc
            try:
                ray.get(
                    [
                        worker.destroy_weights_update_group.remote(
                            group_name=group_name
                        )
                        for worker in rollout_workers
                    ]
                )
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            if trainer_slept:
                self.train_group.wakeup()
            if sync_error is None and cleanup_error is not None:
                raise cleanup_error

    @staticmethod
    def _assert_version(
        rollout_workers: Sequence[Any],
        *,
        policy_version: int,
    ) -> None:
        versions = ray.get(
            [worker.get_weight_version.remote() for worker in rollout_workers]
        )
        expected = str(policy_version)
        for index, version in enumerate(versions):
            if version != expected:
                raise RuntimeError(
                    f"rollout worker {index} expected policy_version={expected}, got {version!r}"
                )

    @staticmethod
    def _modes(mode: str) -> tuple[str, ...]:
        if mode == "auto":
            return ("distributed", "disk")
        if mode in ("disk", "distributed"):
            return (mode,)
        raise ValueError(
            "unsupported weight sync mode: "
            f"{mode!r}; supported modes are 'auto', 'disk', and 'distributed'"
        )

    @staticmethod
    def _rollout_rank_offsets(
        rollout_world_sizes: Sequence[int],
        *,
        start: int,
    ) -> list[int]:
        offsets: list[int] = []
        for world_size in rollout_world_sizes:
            offsets.append(start)
            start += int(world_size)
        return offsets
