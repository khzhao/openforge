# Copyright 2026 openforge

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Sequence

import torch
import torch.distributed as dist

from openforge.rollout.sglang.client import SGLangClient
from openforge.utils.concurrency import fan_out, submit_fan_out, wait_futures
from openforge.utils.distributed import init_custom_process_group
from openforge.utils.networking import get_free_port
from openforge.utils.packed import (
    build_tensor_bucket_meta,
    flatten_tensor_bucket,
)

__all__ = ["WeightUpdater"]

WeightUpdateMode = Literal["distributed", "disk"]


class WeightUpdater:
    """Coordinate weight syncs between train workers and rollout workers."""

    REQUEST_TIMEOUT_SECONDS = 300.0

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
        workers: Sequence[tuple[str, int]],
        *,
        policy_version: int,
        mode: WeightUpdateMode = "distributed",
    ) -> None:
        normalized_workers = [
            (str(worker_url).rstrip("/"), int(world_size))
            for worker_url, world_size in workers
        ]
        if not normalized_workers:
            return

        normalized_urls = [worker_url for worker_url, _ in normalized_workers]
        clients = [SGLangClient(worker_url) for worker_url in normalized_urls]
        success = False
        try:
            self._call_clients(clients, "pause_generation")
            if mode == "distributed":
                self._sync_distributed(
                    clients,
                    rollout_world_sizes=[
                        world_size for _, world_size in normalized_workers
                    ],
                    policy_version=policy_version,
                )
            elif mode == "disk":
                self._sync_disk(
                    clients,
                    policy_version=policy_version,
                )
            else:
                raise ValueError(f"unsupported weight sync mode: {mode!r}")
            self._assert_version(
                clients,
                worker_urls=normalized_urls,
                policy_version=policy_version,
            )
            success = True
        finally:
            if success:
                self._call_clients(clients, "continue_generation")

    def _call_clients(
        self,
        clients: Sequence[SGLangClient],
        method: str,
        **kwargs: Any,
    ) -> None:
        fan_out(
            [
                lambda client=client: getattr(client, method)(
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                    **kwargs,
                )
                for client in clients
            ]
        )

    def _sync_disk(
        self,
        clients: Sequence[SGLangClient],
        *,
        policy_version: int,
    ) -> None:
        checkpoint_path = self.train_group.export_checkpoint(
            policy_version=policy_version
        )
        self._call_clients(
            clients,
            "update_weights_from_disk",
            model_path=checkpoint_path,
            weight_version=str(policy_version),
        )

    def _sync_distributed(
        self,
        clients: Sequence[SGLangClient],
        *,
        rollout_world_sizes: Sequence[int],
        policy_version: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed weight sync requires CUDA")
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

        process_group = None
        sync_error = None
        cleanup_error = None
        with ThreadPoolExecutor(max_workers=max(1, len(clients))) as executor:
            init_futures = submit_fan_out(
                executor,
                [
                    lambda client=client, start_rank=start_rank: (
                        client.init_weights_update_group(
                            master_address=master_address,
                            master_port=master_port,
                            rank_offset=start_rank,
                            world_size=world_size,
                            group_name=group_name,
                            backend="nccl",
                            timeout=self.REQUEST_TIMEOUT_SECONDS,
                        )
                    )
                    for client, start_rank in zip(
                        clients,
                        self._rollout_rank_offsets(
                            rollout_world_sizes,
                            start=rank_offset,
                        ),
                        strict=True,
                    )
                ],
            )
            try:
                process_group = init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
                wait_futures(init_futures)

                for bucket_index, (meta, bucket) in enumerate(
                    zip(bucket_metas, named_buckets, strict=True)
                ):
                    bucket_on_device = [
                        (name, tensor.to(source_device).contiguous())
                        for name, tensor in bucket
                    ]
                    flattened_bucket = flatten_tensor_bucket(bucket_on_device)
                    flush_cache = bucket_index == num_buckets - 1
                    update_futures = submit_fan_out(
                        executor,
                        [
                            lambda client=client: (
                                client.update_weights_from_distributed(
                                    names=meta["names"],
                                    dtypes=meta["dtypes"],
                                    shapes=meta["shapes"],
                                    group_name=group_name,
                                    load_format="flattened_bucket",
                                    flush_cache=flush_cache,
                                    weight_version=str(policy_version),
                                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                                )
                            )
                            for client in clients
                        ],
                    )
                    dist.broadcast(flattened_bucket, src=0, group=process_group)
                    del bucket_on_device
                    del flattened_bucket
                    torch.cuda.empty_cache()
                    wait_futures(update_futures)
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
                    wait_futures(
                        submit_fan_out(
                            executor,
                            [
                                lambda client=client: (
                                    client.destroy_weights_update_group(
                                        group_name=group_name,
                                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                                    )
                                )
                                for client in clients
                            ],
                        )
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
        clients: Sequence[SGLangClient],
        *,
        worker_urls: Sequence[str],
        policy_version: int,
    ) -> None:
        expected = str(policy_version)
        for index, (client, worker_url) in enumerate(
            zip(clients, worker_urls, strict=True)
        ):
            version = client.get_weight_version(
                timeout=WeightUpdater.REQUEST_TIMEOUT_SECONDS
            )
            if version != expected:
                raise RuntimeError(
                    "rollout worker expected "
                    f"policy_version={expected}, got {version!r} "
                    f"for {worker_url} at index {index}"
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
