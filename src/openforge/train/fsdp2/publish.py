# Copyright 2026 openforge

import abc
import logging
import socket
from collections.abc import Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

from openforge.configs.models import OpenForgeConfig
from openforge.policy.types import (
    DistributedUpdateSession,
    TensorUpdateSession,
    WeightBucketMeta,
)

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

logger = logging.getLogger(__name__)


class WeightUpdater(abc.ABC):
    """Base class for live FSDP2 -> SGLang weight synchronization."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        model: torch.nn.Module,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        master_addr: str,
        session: TensorUpdateSession | DistributedUpdateSession | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_addr = master_addr
        self.session = session
        self.weight_version = 0

    @property
    def _buffer_size(self) -> int:
        return self.cfg.train.backend_config.update_weight_buffer_size

    def update_weights(self, *, policy_version: int) -> None:
        """Materialize one bucket at a time and push it into rollout."""
        self.weight_version = policy_version
        bucket: list[tuple[str, Any]] = []
        bucket_size = 0

        try:
            self.before_update()
            for name, param in self.model.state_dict().items():
                if not isinstance(param, torch.Tensor):
                    continue
                param_size = param.numel() * param.element_size()
                if bucket and bucket_size + param_size >= self._buffer_size:
                    self.wait_and_update_bucket(bucket)
                    bucket = []
                    bucket_size = 0

                prepared = self._prepare_tensor(param)
                bucket.append((name, prepared))
                bucket_size += param_size

            if bucket:
                self.wait_and_update_bucket(bucket)
        finally:
            try:
                self.after_update()
            finally:
                self.disconnect_rollout_engines()

    def _prepare_tensor(self, tensor: torch.Tensor) -> Any:
        tensor = tensor.to(device=self.device)
        if isinstance(tensor, DTensor):
            tensor = tensor.redistribute(
                placements=[Replicate()] * tensor.device_mesh.ndim,
                async_op=True,
            ).to_local()
        return tensor

    def wait_and_update_bucket(self, bucket: list[tuple[str, Any]]) -> None:
        resolved_bucket = [
            (name, tensor.wait()) if hasattr(tensor, "wait") else (name, tensor)
            for name, tensor in bucket
        ]
        self.update_bucket_weights(resolved_bucket)

    def before_update(self) -> None:
        """Subclass hook for update initialization."""

    def after_update(self) -> None:
        """Subclass hook for update cleanup."""

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        *,
        rollout_workers: Sequence[ray.actor.ActorHandle],
        rollout_engines: Sequence[Any],
    ) -> None:
        """Attach rollout engines for a subsequent update."""

    @abc.abstractmethod
    def disconnect_rollout_engines(self) -> None:
        """Release rollout-specific state after an update."""

    @abc.abstractmethod
    def update_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Push one resolved bucket into rollout."""


class TensorWeightUpdater(WeightUpdater):
    """Push buckets through SGLang's tensor RPC on same-node-pool engines."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rollout_worker: ray.actor.ActorHandle | None = None
        self._gather_group = None
        self._gather_group_size = 0
        self._gather_src_rank: int | None = None
        self._created_groups: list[Any] = []

    def connect_rollout_engines(
        self,
        *,
        rollout_workers: Sequence[ray.actor.ActorHandle],
        rollout_engines: Sequence[Any],
    ) -> None:
        self.disconnect_rollout_engines()

        engine_gpu_counts = [engine.gpus_per_engine for engine in rollout_engines]
        total_engine_gpus = sum(engine_gpu_counts)
        if total_engine_gpus > self.world_size:
            raise ValueError(
                "tensor weight sync requires rollout GPUs on the shared node_pool "
                f"to fit within the training world size: {total_engine_gpus} > {self.world_size}"
            )

        offset = 0
        for worker, count in zip(rollout_workers, engine_gpu_counts, strict=True):
            group_ranks = list(range(offset, offset + count))
            offset += count
            group = dist.new_group(ranks=group_ranks, backend="gloo")
            self._created_groups.append(group)
            if self.rank in group_ranks:
                self._rollout_worker = worker
                self._gather_group = group
                self._gather_group_size = len(group_ranks)
                self._gather_src_rank = group_ranks[0]

    def disconnect_rollout_engines(self) -> None:
        for group in self._created_groups:
            try:
                dist.destroy_process_group(group)
            except Exception:
                logger.debug(
                    "failed to destroy tensor sync process group", exc_info=True
                )
        self._created_groups.clear()
        self._rollout_worker = None
        self._gather_group = None
        self._gather_group_size = 0
        self._gather_src_rank = None

    def before_update(self) -> None:
        if self.session is not None:
            return
        if self._rollout_worker is None or self.rank != self._gather_src_rank:
            return
        ray.get(
            self._rollout_worker.begin_tensor_update.remote(
                TensorUpdateSession(
                    session_id=f"tensor-{self.weight_version}",
                    policy_version=self.weight_version,
                    load_format="flattened_bucket",
                    engine_ids=[],
                )
            )
        )

    def after_update(self) -> None:
        if self.session is not None:
            return
        if self._rollout_worker is None or self.rank != self._gather_src_rank:
            return
        ray.get(
            self._rollout_worker.finish_tensor_update.remote(
                TensorUpdateSession(
                    session_id=f"tensor-{self.weight_version}",
                    policy_version=self.weight_version,
                    load_format="flattened_bucket",
                    engine_ids=[],
                )
            )
        )

    def update_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        if self._gather_group is None or self._rollout_worker is None:
            return

        from openforge.rollout.sglang.serialization import (
            serialize_named_tensors_for_sglang,
        )

        monkey_patch_torch_reductions()
        serialized_tensors = serialize_named_tensors_for_sglang(named_tensors)
        if self._gather_src_rank == self.rank:
            gathered_batches = [None] * self._gather_group_size
        else:
            gathered_batches = None

        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_batches,
            dst=self._gather_src_rank,
            group=self._gather_group,
        )

        if self.rank != self._gather_src_rank:
            return

        assert gathered_batches is not None
        if not gathered_batches or not gathered_batches[0]:
            return

        num_payloads = len(gathered_batches[0])
        for payload_index in range(num_payloads):
            ray.get(
                self._rollout_worker.apply_tensor_bucket.remote(
                    serialized_named_tensors=[
                        rank_payloads[payload_index]
                        for rank_payloads in gathered_batches
                    ],
                    load_format="flattened_bucket",
                    policy_version=self.weight_version,
                )
            )


class DistributedWeightUpdater(WeightUpdater):
    """Push metadata over RPC and bucket payloads over a temporary process group."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rollout_workers: list[ray.actor.ActorHandle] = []
        self._rollout_engines: list[Any] = []
        self._process_group = None
        self._group_name: str | None = None
        self._backend = "nccl" if self.device.type == "cuda" else "gloo"

    def connect_rollout_engines(
        self,
        *,
        rollout_workers: Sequence[ray.actor.ActorHandle],
        rollout_engines: Sequence[Any],
    ) -> None:
        self._rollout_workers = list(rollout_workers)
        self._rollout_engines = list(rollout_engines)

    def disconnect_rollout_engines(self) -> None:
        self._rollout_workers.clear()
        self._rollout_engines.clear()

    def before_update(self) -> None:
        if self.rank != 0 or not self._rollout_workers:
            return

        from sglang.srt.utils.common import init_custom_process_group

        refs = []
        if isinstance(self.session, DistributedUpdateSession):
            self._group_name = self.session.group_name
            master_addr = self.session.master_addr
            master_port = self.session.master_port
            world_size = self.session.world_size
            backend = self.session.backend
        else:
            counts = [engine.gpus_per_engine for engine in self._rollout_engines]
            cumulative = [0]
            for count in counts:
                cumulative.append(cumulative[-1] + count)

            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            self._group_name = f"openforge-rollout-sync-{self.weight_version}"
            world_size = 1 + cumulative[-1]
            master_addr = self.master_addr
            backend = self._backend
            for index, worker in enumerate(self._rollout_workers):
                refs.append(
                    worker.begin_distributed_update.remote(
                        DistributedUpdateSession(
                            session_id=self._group_name,
                            policy_version=self.weight_version,
                            load_format="flattened_bucket",
                            engine_ids=[],
                            master_addr=master_addr,
                            master_port=master_port,
                            group_name=self._group_name,
                            world_size=world_size,
                            backend=backend,
                            rank_offsets={},
                        )
                    )
                )

        self._process_group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=self._group_name,
            device_id=self.device if self.device.type == "cuda" else None,
        )
        ray.get(refs)

    def after_update(self) -> None:
        if self.rank != 0:
            return

        refs = []
        if self._group_name is not None and self.session is None:
            refs = [
                worker.finish_distributed_update.remote(
                    DistributedUpdateSession(
                        session_id=self._group_name,
                        policy_version=self.weight_version,
                        load_format="flattened_bucket",
                        engine_ids=[],
                        master_addr=self.master_addr,
                        master_port=0,
                        group_name=self._group_name,
                        world_size=0,
                        backend=self._backend,
                        rank_offsets={},
                    )
                )
                for worker in self._rollout_workers
            ]
        if refs:
            ray.get(refs)
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None
        self._group_name = None

    def update_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        if self.rank != 0 or not named_tensors or self._group_name is None:
            return

        bucket = WeightBucketMeta(
            names=[name for name, _ in named_tensors],
            dtypes=[
                str(tensor.dtype).replace("torch.", "") for _, tensor in named_tensors
            ],
            shapes=[list(tensor.shape) for _, tensor in named_tensors],
        )
        refs = [
            worker.apply_distributed_bucket.remote(
                bucket=bucket,
                policy_version=self.weight_version,
                load_format="flattened_bucket",
                group_name=self._group_name,
            )
            for worker in self._rollout_workers
        ]

        flattened_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        dist.broadcast(
            flattened_bucket.get_flattened_tensor(),
            src=0,
            group=self._process_group,
        )
        ray.get(refs)
