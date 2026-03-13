# Copyright 2026 openforge

import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import ray
import torch
import torch.distributed as dist
from huggingface_hub import save_torch_state_dict
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer

from openforge.train.fsdp2.base import FSDP2Engine
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.utils.distributed import init_gloo_group
from openforge.utils.packed import serialize_tensor_bucket
from openforge.utils.ray import get_current_ray_node_ip_address
from openforge.utils.torch import get_torch_dtype

__all__ = ["TrainWorker"]


class TrainWorker:
    """Thin Ray-facing wrapper over one FSDP2 engine."""

    def initialize(self, spec: TrainWorkerSpec) -> TrainWorkerState:
        backend_name = spec.cfg.train.backend
        if backend_name == "megatron":
            raise NotImplementedError("Megatron train workers are not implemented yet")
        if backend_name != "fsdp2":
            raise ValueError(f"Unsupported backend: {backend_name}")

        self.spec = spec
        self.engine = FSDP2Engine(spec)
        self.state = TrainWorkerState(
            rank=spec.rank,
            world_size=spec.world_size,
            backend=backend_name,
            device=str(self.engine.device),
            sleeping=False,
        )
        return self.state

    def step(
        self,
        microbatches: Sequence[TensorDict],
        *,
        global_step: int | None = None,
    ) -> TrainStepResult:
        if not microbatches:
            raise ValueError("step requires at least one microbatch")

        self.engine.zero_grad()
        last_index = len(microbatches) - 1
        for index, batch in enumerate(microbatches):
            context = self.engine.no_sync() if index < last_index else nullcontext()
            with context:
                outputs = self.engine.forward(batch)
                self.engine.backward(outputs)

        metrics = self.engine.step_optimizer()
        return TrainStepResult(
            rank=self.spec.rank,
            global_step=global_step,
            metrics={
                **metrics,
                "global_step": -1.0 if global_step is None else float(global_step),
            },
        )

    def sleep(self) -> None:
        self.engine.sleep()
        self.state.sleeping = True

    def wakeup(self) -> None:
        self.engine.wakeup()
        self.state.sleeping = False

    def status(self) -> TrainWorkerState:
        return self.state

    def build_tensor_buckets(
        self,
        *,
        bucket_bytes: int,
    ) -> list[list[tuple[str, torch.Tensor]]] | None:
        named_buckets = self._build_named_tensor_buckets(bucket_bytes=bucket_bytes)
        if self.spec.rank != 0:
            return None
        return named_buckets

    def export_checkpoint(self, *, policy_version: int) -> str | None:
        state_dict = get_model_state_dict(
            self.engine.main_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.spec.rank != 0:
            return None

        checkpoint_dir = (
            Path(self.spec.cfg.train.checkpoints)
            / f"policy-{policy_version}-{uuid.uuid4().hex[:8]}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_torch_state_dict(state_dict, checkpoint_dir)
        self.engine.main_model.config.save_pretrained(checkpoint_dir)
        generation_config = getattr(self.engine.main_model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(checkpoint_dir)
        AutoTokenizer.from_pretrained(
            self.spec.cfg.model.tokenizer_name_or_path,
            trust_remote_code=True,
        ).save_pretrained(checkpoint_dir)
        return str(checkpoint_dir)

    @staticmethod
    def node_ip_address() -> str:
        return get_current_ray_node_ip_address()

    def push_weights_to_rollouts_from_tensor(
        self,
        *,
        rollout_workers: Sequence[Any],
        rollout_world_sizes: Sequence[int],
        policy_version: int,
        bucket_bytes: int,
    ) -> None:
        serialized_buckets = self._build_rank_local_serialized_tensor_buckets(
            bucket_bytes=bucket_bytes
        )
        gloo_group = init_gloo_group()
        bucket_counts = [None] * self.spec.world_size
        dist.all_gather_object(bucket_counts, len(serialized_buckets), group=gloo_group)
        if any(count != bucket_counts[0] for count in bucket_counts):
            raise RuntimeError(
                f"tensor sync bucket count mismatch across train ranks: {bucket_counts}"
            )

        for bucket_index, serialized_bucket in enumerate(serialized_buckets):
            gathered_buckets = (
                [None] * self.spec.world_size if self.spec.rank == 0 else None
            )
            dist.gather_object(
                obj=serialized_bucket,
                object_gather_list=gathered_buckets,
                dst=0,
                group=gloo_group,
            )

            bucket_error = None
            if self.spec.rank == 0:
                try:
                    assert gathered_buckets is not None
                    flush_cache = bucket_index == len(serialized_buckets) - 1
                    refs = []
                    rank_offset = 0
                    for rollout_worker, rollout_world_size in zip(
                        rollout_workers,
                        rollout_world_sizes,
                        strict=True,
                    ):
                        next_rank_offset = rank_offset + int(rollout_world_size)
                        refs.append(
                            rollout_worker.update_weights_from_tensor.remote(
                                serialized_named_tensors=gathered_buckets[
                                    rank_offset:next_rank_offset
                                ],
                                policy_version=policy_version,
                                load_format="flattened_bucket",
                                flush_cache=flush_cache,
                            )
                        )
                        rank_offset = next_rank_offset
                    if rank_offset != len(gathered_buckets):
                        raise RuntimeError(
                            "tensor sync expected one gathered payload per train rank"
                        )
                    ray.get(refs)
                except Exception as exc:  # noqa: BLE001
                    bucket_error = repr(exc)

            bucket_errors = [None] * self.spec.world_size
            dist.all_gather_object(bucket_errors, bucket_error, group=gloo_group)
            first_error = next(
                (error for error in bucket_errors if error is not None),
                None,
            )
            if first_error is not None:
                raise RuntimeError(
                    f"tensor sync rollout update failed on bucket {bucket_index}: {first_error}"
                )

    def shutdown(self) -> None:
        self.engine.shutdown()

    def _build_named_tensor_buckets(
        self,
        *,
        bucket_bytes: int,
    ) -> list[list[tuple[str, torch.Tensor]]]:
        if bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be > 0")

        state_dict = get_model_state_dict(
            self.engine.main_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.spec.rank != 0:
            return []

        publish_dtype = get_torch_dtype(
            self.spec.cfg.train.config.mixed_precision.param_dtype
        )
        parameter_names = {
            name for name, _parameter in self.engine.main_model.named_parameters()
        }
        named_tensors = [
            (
                name,
                tensor.detach().to(dtype=publish_dtype).cpu().contiguous(),
            )
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor) and name in parameter_names
        ]
        if not named_tensors:
            raise RuntimeError("main_model state_dict did not contain any tensors")

        buckets: list[list[tuple[str, torch.Tensor]]] = []
        bucket: list[tuple[str, torch.Tensor]] = []
        current_bucket_bytes = 0
        for name, tensor in named_tensors:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and current_bucket_bytes + tensor_bytes > bucket_bytes:
                buckets.append(bucket)
                bucket = []
                current_bucket_bytes = 0
            bucket.append((name, tensor))
            current_bucket_bytes += tensor_bytes
        if bucket:
            buckets.append(bucket)
        return buckets

    def _build_rank_local_serialized_tensor_buckets(
        self,
        *,
        bucket_bytes: int,
    ) -> list[str]:
        self._patch_torch_reductions_for_sglang()
        publish_dtype = get_torch_dtype(
            self.spec.cfg.train.config.mixed_precision.param_dtype
        )
        parameter_names = {
            name for name, _parameter in self.engine.main_model.named_parameters()
        }

        serialized_buckets: list[str] = []
        bucket: list[tuple[str, torch.Tensor | Any]] = []
        current_bucket_bytes = 0
        for name, tensor in self.engine.main_model.state_dict().items():
            if not isinstance(tensor, torch.Tensor) or name not in parameter_names:
                continue
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and current_bucket_bytes + tensor_bytes > bucket_bytes:
                serialized_buckets.append(
                    self._serialize_rank_local_tensor_bucket(
                        bucket=bucket,
                        publish_dtype=publish_dtype,
                    )
                )
                bucket = []
                current_bucket_bytes = 0
            bucket.append((name, self._prepare_rank_local_publish_tensor(tensor)))
            current_bucket_bytes += tensor_bytes

        if bucket:
            serialized_buckets.append(
                self._serialize_rank_local_tensor_bucket(
                    bucket=bucket,
                    publish_dtype=publish_dtype,
                )
            )
        if not serialized_buckets:
            raise RuntimeError("main_model state_dict did not contain any tensors")
        return serialized_buckets

    def _prepare_rank_local_publish_tensor(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor | Any:
        if isinstance(tensor, DTensor):
            return tensor.redistribute(
                placements=[Replicate()] * tensor.device_mesh.ndim,
                async_op=True,
            ).to_local()
        if tensor.device != self.engine.device:
            tensor = tensor.to(self.engine.device)
        return tensor

    def _serialize_rank_local_tensor_bucket(
        self,
        *,
        bucket: list[tuple[str, torch.Tensor | Any]],
        publish_dtype: torch.dtype,
    ) -> str:
        ready_bucket = [
            (
                name,
                (tensor.wait() if hasattr(tensor, "wait") else tensor)
                .detach()
                .to(dtype=publish_dtype)
                .contiguous(),
            )
            for name, tensor in bucket
        ]
        return serialize_tensor_bucket(ready_bucket)

    @staticmethod
    def _patch_torch_reductions_for_sglang() -> None:
        if not torch.cuda.is_available():
            return
        try:
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        except ImportError:
            from sglang.srt.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
