# Copyright 2026 openforge

import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import ray
import torch
from huggingface_hub import save_torch_state_dict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer

from openforge.train.fsdp2.base import FSDP2Engine
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.utils.networking import get_free_port
from openforge.utils.packed import serialize_tensor_bucket
from openforge.utils.ray import get_current_ray_node_ip_address
from openforge.utils.torch import get_torch_dtype
from openforge.utils.train_batching import pack_microbatch_group, pack_minibatch

__all__ = ["TrainWorker", "RayTrainWorker"]


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
        mini_batch: dict[str, torch.Tensor],
        *,
        global_step: int | None = None,
    ) -> TrainStepResult:
        microbatches = pack_minibatch(
            mini_batch,
            micro_batch_size=self.spec.cfg.train.micro_batch_size,
        )
        self._prepare_log_probs(microbatches)
        return self._step_microbatches(
            microbatches,
            global_step=global_step,
        )

    def step_update(
        self,
        microbatch_groups: Sequence[Sequence[dict[str, torch.Tensor]]],
        *,
        global_step: int | None = None,
    ) -> list[TrainStepResult]:
        packed_groups = []
        for microbatches in microbatch_groups:
            packed_batches = pack_microbatch_group(list(microbatches))
            self._prepare_log_probs(packed_batches)
            packed_groups.append(packed_batches)
        results: list[TrainStepResult] = []
        for _ in range(self.spec.cfg.train.ppo_epochs):
            for packed_batches in packed_groups:
                results.append(
                    self._step_microbatches(
                        packed_batches,
                        global_step=global_step,
                    )
                )
        return results

    def _step_microbatches(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        *,
        global_step: int | None,
    ) -> TrainStepResult:
        loss_weights = [
            float(batch["loss_mask"].sum().clamp_min(1.0)) for batch in microbatches
        ]
        total_loss_weight = max(sum(loss_weights), 1.0)
        self.engine.zero_grad()
        last_index = len(microbatches) - 1
        forward_metrics: dict[str, float] = {}
        for index, (batch, loss_weight) in enumerate(
            zip(microbatches, loss_weights, strict=True)
        ):
            context = self.engine.no_sync() if index < last_index else nullcontext()
            with context:
                outputs = self.engine.forward(batch)
                outputs["loss"] = outputs["loss"] * (loss_weight / total_loss_weight)
                self.engine.backward(outputs)
            for key, value in outputs.items():
                if key == "loss" or value is None or value.ndim != 0:
                    continue
                forward_metrics[key] = forward_metrics.get(key, 0.0) + float(value) * (
                    loss_weight / total_loss_weight
                )

        metrics = self.engine.step_optimizer()
        return TrainStepResult(
            rank=self.spec.rank,
            global_step=global_step,
            metrics={
                **forward_metrics,
                **metrics,
                "global_step": -1.0 if global_step is None else float(global_step),
            },
        )

    def _prepare_log_probs(
        self,
        microbatches: list[dict[str, torch.Tensor]],
    ) -> None:
        if self.engine.ref_model is not None:
            self.engine.offload_train_state()
            self.engine.onload_ref_state()
            try:
                for batch in microbatches:
                    batch["ref_log_probs"] = self.engine.compute_token_log_probs(
                        batch,
                        model="ref",
                    )
            finally:
                self.engine.offload_ref_state()
                self.engine.onload_train_state()

        for batch in microbatches:
            batch["old_log_probs"] = self.engine.compute_token_log_probs(
                batch,
                model="main",
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

    @staticmethod
    def get_free_port(
        *,
        start: int = 10000,
        block_size: int = 1,
    ) -> int:
        return get_free_port(start=start, block_size=block_size)

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


RayTrainWorker = ray.remote(TrainWorker)
