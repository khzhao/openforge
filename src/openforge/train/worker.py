# Copyright 2026 openforge

import os
import uuid
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from huggingface_hub import save_torch_state_dict
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers import AutoTokenizer

from openforge.train.fsdp2.base import FSDP2Engine
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState
from openforge.utils.distributed import init_custom_process_group
from openforge.utils.networking import get_free_port
from openforge.utils.torch import get_torch_dtype_name

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
        self.distributed_update_buckets: dict[
            str, list[list[tuple[str, torch.Tensor]]]
        ] = {}
        self.distributed_update_groups: dict[str, dist.ProcessGroup] = {}
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
        self.state = replace(self.state, sleeping=True)

    def wakeup(self) -> None:
        self.engine.wakeup()
        self.state = replace(self.state, sleeping=False)

    def status(self) -> TrainWorkerState:
        return self.state

    def build_tensor_buckets(self, *, bucket_bytes: int) -> list[str] | None:
        named_buckets = self._build_named_tensor_buckets(bucket_bytes=bucket_bytes)
        if self.spec.rank != 0:
            return None
        return [self._serialize_tensor_bucket(bucket) for bucket in named_buckets]

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

    def prepare_distributed_update(
        self,
        *,
        bucket_bytes: int,
        world_size: int,
    ) -> dict[str, object] | None:
        named_buckets = self._build_named_tensor_buckets(bucket_bytes=bucket_bytes)
        if self.spec.rank != 0:
            return None

        group_name = f"openforge-weight-update-{uuid.uuid4().hex[:10]}"
        master_port = get_free_port(start=max(self.spec.master_port + 100, 20000))
        self.distributed_update_buckets[group_name] = named_buckets
        return {
            "group_name": group_name,
            "master_addr": self.spec.master_addr,
            "master_port": master_port,
            "world_size": world_size,
            "bucket_metas": [
                self._build_bucket_meta(bucket) for bucket in named_buckets
            ],
        }

    def connect_distributed_update(self, *, plan: dict[str, object]) -> None:
        if self.spec.rank != 0:
            return
        if self.engine.device.type != "cuda":
            raise RuntimeError("distributed weight update requires CUDA")

        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        torch.cuda.set_device(self.engine.device)
        group_name = str(plan["group_name"])
        self.distributed_update_groups[group_name] = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{plan['master_addr']}:{plan['master_port']}",
            world_size=int(plan["world_size"]),
            rank=0,
            group_name=group_name,
        )

    def broadcast_distributed_bucket(
        self,
        *,
        plan: dict[str, object],
        bucket_index: int,
    ) -> None:
        if self.spec.rank != 0:
            return

        group_name = str(plan["group_name"])
        process_group = self.distributed_update_groups[group_name]
        for _name, tensor in self.distributed_update_buckets[group_name][bucket_index]:
            device_tensor = tensor.to(self.engine.device).contiguous()
            dist.broadcast(device_tensor, src=0, group=process_group)

    def finish_distributed_update(self, *, plan: dict[str, object]) -> None:
        if self.spec.rank != 0:
            return

        group_name = str(plan["group_name"])
        process_group = self.distributed_update_groups.pop(group_name, None)
        if process_group is not None:
            dist.destroy_process_group(process_group)
        self.distributed_update_buckets.pop(group_name, None)

    def shutdown(self) -> None:
        for process_group in self.distributed_update_groups.values():
            dist.destroy_process_group(process_group)
        self.distributed_update_groups.clear()
        self.distributed_update_buckets.clear()
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

        named_tensors = [
            (name, tensor.detach().cpu().contiguous())
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
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

    def _build_bucket_meta(
        self,
        bucket: list[tuple[str, torch.Tensor]],
    ) -> dict[str, list[object]]:
        return {
            "names": [name for name, _tensor in bucket],
            "dtypes": [get_torch_dtype_name(tensor.dtype) for _name, tensor in bucket],
            "shapes": [list(tensor.shape) for _name, tensor in bucket],
        }

    def _serialize_tensor_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> str:
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        flattened_bucket = FlattenedTensorBucket(named_tensors=bucket)
        payload = {
            "flattened_tensor": flattened_bucket.get_flattened_tensor(),
            "metadata": flattened_bucket.get_metadata(),
        }
        return MultiprocessingSerializer.serialize(payload, output_str=True)
