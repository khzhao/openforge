# Copyright 2026 openforge

from contextlib import contextmanager
from datetime import timedelta
from typing import Iterator

import torch
import torch.distributed as dist
import torch.optim as optim
from tensordict import TensorDict
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from transformers import AutoModelForCausalLM

from openforge.train.types import TrainWorkerSpec
from openforge.utils.torch import get_torch_dtype

from .lr_scheduler import get_lr_scheduler
from .memory import offload_optimizer, offload_params, onload_optimizer, onload_params
from .runtime import apply_fsdp2, create_device_mesh


class FSDP2Engine:
    """FSDP2Engine for training. Essentially a wrapper around PyTorch FSDP2."""

    def __init__(self, spec: TrainWorkerSpec) -> None:
        self.initialize(spec)

    def initialize(self, spec: TrainWorkerSpec) -> None:
        self.spec = spec
        self.cfg = spec.cfg
        self.rank = spec.rank
        self.world_size = spec.world_size
        self.master_addr = spec.master_addr
        self.master_port = spec.master_port

        # 1. Initialize the device and device mesh
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            assert torch.cuda.device_count() == 1, "Expected only 1 GPU per worker"
            torch.cuda.set_device(0)
            self.device = torch.device("cuda", 0)
        self.device_mesh = create_device_mesh(
            dp_size=self.cfg.train.parallelism.data_parallel_size,
            fsdp_size=self.cfg.train.parallelism.fsdp_parallel_size,
            world_size=self.world_size,
            device_type=self.device.type,
        )

        # 2. Initialize the process group
        if not dist.is_initialized():
            assert dist.get_rank() == self.rank, (
                f"Rank mismatch: {dist.get_rank()} != {self.rank}"
            )
            assert dist.get_world_size() == self.world_size, (
                f"World size mismatch: {dist.get_world_size()} != {self.world_size}"
            )
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=self.rank,
                world_size=self.world_size,
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                timeout=timedelta(seconds=30),
            )

        # 3. Create or initialize the model + optimizer + scheduler
        model_name_or_path = self.cfg.model.model_name_or_path
        ref_model_name_or_path = self.cfg.model.reference_model_name_or_path

        self.main_model = self._create_model(model_name_or_path, is_eval_only=False)
        self.ref_model = self._create_model(ref_model_name_or_path, is_eval_only=True)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.grad_scaler = self._create_grad_scaler()

        # 4. Miscellaneous components
        self.sleeping = False

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        model = self.main_model
        model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)

    def forward(self, batch: TensorDict) -> None:
        pass

    def backward(self) -> None:
        pass

    def step_optimizer(self) -> None:
        pass

    def sleep(self) -> None:
        offload_params(self.main_model)
        offload_optimizer(self.optimizer)
        self.sleeping = True

    def wakeup(self) -> None:
        onload_params(self.main_model, self.device)
        onload_optimizer(self.optimizer, self.device)
        self.sleeping = False

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _create_model(
        self,
        model_name_or_path: str | None,
        is_eval_only: bool = False,
    ) -> FSDPModule:
        if model_name_or_path is None:
            return None

        # 1. Get Offload / Mixed Precision Policy
        fsdp_cfg = self.cfg.train.backend_config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.param_dtype),
            reduce_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.reduce_dtype),
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=fsdp_cfg.offload.pin_memory)
            if fsdp_cfg.offload.mode == "cpu"
            else OffloadPolicy()
        )

        # 2. Create the model with the right parameters and set train/eval mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = apply_fsdp2(
            model=model,
            device_mesh=self.device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            shard_modules=fsdp_cfg.shard_modules,
        )

        if is_eval_only:
            model.requires_grad_(False)
            model.eval()
        else:
            if fsdp_cfg.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            model.train()
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        optim_cfg = self.cfg.train.backend_config.optim
        return optim.AdamW(
            self.main_model.parameters(),
            lr=optim_cfg.lr,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_eps,
            weight_decay=optim_cfg.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        fsdp_cfg = self.cfg.train.backend_config
        return get_lr_scheduler(
            scheduler_type=fsdp_cfg.scheduler.type,
            optimizer=self.optimizer,
            num_warmup_steps=fsdp_cfg.scheduler.warmup_steps,
            num_training_steps=max(1, fsdp_cfg.scheduler.warmup_steps * 2),
            num_cycles=fsdp_cfg.scheduler.num_cycles,
            min_lr=fsdp_cfg.scheduler.min_lr,
            min_lr_rate=fsdp_cfg.scheduler.min_lr_rate,
        )

    def _create_grad_scaler(self) -> torch.amp.GradScaler:
        amp_cfg = self.cfg.train.backend_config.amp
        amp_enabled = bool(amp_cfg.enabled and self.device.type == "cuda")
        use_grad_scaler = (
            amp_enabled and amp_cfg.use_grad_scaler and amp_cfg.precision == "float16"
        )
        return torch.amp.GradScaler(
            self.device.type,
            enabled=use_grad_scaler,
        )
