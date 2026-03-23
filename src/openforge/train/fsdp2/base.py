# Copyright 2026 openforge

from contextlib import contextmanager
from datetime import timedelta
from typing import Iterator, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from transformers import AutoModelForCausalLM

from openforge.train.backend import TrainBackend
from openforge.train.types import TrainWorkerSpec
from openforge.utils.torch import get_torch_dtype

from .lr_scheduler import get_lr_scheduler
from .memory import offload_optimizer, offload_params, onload_optimizer, onload_params
from .utils import apply_fsdp2, create_device_mesh


def _selected_token_log_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return log-probs for the selected next-token targets only."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be rank-1, got shape {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "logits and targets must align on sequence length: "
            f"{logits.shape[0]} != {targets.shape[0]}"
        )
    return -F.cross_entropy(logits, targets, reduction="none")


class FSDP2Engine(TrainBackend):
    """FSDP2Engine for training. Essentially a wrapper around PyTorch FSDP2."""

    def __init__(self, spec: TrainWorkerSpec | None = None) -> None:
        if spec is not None:
            self.initialize(spec)

    def initialize(self, spec: TrainWorkerSpec) -> None:
        from openforge.runtime import create_algorithm

        self.spec = spec
        self.cfg = spec.cfg
        self.rank = spec.rank
        self.world_size = spec.world_size
        self.master_addr = spec.master_addr
        self.master_port = spec.master_port
        self.algorithm = create_algorithm(spec.cfg)

        # 1. Initialize the device and device mesh
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            assert torch.cuda.device_count() == 1, "Expected only 1 GPU per worker"
            torch.cuda.set_device(0)
            self.device = torch.device("cuda", 0)

        # 2. Initialize the process group
        if not dist.is_initialized():
            device_id = self.device if self.device.type == "cuda" else None
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=self.rank,
                world_size=self.world_size,
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                timeout=timedelta(seconds=30),
                device_id=device_id,
            )

        assert dist.get_rank() == self.rank, (
            f"Rank mismatch: {dist.get_rank()} != {self.rank}"
        )
        assert dist.get_world_size() == self.world_size, (
            f"World size mismatch: {dist.get_world_size()} != {self.world_size}"
        )
        self.device_mesh = create_device_mesh(
            dp_size=self.cfg.train.parallel.data_parallel_size,
            fsdp_size=self.cfg.train.parallel.fsdp_parallel_size,
            world_size=self.world_size,
            device_type=self.device.type,
        )

        # 3. Create or initialize the model + optimizer + scheduler
        model_name_or_path = self.cfg.model.model_name_or_path
        ref_model_name_or_path = self.cfg.model.reference_model_name_or_path
        attn_implementation = self.cfg.model.attn_implementation

        self.main_model = self._create_model(
            model_name_or_path, attn_implementation, is_eval_only=False
        )
        self.ref_model = self._create_model(
            ref_model_name_or_path, attn_implementation, is_eval_only=True
        )
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.grad_scaler = self._create_grad_scaler()

        # 4. Finalize components
        self._finalize()

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

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        tokens = batch["tokens"].to(self.device).long()
        loss_mask = batch["loss_mask"].to(self.device).float()
        advantages = batch["advantages"].to(self.device).float()
        old_log_probs = batch["old_log_probs"].to(self.device).float()
        rollout_log_probs = batch["rollout_log_probs"].to(self.device).float()
        position_ids = batch["position_ids"].to(self.device).long()

        # 1. Compute log probabilities for the sampled actions
        targets = tokens[1:]
        if self.cfg.algo.entropy_coef > 0.0:
            curr_full_log_probs = self._compute_log_probs(
                model=self.main_model,
                tokens=tokens,
                position_ids=position_ids,
            )
            entropy = -(curr_full_log_probs.exp() * curr_full_log_probs).sum(dim=-1)
            curr_log_probs = curr_full_log_probs.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
        else:
            entropy = None
            curr_log_probs = self._compute_token_log_probs(
                model=self.main_model,
                tokens=tokens,
                position_ids=position_ids,
            )

        ref_log_probs = None
        if self.ref_model is not None:
            ref_log_probs = self._compute_token_log_probs(
                model=self.ref_model,
                tokens=tokens,
                position_ids=position_ids,
            )

        # 2. Delegate the algorithm-specific loss math
        outputs: dict[str, torch.Tensor | None] = {
            "curr_log_probs": curr_log_probs.detach(),
            "ref_log_probs": None if ref_log_probs is None else ref_log_probs.detach(),
        }
        outputs.update(
            self.algorithm.compute_loss(
                curr_log_probs=curr_log_probs,
                old_log_probs=old_log_probs,
                rollout_log_probs=rollout_log_probs,
                advantages=advantages[1:],
                loss_mask=loss_mask,
                entropy=entropy,
                ref_log_probs=ref_log_probs,
            )
        )
        return outputs

    @torch.no_grad()
    def compute_old_log_probs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"].to(self.device).long()
        position_ids = batch["position_ids"].to(self.device).long()
        token_log_probs = self._compute_token_log_probs(
            model=self.main_model,
            tokens=tokens,
            position_ids=position_ids,
        )
        return token_log_probs.float().cpu()

    def backward(self, forward_out: dict[str, torch.Tensor]) -> None:
        loss = forward_out["loss"] / self.cfg.train.gradient_accumulation_steps
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def step_optimizer(self) -> dict[str, float]:
        optim_cfg = self.cfg.train.config.optim

        # 1. Unscale the gradients if necessary
        if self.use_grad_scaler:
            self.grad_scaler.unscale_(self.optimizer)

        # 2. Clip the gradients
        nn.utils.clip_grad_norm_(
            self.main_model.parameters(),
            optim_cfg.max_grad_norm,
        )

        # 3. Step the optimizer and scheduler
        if self.use_grad_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()

        # 4. Return the metrics
        metrics = {
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": -1.0,
            "gradient_accumulation_steps": float(
                self.cfg.train.gradient_accumulation_steps
            ),
        }
        if self.use_grad_scaler:
            metrics["grad_scale"] = float(self.grad_scaler.get_scale())
        return metrics

    def sleep(self) -> None:
        offload_params(self.main_model, offload_grad=True)
        if self.ref_model is not None:
            offload_params(self.ref_model, offload_grad=False)
        offload_optimizer(self.optimizer)
        self.sleeping = True

    def wakeup(self) -> None:
        onload_params(self.main_model, self.device, onload_grad=True)
        if self.ref_model is not None:
            onload_params(self.ref_model, self.device, onload_grad=False)
        onload_optimizer(self.optimizer, self.device)
        self.sleeping = False

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _compute_log_probs(
        self,
        *,
        model: FSDPModule,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._compute_logits(
            model=model,
            tokens=tokens,
            position_ids=position_ids,
        )
        return F.log_softmax(logits.float(), dim=-1)

    def _compute_token_log_probs(
        self,
        *,
        model: FSDPModule,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._compute_logits(
            model=model,
            tokens=tokens,
            position_ids=position_ids,
        )
        targets = tokens[1:]
        return _selected_token_log_probs(logits, targets)

    def _compute_logits(
        self,
        *,
        model: FSDPModule,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=get_torch_dtype(self.cfg.train.config.amp.precision),
            enabled=self.amp_enabled,
        ):
            return model(
                input_ids=tokens.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                use_cache=False,
            ).logits.squeeze(0)[:-1, :]

    def _create_model(
        self,
        model_name_or_path: str | None,
        attn_implementation: Literal["flash_attention_2", "eager", "sdpa"],
        is_eval_only: bool = False,
    ) -> FSDPModule | None:
        if model_name_or_path is None:
            return None

        # 1. Get Offload / Mixed Precision Policy
        fsdp_cfg = self.cfg.train.config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.param_dtype),
            reduce_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.reduce_dtype),
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=fsdp_cfg.offload.pin_memory)
            if fsdp_cfg.offload.mode == "cpu"
            else OffloadPolicy()
        )

        # 2. Create the model with the right parameters and set train/eval mode.
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if not is_eval_only and fsdp_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

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
            model.train()
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        optim_cfg = self.cfg.train.config.optim
        return optim.AdamW(
            self.main_model.parameters(),
            lr=optim_cfg.lr,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_eps,
            weight_decay=optim_cfg.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        fsdp_cfg = self.cfg.train.config
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
        amp_cfg = self.cfg.train.config.amp
        self.amp_enabled = bool(amp_cfg.enabled and self.device.type == "cuda")
        self.use_grad_scaler = (
            self.amp_enabled
            and amp_cfg.use_grad_scaler
            and amp_cfg.precision == "float16"
        )
        return torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_grad_scaler,
        )

    def _finalize(self):
        onload_params(self.main_model, self.device, onload_grad=True)
        if self.ref_model is not None:
            onload_params(self.ref_model, self.device, onload_grad=False)
        onload_optimizer(self.optimizer, self.device)
        self.sleeping = False
