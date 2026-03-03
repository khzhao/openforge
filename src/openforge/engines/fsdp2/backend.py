# Copyright 2026 openforge

import math
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from transformers import AutoModelForCausalLM

from openforge.configs import FSDP2Config, OpenForgeConfig
from openforge.engines.abcs import TrainBackend

from .base import apply_fsdp2, create_device_mesh, get_torch_dtype


class FSDP2Backend(TrainBackend):
    """Lean FSDP2 backend.

    Expected packed batch contract:
    - tokens: LongTensor [N]
    - cu_seqlens: IntTensor [B+1]
    - loss_mask: Float/Bool Tensor [N]
    - rewards: Float Tensor [N]
    - position_ids: LongTensor [N] (optional)
    """

    def __init__(self) -> None:
        self.cfg: OpenForgeConfig | None = None
        self.rank: int | None = None
        self.world_size: int | None = None
        self.master_addr: str | None = None
        self.master_port: int | None = None

        self.device: torch.device | None = None

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.grad_scaler: torch.amp.GradScaler | None = None

        self._offloaded: bool | None = None

    def initialize(
        self,
        cfg: OpenForgeConfig,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self._validate_init_cfg(cfg)
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialize_process_group()
        self._initialize_device()
        self._initialize_train_components()
        self._offloaded = False

    def zero_grad(self) -> None:
        self._optimizer().zero_grad(set_to_none=True)

    def forward(self, batch: TensorDict) -> TensorDict:
        model = self._model()
        device = self._device()
        amp_cfg = self._backend_cfg.amp
        amp_enabled = bool(amp_cfg.enabled and device.type == "cuda")
        amp_dtype = get_torch_dtype(amp_cfg.precision)

        tokens = batch.get("tokens").to(device)
        cu_seqlens = batch.get("cu_seqlens").to(device)
        loss_mask = batch.get("loss_mask").to(device).float()
        rewards = batch.get("rewards").to(device).float()
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(device)

        numerator = torch.zeros((), device=device)
        denominator = torch.zeros((), device=device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            for i in range(cu_seqlens.numel() - 1):
                start = int(cu_seqlens[i])
                end = int(cu_seqlens[i + 1])

                input_ids = tokens[start:end].unsqueeze(0)
                kwargs = {"input_ids": input_ids}
                if position_ids is not None:
                    kwargs["position_ids"] = position_ids[start:end].unsqueeze(0)

                logits = model(**kwargs).logits[0, :-1, :]
                target = tokens[start + 1 : end]
                sample_mask = loss_mask[start + 1 : end]
                sample_rewards = rewards[start + 1 : end]

                log_probs = torch.log_softmax(logits, dim=-1)
                chosen_log_probs = torch.gather(
                    log_probs, dim=-1, index=target.unsqueeze(-1)
                ).squeeze(-1)

                numerator += -(sample_rewards * chosen_log_probs * sample_mask).sum()
                denominator += sample_mask.sum()

            loss = numerator / denominator.clamp_min(1.0)

        return TensorDict(
            {
                "loss": loss,
            },
            batch_size=[],
        )

    def backward(self, forward_out: TensorDict) -> None:
        scaler = self._grad_scaler()
        loss = forward_out.get("loss") / self._gradient_accumulation_steps()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def no_sync(self) -> AbstractContextManager[None]:
        return nullcontext()

    def step_optimizer(self, *, global_step: int | None = None) -> dict[str, float]:
        model = self._model()
        optimizer = self._optimizer()
        scheduler = self._scheduler()
        scaler = self._grad_scaler()
        fsdp_cfg = self._backend_cfg

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            fsdp_cfg.optim.max_grad_norm,
        )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        metrics = {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "global_step": -1.0 if global_step is None else float(global_step),
            "gradient_accumulation_steps": float(self._gradient_accumulation_steps()),
        }
        if scaler.is_enabled():
            metrics["grad_scale"] = float(scaler.get_scale())
        return metrics

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> str:
        cfg = self._cfg()
        model = self._model()
        optimizer = self._optimizer()
        scheduler = self._scheduler()
        scaler = self._grad_scaler()
        rank = self._rank()

        ckpt_dir = Path(cfg.train.checkpoints_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{step:08d}.rank_{rank:05d}.pt"

        payload: dict[str, object] = {
            "step": step,
            "policy_version": policy_version,
            "model_state": model.state_dict(),
        }
        if save_optimizer:
            payload["optimizer_state"] = optimizer.state_dict()
            payload["scheduler_state"] = scheduler.state_dict()
            if scaler.is_enabled():
                payload["grad_scaler_state"] = scaler.state_dict()

        torch.save(payload, path)
        return str(path)

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> tuple[int, int] | None:
        cfg = self._cfg()
        model = self._model()
        optimizer = self._optimizer()
        scheduler = self._scheduler()
        scaler = self._grad_scaler()

        ckpt_dir = Path(cfg.train.checkpoints_dir)
        path = self._resolve_checkpoint_path(ckpt_dir, latest=latest, step=step)
        if path is None:
            return None

        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        if load_optimizer and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if load_optimizer and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if load_optimizer and scaler.is_enabled() and "grad_scaler_state" in payload:
            scaler.load_state_dict(payload["grad_scaler_state"])

        loaded_step = int(payload["step"])
        loaded_policy_version = int(payload.get("policy_version", loaded_step))
        return loaded_step, loaded_policy_version

    def offload(self) -> None:
        model = self._model()
        optimizer = self._optimizer()
        fsdp_cfg = self._backend_cfg
        offloaded = self._offload_state()

        if offloaded:
            return
        if fsdp_cfg.offload.mode == "cpu":
            self._offloaded = True
            return

        model.to("cpu")
        self._move_optimizer_state(optimizer, "cpu")
        self.clear_memory()
        self._offloaded = True

    def onload(self) -> None:
        model = self._model()
        optimizer = self._optimizer()
        device = self._device()
        fsdp_cfg = self._backend_cfg
        offloaded = self._offload_state()

        if not offloaded:
            return
        if fsdp_cfg.offload.mode == "cpu":
            self._offloaded = False
            return

        model.to(device)
        self._move_optimizer_state(optimizer, device)
        self._offloaded = False

    def clear_memory(self) -> None:
        if self._device().type == "cuda":
            torch.cuda.empty_cache()

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _validate_init_cfg(self, cfg: OpenForgeConfig) -> None:
        if cfg.train.backend != "fsdp2":
            raise ValueError(
                f"FSDP2Backend requires backend=fsdp2, got {cfg.train.backend}"
            )
        if cfg.train.use_critic:
            raise NotImplementedError(
                "FSDP2Backend currently supports actor-only training."
            )
        assert isinstance(cfg.train.backend_cfg, FSDP2Config)

    def _initialize_process_group(self) -> None:
        if dist.is_initialized():
            return

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=self._rank(),
            world_size=self._world_size(),
            init_method=f"tcp://{self._master_addr()}:{self._master_port()}",
            timeout=timedelta(seconds=30),
        )

    def _initialize_device(self) -> None:
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            assert torch.cuda.device_count() == 1, "Expected only 1 GPU per worker"
            torch.cuda.set_device(0)
            self.device = torch.device("cuda", 0)

    def _initialize_train_components(self) -> None:
        cfg = self._cfg()
        fsdp_cfg = self._backend_cfg
        device = self._device()

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name_or_path,
            trust_remote_code=True,
        )
        if fsdp_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.train()

        mesh = create_device_mesh(
            dp_size=cfg.train.data_parallel_size,
            world_size=self._world_size(),
            device_type=device.type,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.param_dtype),
            reduce_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.reduce_dtype),
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=fsdp_cfg.offload.pin_memory)
            if fsdp_cfg.offload.mode == "cpu"
            else OffloadPolicy()
        )

        self.model = apply_fsdp2(
            model=model,
            device_mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
        )
        self.model.to(device)

        optim_cfg = fsdp_cfg.optim
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optim_cfg.lr,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_eps,
            weight_decay=optim_cfg.weight_decay,
        )
        self.lr_scheduler = self._build_lr_scheduler()

        amp_cfg = fsdp_cfg.amp
        amp_enabled = bool(amp_cfg.enabled and device.type == "cuda")
        use_grad_scaler = (
            amp_enabled and amp_cfg.use_grad_scaler and amp_cfg.precision == "float16"
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_grad_scaler,
        )

    def _resolve_checkpoint_path(
        self,
        checkpoints_dir: Path,
        *,
        latest: bool,
        step: int | None,
    ) -> Path | None:
        rank = self._rank()
        if not checkpoints_dir.exists():
            return None

        if step is not None:
            candidate = checkpoints_dir / f"step_{step:08d}.rank_{rank:05d}.pt"
            return candidate if candidate.exists() else None

        if not latest:
            return None

        candidates = sorted(checkpoints_dir.glob(f"step_*.rank_{rank:05d}.pt"))
        if not candidates:
            return None
        return candidates[-1]

    @staticmethod
    def _move_optimizer_state(
        optimizer: torch.optim.Optimizer, device: torch.device | str
    ) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        optimizer = self._optimizer()
        fsdp_cfg = self._backend_cfg

        scheduler_cfg = fsdp_cfg.scheduler
        warmup_steps = scheduler_cfg.warmup_steps
        decay_steps = max(1, warmup_steps)

        min_lr_ratio = (
            scheduler_cfg.min_lr_rate
            if scheduler_cfg.min_lr_rate is not None
            else scheduler_cfg.min_lr / fsdp_cfg.optim.lr
        )
        min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

        def lr_lambda(step_idx: int) -> float:
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)

            if scheduler_cfg.type == "constant":
                return 1.0

            progress = min(1.0, float(step_idx - warmup_steps) / float(decay_steps))

            if scheduler_cfg.type == "linear":
                decay = 1.0 - progress
                return min_lr_ratio + (1.0 - min_lr_ratio) * decay

            cosine = 0.5 * (
                1.0 + math.cos(math.pi * 2.0 * scheduler_cfg.num_cycles * progress)
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _gradient_accumulation_steps(self) -> int:
        return self._cfg().train.gradient_accumulation_steps

    def _cfg(self) -> OpenForgeConfig:
        assert self.cfg is not None
        return self.cfg

    def _rank(self) -> int:
        assert self.rank is not None
        return self.rank

    def _world_size(self) -> int:
        assert self.world_size is not None
        return self.world_size

    def _master_addr(self) -> str:
        assert self.master_addr is not None
        return self.master_addr

    def _master_port(self) -> int:
        assert self.master_port is not None
        return self.master_port

    def _device(self) -> torch.device:
        assert self.device is not None
        return self.device

    def _model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model

    def _optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None
        return self.optimizer

    def _scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        assert self.lr_scheduler is not None
        return self.lr_scheduler

    def _grad_scaler(self) -> torch.amp.GradScaler:
        assert self.grad_scaler is not None
        return self.grad_scaler

    def _offload_state(self) -> bool:
        assert self._offloaded is not None
        return self._offloaded

    @property
    def _backend_cfg(self) -> FSDP2Config:
        cfg = self._cfg()
        assert isinstance(cfg.train.backend_cfg, FSDP2Config)
        return cfg.train.backend_cfg
