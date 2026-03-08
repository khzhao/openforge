# Copyright 2026 openforge

from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from transformers import AutoConfig, AutoModelForCausalLM

from openforge.configs.models import OpenForgeConfig
from openforge.configs.train import FSDP2Config
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.train.backend import TrainBackend
from openforge.train.types import CheckpointInfo, TrainStepResult, TrainWorkerSpec

from .memory import offload_optimizer, offload_params, onload_optimizer, onload_params
from .publish import DistributedWeightUpdater, TensorWeightUpdater
from .runtime import apply_fsdp2, create_device_mesh, get_torch_dtype
from .scheduler import get_lr_scheduler


class FSDP2Backend(TrainBackend):
    """Lean FSDP2 backend.

    Expected packed batch contract:
    - tokens: LongTensor [N]
    - loss_mask: Float/Bool Tensor [N]
    - rewards: Float Tensor [N]
    - position_ids: LongTensor [N]
    """

    def __init__(self) -> None:
        self.cfg: OpenForgeConfig | None = None
        self.rank: int | None = None
        self.world_size: int | None = None
        self.master_addr: str | None = None
        self.master_port: int | None = None

        self.device: torch.device | None = None

        self.model: torch.nn.Module | None = None
        self.ref_model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.grad_scaler: torch.amp.GradScaler | None = None

        self._sleeping: bool | None = None

    def initialize(
        self,
        spec: TrainWorkerSpec,
    ) -> None:
        cfg = spec.cfg
        rank = spec.rank
        world_size = spec.world_size
        master_addr = spec.master_addr
        master_port = spec.master_port
        self._validate_init_cfg(cfg)
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialize_device()
        self._initialize_process_group()
        self._initialize_train_components()
        self._sleeping = False

    def zero_grad(self) -> None:
        self._optimizer().zero_grad(set_to_none=True)

    def forward(self, batch: TensorDict) -> TensorDict:
        model = self._model()
        ref_model = self._maybe_ref_model()
        device = self._device()
        tokens = batch.get("tokens").to(device)
        loss_mask = batch.get("loss_mask").to(device).float()
        rewards = batch.get("rewards").to(device).float()
        position_ids = batch.get("position_ids").to(device)
        transition_mask = self._transition_mask(
            loss_mask=loss_mask,
            position_ids=position_ids,
        )

        curr_log_probs = self._compute_log_probs(
            model=model,
            tokens=tokens,
            position_ids=position_ids,
        )
        loss = self._masked_mean(
            values=-(rewards[1:] * curr_log_probs),
            mask=transition_mask,
        )

        ref_log_probs = None
        if ref_model is not None and self._algo_cfg.kl_coef > 0.0:
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(
                    model=ref_model,
                    tokens=tokens,
                    position_ids=position_ids,
                )
            loss = loss + (
                self._algo_cfg.kl_coef
                * self._masked_mean(
                    values=curr_log_probs - ref_log_probs,
                    mask=transition_mask,
                )
            )

        outputs = {
            "loss": loss,
            "curr_log_probs": curr_log_probs.detach(),
        }
        if ref_log_probs is not None:
            outputs["ref_log_probs"] = ref_log_probs.detach()
        return TensorDict(outputs, batch_size=[])

    def backward(self, forward_out: TensorDict) -> None:
        scaler = self._grad_scaler()
        loss = forward_out.get("loss") / self._gradient_accumulation_steps()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def no_sync(self) -> AbstractContextManager[None]:
        return nullcontext()

    def step_optimizer(self, *, global_step: int | None = None) -> TrainStepResult:
        model = self._model()
        optimizer = self._optimizer()
        scheduler = self._scheduler()
        scaler = self._grad_scaler()
        fsdp_cfg = self._backend_cfg
        scaler_enabled = scaler.is_enabled()

        if scaler_enabled:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            fsdp_cfg.optim.max_grad_norm,
        )

        if scaler_enabled:
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
        if scaler_enabled:
            metrics["grad_scale"] = float(scaler.get_scale())
        return TrainStepResult(
            rank=self._rank(),
            global_step=global_step,
            metrics=metrics,
        )

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> CheckpointInfo:
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
        return CheckpointInfo(
            step=step,
            policy_version=policy_version,
            path=str(path),
        )

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> CheckpointInfo | None:
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
        return CheckpointInfo(
            step=loaded_step,
            policy_version=loaded_policy_version,
            path=str(path),
        )

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef | None:
        if self._rank() != 0:
            return None
        checkpoint = self.save_checkpoint(
            step=step,
            policy_version=policy_version,
            save_optimizer=False,
        )
        return PolicyArtifactRef(
            step=checkpoint.step,
            policy_version=checkpoint.policy_version,
            path=checkpoint.path,
        )

    def push_tensor_update(
        self,
        session: TensorUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        rollout_workers, rollout_engines = self._session_targets(session)
        updater = TensorWeightUpdater(
            self._cfg(),
            self._model(),
            rank=self._rank(),
            world_size=self._world_size(),
            device=self._device(),
            master_addr=self._master_addr(),
            session=session,
        )
        updater.connect_rollout_engines(
            rollout_workers=rollout_workers,
            rollout_engines=rollout_engines,
        )
        updater.update_weights(policy_version=policy_version)

    def push_distributed_update(
        self,
        session: DistributedUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        rollout_workers, rollout_engines = self._session_targets(session)
        updater = DistributedWeightUpdater(
            self._cfg(),
            self._model(),
            rank=self._rank(),
            world_size=self._world_size(),
            device=self._device(),
            master_addr=self._master_addr(),
            session=session,
        )
        updater.connect_rollout_engines(
            rollout_workers=rollout_workers,
            rollout_engines=rollout_engines,
        )
        updater.update_weights(policy_version=policy_version)

    def sleep(self) -> None:
        model = self._model()
        optimizer = self._optimizer()
        fsdp_cfg = self._backend_cfg
        sleeping = self._sleep_state()

        if sleeping:
            return
        if fsdp_cfg.offload.mode != "cpu":
            offload_params(model, offload_grad=True)
            offload_optimizer(optimizer)
        self.clear_memory()
        self._sleeping = True

    def wakeup(self) -> None:
        model = self._model()
        optimizer = self._optimizer()
        device = self._device()
        fsdp_cfg = self._backend_cfg
        sleeping = self._sleep_state()

        if not sleeping:
            return
        if fsdp_cfg.offload.mode != "cpu":
            onload_params(model, device, onload_grad=True)
            onload_optimizer(optimizer, device)
        self.clear_memory()
        self._sleeping = False

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
        assert isinstance(cfg.train.backend_config, FSDP2Config)

    def _initialize_process_group(self) -> None:
        if dist.is_initialized():
            return

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = self._device() if self._device().type == "cuda" else None
        dist.init_process_group(
            backend=backend,
            rank=self._rank(),
            world_size=self._world_size(),
            init_method=f"tcp://{self._master_addr()}:{self._master_port()}",
            timeout=timedelta(seconds=30),
            device_id=device_id,
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
        mesh, mp_policy, offload_policy = self._create_model_runtime(
            device=device,
            cpu_offload=fsdp_cfg.offload.mode == "cpu",
        )

        self.model = self._create_fsdp_model(
            model_name_or_path=cfg.model.model_name_or_path,
            device=device,
            device_mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            enable_gradient_checkpointing=fsdp_cfg.gradient_checkpointing,
            train_mode=True,
        )

        self.ref_model = None
        if cfg.model.reference_model_name_or_path is not None:
            ref_offload_policy: OffloadPolicy = (
                CPUOffloadPolicy(pin_memory=fsdp_cfg.offload.pin_memory)
                if device.type == "cuda"
                else OffloadPolicy()
            )
            self.ref_model = self._create_fsdp_model(
                model_name_or_path=cfg.model.reference_model_name_or_path,
                device=device,
                device_mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=ref_offload_policy,
                reshard_after_forward=fsdp_cfg.reshard_after_forward,
                enable_gradient_checkpointing=False,
                train_mode=False,
            )

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

    def _build_model_and_state_for_fsdp(
        self,
        model_name_or_path: str,
    ) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
        rank = self._rank()
        world_size = self._world_size()
        device = self._device()

        def maybe_barrier() -> None:
            if world_size <= 1 or not dist.is_initialized():
                return
            if device.type == "cuda":
                dist.barrier(device_ids=[device.index])
            else:
                dist.barrier()

        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            full_state = model.state_dict()
            maybe_barrier()
            return model, full_state

        maybe_barrier()

        model_cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        tie_word_embeddings = bool(getattr(model_cfg, "tie_word_embeddings", False))
        init_device = (
            torch.device("cpu") if tie_word_embeddings else torch.device("meta")
        )
        with init_device:
            model = AutoModelForCausalLM.from_config(
                model_cfg,
                trust_remote_code=True,
            )
        return model, {}

    def _create_fsdp_model(
        self,
        *,
        model_name_or_path: str,
        device: torch.device,
        device_mesh,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
        reshard_after_forward: bool,
        enable_gradient_checkpointing: bool,
        train_mode: bool,
    ) -> torch.nn.Module:
        model, full_state = self._build_model_and_state_for_fsdp(model_name_or_path)
        if self._model_uses_linear_attention(model):
            raise NotImplementedError(
                "FSDP2Backend packed forward currently requires full-attention-only models. "
                "Detected linear_attention layers in model config."
            )
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if train_mode:
            model.train()
        else:
            model.eval()

        model = apply_fsdp2(
            model=model,
            device_mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            shard_modules=self._backend_cfg.shard_modules,
        )
        model = self._load_full_state_dict_from_rank0(
            model,
            full_state,
            device=device,
            cpu_offload=isinstance(offload_policy, CPUOffloadPolicy),
        )
        if not train_mode:
            model.requires_grad_(False)
            model.eval()
        return model

    def _load_full_state_dict_from_rank0(
        self,
        model: torch.nn.Module,
        full_state: dict[str, torch.Tensor],
        *,
        device: torch.device,
        cpu_offload: bool,
    ) -> torch.nn.Module:
        target_device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda"
            else device
        )

        if self._rank() == 0:
            model = model.to(device=target_device, non_blocking=True)
        else:
            model = model.to_empty(device=target_device)

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload,
            broadcast_from_rank0=True,
        )
        set_model_state_dict(model, full_state, options=options)

        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if cpu_offload and target_device.type == "cuda":
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(target_device)
        return model

    def _create_model_runtime(
        self,
        *,
        device: torch.device,
        cpu_offload: bool,
    ) -> tuple[object, MixedPrecisionPolicy, OffloadPolicy]:
        cfg = self._cfg()
        fsdp_cfg = self._backend_cfg
        mesh = create_device_mesh(
            dp_size=cfg.train.parallelism.data_parallel_size,
            world_size=self._world_size(),
            device_type=device.type,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.param_dtype),
            reduce_dtype=get_torch_dtype(fsdp_cfg.mixed_precision.reduce_dtype),
        )
        offload_policy: OffloadPolicy = (
            CPUOffloadPolicy(pin_memory=fsdp_cfg.offload.pin_memory)
            if cpu_offload
            else OffloadPolicy()
        )
        return mesh, mp_policy, offload_policy

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

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        optimizer = self._optimizer()
        fsdp_cfg = self._backend_cfg

        scheduler_cfg = fsdp_cfg.scheduler
        return get_lr_scheduler(
            scheduler_type=scheduler_cfg.type,
            optimizer=optimizer,
            num_warmup_steps=scheduler_cfg.warmup_steps,
            num_training_steps=max(1, scheduler_cfg.warmup_steps * 2),
            num_cycles=scheduler_cfg.num_cycles,
            min_lr=scheduler_cfg.min_lr,
            min_lr_rate=scheduler_cfg.min_lr_rate,
        )

    def _session_targets(self, session) -> tuple[list[object], list[object]]:
        rollout_workers = session.transport_metadata.get("rollout_workers", [])
        rollout_engines = session.transport_metadata.get("rollout_engines", [])
        if len(rollout_workers) != len(rollout_engines):
            raise ValueError(
                "session transport metadata must provide rollout_workers and "
                "rollout_engines with matching lengths"
            )
        return list(rollout_workers), list(rollout_engines)

    def _gradient_accumulation_steps(self) -> int:
        return self._cfg().train.gradient_accumulation_steps

    def _compute_log_probs(
        self,
        *,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        device = self._device()
        amp_cfg = self._backend_cfg.amp
        amp_enabled = bool(amp_cfg.enabled and device.type == "cuda")
        amp_dtype = get_torch_dtype(amp_cfg.precision)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(
                input_ids=tokens.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
            ).logits[0, :-1, :]
        return self._chosen_log_probs(logits=logits, tokens=tokens)

    @staticmethod
    def _chosen_log_probs(
        *, logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        target = tokens[1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _transition_mask(
        *,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        transition_mask = loss_mask[1:].clone()

        # A reset to position 0 marks the first token of a packed sequence.
        transition_mask[position_ids[1:] == 0] = 0.0
        return transition_mask

    @staticmethod
    def _masked_mean(*, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _packed_weighted_nll_loss(
        *,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        loss_mask: torch.Tensor,
        rewards: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        transition_mask = FSDP2Backend._transition_mask(
            loss_mask=loss_mask,
            position_ids=position_ids,
        )
        chosen_log_probs = FSDP2Backend._chosen_log_probs(logits=logits, tokens=tokens)
        return FSDP2Backend._masked_mean(
            values=-(rewards[1:] * chosen_log_probs),
            mask=transition_mask,
        )

    @staticmethod
    def _model_uses_linear_attention(model: torch.nn.Module) -> bool:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return False
        text_cfg = getattr(cfg, "text_config", cfg)
        layer_types = getattr(text_cfg, "layer_types", None)
        if layer_types is None:
            return False
        return "linear_attention" in layer_types

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

    def _maybe_ref_model(self) -> torch.nn.Module | None:
        return self.ref_model

    def _optimizer(self) -> torch.optim.Optimizer:
        assert self.optimizer is not None
        return self.optimizer

    def _scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        assert self.lr_scheduler is not None
        return self.lr_scheduler

    def _grad_scaler(self) -> torch.amp.GradScaler:
        assert self.grad_scaler is not None
        return self.grad_scaler

    def _sleep_state(self) -> bool:
        assert self._sleeping is not None
        return self._sleeping

    @property
    def _algo_cfg(self):
        return self._cfg().algo

    @property
    def _backend_cfg(self) -> FSDP2Config:
        cfg = self._cfg()
        assert isinstance(cfg.train.backend_config, FSDP2Config)
        return cfg.train.backend_config
