# Copyright 2026 openforge

from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FSDPModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from openforge.train.types import CheckpointInfo
from openforge.utils.memory import clear_memory

from .base import FSDP2Engine

_CHECKPOINT_KEY = "fsdp_checkpoint"
_DCP_CHECKPOINT_DIRNAME = "dcp_checkpoint"
_METADATA_FILENAME = "trainer_state.pt"
_FULL_WEIGHTS_PATH = Path("model_checkpoint") / "full_weights.pt"


def _require_distributed_initialized() -> None:
    if not dist.is_initialized():
        raise RuntimeError("Distributed training is not initialized")


def _as_module(model: FSDPModule) -> nn.Module:
    return cast(nn.Module, model)


def _state_dict_options(*, full_state_dict: bool) -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True,
    )


def _build_stateful_checkpoint(
    *,
    model: FSDPModule,
    optimizer: Optimizer | None,
    lr_scheduler: LRScheduler | None,
    grad_scaler: torch.amp.GradScaler | None,
) -> "FSDP2Checkpoint":
    return FSDP2Checkpoint(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        options=_state_dict_options(full_state_dict=False),
    )


def _dcp_checkpoint_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / _DCP_CHECKPOINT_DIRNAME


def _metadata_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / _METADATA_FILENAME


def _save_metadata(base_dir: str | Path, metadata: dict[str, Any]) -> None:
    if dist.get_rank() == 0:
        path = _metadata_path(base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metadata, path)
    dist.barrier()


def _load_metadata(base_dir: str | Path) -> dict[str, Any]:
    dist.barrier()
    metadata = torch.load(_metadata_path(base_dir), map_location="cpu")
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a dictionary")
    return metadata


class FSDP2Checkpoint(Stateful):
    """Stateful wrapper for FSDP2 checkpointing with DCP."""

    def __init__(
        self,
        model: FSDPModule,
        optimizer: Optimizer | None,
        lr_scheduler: LRScheduler | None,
        grad_scaler: torch.amp.GradScaler | None,
        options: StateDictOptions,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_scaler = grad_scaler
        self.options = options

    def state_dict(self) -> dict[str, Any]:
        """Return the state dict for saving."""
        if self.optimizer is None:
            state_dict: dict[str, Any] = {
                "model": get_model_state_dict(
                    _as_module(self.model),
                    options=self.options,
                )
            }
        else:
            model_state_dict, optimizer_state_dict = get_state_dict(
                _as_module(self.model),
                self.optimizer,
                options=self.options,
            )
            state_dict = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
            }
        if self.lr_scheduler is not None:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.grad_scaler is not None and self.grad_scaler.is_enabled():
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a previously saved state dict."""
        if self.optimizer is not None and "optimizer" in state_dict:
            set_state_dict(
                model=_as_module(self.model),
                optimizers=self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=self.options,
            )
        else:
            set_model_state_dict(
                _as_module(self.model),
                state_dict["model"],
                options=self.options,
            )
        scheduler_state = state_dict.get("lr_scheduler")
        if self.lr_scheduler is not None and scheduler_state is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        grad_scaler_state = state_dict.get("grad_scaler")
        if self.grad_scaler is not None and grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(grad_scaler_state)


def save_checkpoint(
    model: FSDPModule,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    save_dir: str,
    *,
    step: int,
    policy_version: int,
    grad_scaler: torch.amp.GradScaler | None = None,
    save_optimizer: bool = True,
    save_full_weights: bool = False,
) -> None:
    """Save an FSDP2 checkpoint."""
    _require_distributed_initialized()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    clear_memory()
    dist.barrier()

    dcp_save_path = _dcp_checkpoint_path(save_path)
    checkpoint = _build_stateful_checkpoint(
        model=model,
        optimizer=optimizer if save_optimizer else None,
        lr_scheduler=lr_scheduler if save_optimizer else None,
        grad_scaler=grad_scaler if save_optimizer else None,
    )
    dcp.save(
        {_CHECKPOINT_KEY: checkpoint},
        checkpoint_id=str(dcp_save_path),
    )
    _save_metadata(
        save_path,
        {
            "step": step,
            "policy_version": policy_version,
            "has_optimizer_state": save_optimizer,
        },
    )
    logger.info("[FSDP2] Saved DCP checkpoint to {}", dcp_save_path)

    dist.barrier()

    if save_full_weights:
        model_state_dict = get_model_state_dict(
            _as_module(model),
            options=_state_dict_options(full_state_dict=True),
        )
        if dist.get_rank() == 0:
            model_save_path = save_path / _FULL_WEIGHTS_PATH
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state_dict, model_save_path)
            logger.info("[FSDP2] Saved full weights to {}", model_save_path)

    dist.barrier()


def load_checkpoint(
    model: FSDPModule,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    load_dir: str,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    """Load an FSDP2 checkpoint."""
    _require_distributed_initialized()

    dcp_load_path = _dcp_checkpoint_path(load_dir)
    checkpoint = _build_stateful_checkpoint(
        model=model,
        optimizer=optimizer if load_optimizer else None,
        lr_scheduler=lr_scheduler if load_optimizer else None,
        grad_scaler=grad_scaler if load_optimizer else None,
    )
    dcp.load(
        {_CHECKPOINT_KEY: checkpoint},
        checkpoint_id=str(dcp_load_path),
    )

    dist.barrier()
    logger.info("[FSDP2] Loaded checkpoint from {}", load_dir)
    return _load_metadata(load_dir)


def get_full_state_dict(model: FSDPModule) -> dict[str, Any]:
    """Get the full model state dict gathered from all ranks."""
    return get_model_state_dict(
        _as_module(model),
        options=_state_dict_options(full_state_dict=True),
    )


def _resolve_checkpoint_path(
    checkpoints_dir: Path,
    *,
    latest: bool,
    step: int | None,
) -> Path | None:
    if not checkpoints_dir.exists():
        return None

    if step is not None:
        candidate = checkpoints_dir / f"step_{step:08d}"
        metadata = candidate / _METADATA_FILENAME
        return candidate if metadata.exists() else None

    if not latest:
        return None

    candidates = sorted(
        path
        for path in checkpoints_dir.glob("step_*")
        if path.is_dir() and (path / _METADATA_FILENAME).exists()
    )
    if not candidates:
        return None
    return candidates[-1]


def save_backend_checkpoint(
    backend: FSDP2Engine,
    *,
    step: int,
    policy_version: int,
    save_optimizer: bool = True,
) -> CheckpointInfo:
    """Save a checkpoint for an initialized FSDP2 engine."""
    checkpoints_dir = Path(backend.cfg.train.checkpoints)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoints_dir / f"step_{step:08d}"
    save_checkpoint(
        model=cast(FSDPModule, backend.main_model),
        optimizer=backend.optimizer,
        lr_scheduler=backend.scheduler,
        save_dir=str(path),
        step=step,
        policy_version=policy_version,
        grad_scaler=backend.grad_scaler,
        save_optimizer=save_optimizer,
        save_full_weights=False,
    )
    return CheckpointInfo(
        step=step,
        policy_version=policy_version,
        path=str(path),
    )


def load_backend_checkpoint(
    backend: FSDP2Engine,
    *,
    latest: bool = True,
    step: int | None = None,
    load_optimizer: bool = True,
) -> CheckpointInfo | None:
    """Load a checkpoint for an initialized FSDP2 engine if one exists."""
    checkpoints_dir = Path(backend.cfg.train.checkpoints)
    path = _resolve_checkpoint_path(checkpoints_dir, latest=latest, step=step)
    if path is None:
        return None

    metadata = load_checkpoint(
        model=cast(FSDPModule, backend.main_model),
        optimizer=backend.optimizer,
        lr_scheduler=backend.scheduler,
        load_dir=str(path),
        grad_scaler=backend.grad_scaler,
        load_optimizer=load_optimizer,
    )
    loaded_step = int(metadata["step"])
    loaded_policy_version = int(metadata.get("policy_version", loaded_step))
    return CheckpointInfo(
        step=loaded_step,
        policy_version=loaded_policy_version,
        path=str(path),
    )
