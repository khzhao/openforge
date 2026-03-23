# Copyright 2026 openforge

from __future__ import annotations

from typing import TYPE_CHECKING

from openforge.algo import GRPOAlgorithm
from openforge.configs.models import OpenForgeConfig

if TYPE_CHECKING:
    from openforge.rollout.manager import RolloutManager
    from openforge.train.manager import TrainManager

__all__ = [
    "create_algorithm",
    "create_rollout_manager",
    "create_train_manager",
    "register_rollout",
]


def create_algorithm(cfg: OpenForgeConfig):
    """Create the configured training algorithm."""
    if cfg.algo.name in {"grpo", "grpo_tis"}:
        return GRPOAlgorithm(cfg.algo)
    raise ValueError(f"unsupported algorithm: {cfg.algo.name}")


def create_rollout_manager(
    cfg: OpenForgeConfig,
    placement_groups,
    **kwargs,
) -> RolloutManager:
    """Create a rollout manager."""
    from openforge.rollout.manager import RolloutManager

    manager = RolloutManager(cfg, placement_groups)
    manager.initialize(**kwargs)
    return manager


def create_train_manager(
    cfg: OpenForgeConfig,
    *,
    master_addr: str,
    master_port: int,
    placement_groups,
) -> TrainManager:
    """Create a train manager."""
    from openforge.train.manager import TrainManager

    manager = TrainManager()
    manager.initialize(
        cfg,
        master_addr=master_addr,
        master_port=master_port,
        placement_group=placement_groups,
    )
    return manager


def register_rollout(
    train_manager: TrainManager,
    rollout_manager: RolloutManager,
) -> None:
    """Register a rollout manager with a train manager."""
    train_manager.register_rollout(rollout_manager.engine_workers)
