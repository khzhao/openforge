# Copyright 2026 openforge

from __future__ import annotations

from typing import TYPE_CHECKING

from openforge.algo import GRPOAlgorithm
from openforge.configs.models import OpenForgeConfig

if TYPE_CHECKING:
    from openforge.rollout.manager import RolloutManager
    from openforge.train.runtime import TrainRuntime

__all__ = [
    "create_algorithm",
    "create_rollout_manager",
    "create_train_runtime",
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


def create_train_runtime(
    cfg: OpenForgeConfig,
    *,
    master_addr: str,
    master_port: int,
    placement_groups,
) -> TrainRuntime:
    """Create a train runtime."""
    from openforge.train.manager import TrainManager
    from openforge.train.runtime import TrainRuntime

    manager = TrainManager()
    manager.initialize(
        cfg,
        master_addr=master_addr,
        master_port=master_port,
        placement_group=placement_groups,
    )
    try:
        return TrainRuntime(manager)
    except Exception:
        manager.shutdown()
        raise
