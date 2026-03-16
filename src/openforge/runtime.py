# Copyright 2026 openforge

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.manager import RolloutManager
from openforge.train.group import TrainManager

__all__ = [
    "create_rollout_manager",
    "create_train_manager",
    "register_rollout",
]


def create_rollout_manager(
    cfg: OpenForgeConfig,
    placement_groups,
    **kwargs,
) -> RolloutManager:
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
    train_manager.register_rollout(rollout_manager.engine_workers)
