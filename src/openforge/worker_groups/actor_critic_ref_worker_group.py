# Copyright 2026 openforge

from typing import Any

from tensordict import TensorDictBase

from openforge.configs import OpenForgeConfig


class ActorCriticRefWorkerGroup:
    """Coordinates all actor/critic/ref rank workers."""

    def __init__(self, config: OpenForgeConfig, workers: list[Any]) -> None:
        if not workers:
            raise ValueError("workers must not be empty")
        self.config = config
        self.workers = workers

    def forward_backward(self, batch: TensorDictBase) -> dict[str, float]:
        """Shard/dispatch one optimization step across rank workers."""
        raise NotImplementedError

    def save_checkpoint(self, save_dir: str) -> None:
        """Save checkpoints across all rank workers."""
        raise NotImplementedError
