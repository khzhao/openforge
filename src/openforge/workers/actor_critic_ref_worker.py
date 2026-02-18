# Copyright 2026 openforge

from tensordict import TensorDictBase

from openforge.configs import OpenForgeConfig


class ActorCriticRefWorker:
    """Single rank worker for actor/critic/ref optimization."""

    def __init__(self, config: OpenForgeConfig, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.global_step = 0

    def forward_backward(self, batch: TensorDictBase) -> dict[str, float]:
        """Run one optimization step on this rank."""
        raise NotImplementedError

    def save_checkpoint(self, save_dir: str) -> None:
        """Persist rank-local worker state to a checkpoint directory."""
        raise NotImplementedError
