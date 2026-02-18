# Copyright 2026 openforge

from dataclasses import dataclass

Reward = list[float] | float | None


@dataclass
class RolloutDatum:
    """Single rollout item stored for training consumption."""

    sample_id: str
    prompt_id: str
    group_id: str
    turn_index: int
    rollout_model_version: int
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    old_logprobs: list[float]
    reward: Reward = None
    consumed: bool = False


@dataclass
class DataConfig:
    """Configuration for data source behavior."""

    backend: str = "memory"
