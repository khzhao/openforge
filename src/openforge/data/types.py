# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Session",
    "Trajectory",
    "TrajectoryStatus",
    "Turn",
]


TrajectoryStatus = Literal["active", "completed", "trained", "failed", "discarded"]


@dataclass(slots=True)
class Session:
    """User-facing conversation handle."""

    session_id: str
    model_name: str


@dataclass(slots=True)
class Trajectory:
    """One internal trajectory tracked underneath a user session."""

    trajectory_id: str
    session_id: str
    group_id: str | None
    status: TrajectoryStatus
    expected_group_size: int = 1
    final_reward: float | None = None

    def __post_init__(self) -> None:
        if self.expected_group_size <= 0:
            raise ValueError("expected_group_size must be >= 1")
        if self.final_reward is not None and self.status == "active":
            raise ValueError("final_reward may only be set on terminal trajectories")

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "trained", "failed", "discarded"}


@dataclass(slots=True)
class Turn:
    """One tokenized model completion appended to a trajectory."""

    trajectory_id: str
    turn_index: int
    rollout_model_version: str
    prompt_length: int
    token_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be >= 0")
        if not self.rollout_model_version:
            raise ValueError("rollout_model_version must be non-empty")
        if self.prompt_length < 0 or self.prompt_length > len(self.token_ids):
            raise ValueError("prompt_length must be between 0 and len(token_ids)")
        if len(self.position_ids) != len(self.token_ids):
            raise ValueError("position_ids must have the same length as token_ids")

        predicted_token_count = max(len(self.token_ids) - 1, 0)
        if len(self.loss_mask) != predicted_token_count:
            raise ValueError(
                "loss_mask must have one entry per predicted token: "
                f"{len(self.loss_mask)} != {predicted_token_count}"
            )

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.prompt_length]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]
