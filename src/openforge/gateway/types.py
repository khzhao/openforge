# Copyright 2026 openforge

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openforge.configs.algo import AlgorithmConfig
from openforge.configs.models import ModelConfig
from openforge.configs.rollout import RolloutConfig
from openforge.configs.train import TrainConfig

__all__ = [
    "ChatMessage",
    "EndSessionRequest",
    "EndSessionResponse",
    "EndTrajectoryRequest",
    "EndTrajectoryResponse",
    "GenerateRequest",
    "GenerateResponse",
    "ModelRecord",
    "ModelsResponse",
    "RuntimeConfig",
    "StartSessionRequest",
    "StartSessionResponse",
    "StartTrajectoryRequest",
    "StartTrajectoryResponse",
]


class ModelRecord(BaseModel):
    """One model exposed by the gateway."""

    id: str
    tokenizer: str


class ModelsResponse(BaseModel):
    """Response payload for listing available models."""

    models: list[ModelRecord]
    active_model: str | None


class RuntimeConfig(BaseModel):
    """User-facing runtime config for one started session."""

    algo: AlgorithmConfig
    model: ModelConfig
    train: TrainConfig
    rollout: RolloutConfig


class StartSessionRequest(BaseModel):
    """Request payload for starting a training session."""

    runtime: RuntimeConfig


class StartSessionResponse(BaseModel):
    """Response payload for a started session."""

    session_id: str
    model: str


class StartTrajectoryRequest(BaseModel):
    """Request payload for starting a trajectory, optionally from a parent."""

    session_id: str
    parent_trajectory_id: str | None = None


class StartTrajectoryResponse(BaseModel):
    """Response payload for a started trajectory."""

    session_id: str
    trajectory_id: str
    parent_trajectory_id: str | None


class ChatMessage(BaseModel):
    """One chat-style message supplied by the user."""

    role: str
    content: str


class GenerateRequest(BaseModel):
    """Request payload for generating on one trajectory."""

    session_id: str
    trajectory_id: str
    messages: list[ChatMessage]
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response payload for one generated trajectory continuation."""

    session_id: str
    trajectory_id: str
    token_ids: list[int]
    logprobs: list[float | None]
    finish_reason: str
    rollout_model_version: str


class EndTrajectoryRequest(BaseModel):
    """Request payload for ending one trajectory with a final reward."""

    session_id: str
    trajectory_id: str
    final_reward: float


class EndTrajectoryResponse(BaseModel):
    """Response payload for an ended trajectory."""

    session_id: str
    trajectory_id: str
    status: str


class EndSessionRequest(BaseModel):
    """Request payload for ending a session."""

    session_id: str


class EndSessionResponse(BaseModel):
    """Response payload for an ended session."""

    session_id: str
    status: str
