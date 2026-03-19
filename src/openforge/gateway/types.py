# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ChatMessage",
    "EndSessionRequest",
    "EndSessionResponse",
    "EndTrajectoryRequest",
    "EndTrajectoryResponse",
    "GetPolicyVersionRequest",
    "GetPolicyVersionResponse",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResult",
    "ModelRecord",
    "ModelsResponse",
    "StartSessionRequest",
    "StartSessionResponse",
    "StartSessionResult",
    "StartTrajectoryRequest",
    "StartTrajectoryResponse",
    "StartTrajectoryResult",
]


@dataclass(slots=True)
class StartSessionResult:
    """Internal result for a newly started session."""

    session_id: str
    model: str


@dataclass(slots=True)
class StartTrajectoryResult:
    """Internal result for a newly started trajectory."""

    session_id: str
    trajectory_id: str
    parent_trajectory_id: str | None


@dataclass(slots=True)
class GenerateResult:
    """Internal result for one generated trajectory continuation."""

    session_id: str
    trajectory_id: str
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    rollout_model_version: int


class ModelRecord(BaseModel):
    """One model exposed by the gateway."""

    id: str
    tokenizer: str


class ModelsResponse(BaseModel):
    """Response payload for listing available models."""

    models: list[ModelRecord]
    active_model: str | None


class StartSessionRequest(BaseModel):
    """Request payload for starting a training session."""

    model: str


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
    logprobs: list[float]
    finish_reason: str
    rollout_model_version: int


class GetPolicyVersionRequest(BaseModel):
    """Request payload for retrieving the current policy version."""

    session_id: str


class GetPolicyVersionResponse(BaseModel):
    """Response payload for the current policy version."""

    session_id: str
    policy_version: int


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
