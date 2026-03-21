# Copyright 2026 openforge

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from openforge.configs.algo import AlgorithmConfig
from openforge.configs.models import ModelConfig
from openforge.configs.rollout import RolloutConfig
from openforge.configs.train import TrainConfig

__all__ = [
    "ChatMessage",
    "ChatChoice",
    "DiscardTrajectoryRequest",
    "EndTrajectoriesRequest",
    "EndTrajectoriesResponse",
    "CompletionUsage",
    "ErrorTrajectoriesRequest",
    "ExportCheckpointRequest",
    "ExportCheckpointResponse",
    "EndSessionRequest",
    "EndSessionResponse",
    "EndTrajectoryRequest",
    "EndTrajectoryResponse",
    "ErrorTrajectoryRequest",
    "GenerateRequest",
    "GenerateResponse",
    "ModelRecord",
    "ModelsResponse",
    "RuntimeConfig",
    "StartSessionRequest",
    "StartSessionResponse",
    "StartTrajectoryGroupsRequest",
    "StartTrajectoryGroupsResponse",
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
    policy_version: int


class StartTrajectoryRequest(BaseModel):
    """Request payload for starting a trajectory, optionally inside a group."""

    session_id: str
    group_id: str | None = None


class StartTrajectoryResponse(BaseModel):
    """Response payload for a started trajectory."""

    session_id: str
    trajectory_id: str
    group_id: str | None


class StartTrajectoryGroupsRequest(BaseModel):
    """Request payload for starting trajectories across multiple groups."""

    session_id: str
    counts: list[int]
    group_ids: list[str | None]


class StartTrajectoryGroupsResponse(BaseModel):
    """Response payload for starting trajectories across multiple groups."""

    session_id: str
    trajectory_ids: list[list[str]]


class ChatMessage(BaseModel):
    """One chat-style message supplied by the user."""

    role: str
    content: str


class GenerateRequest(BaseModel):
    """Request payload for generating on one trajectory."""

    session_id: str
    trajectory_id: str
    group_id: str | None = None
    messages: list[ChatMessage]
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class ChatChoice(BaseModel):
    """One assistant choice in an OpenAI-style chat completion."""

    finish_reason: str
    index: int
    message: ChatMessage
    logprobs: None = None


class CompletionUsage(BaseModel):
    """Token usage for one completion."""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    """OpenAI-style chat completion returned by the gateway."""

    id: str
    choices: list[ChatChoice]
    created: int
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    usage: CompletionUsage
    metadata: dict[str, Any] = Field(default_factory=dict)


class EndTrajectoryRequest(BaseModel):
    """Request payload for ending one trajectory with a final reward."""

    session_id: str
    trajectory_id: str
    final_reward: float


class EndTrajectoriesRequest(BaseModel):
    """Request payload for ending multiple trajectories with rewards."""

    session_id: str
    trajectory_ids: list[str]
    final_rewards: list[float]


class ErrorTrajectoryRequest(BaseModel):
    """Request payload for marking a trajectory as errored."""

    session_id: str
    trajectory_id: str


class ErrorTrajectoriesRequest(BaseModel):
    """Request payload for marking multiple trajectories as errored."""

    session_id: str
    trajectory_ids: list[str]


class DiscardTrajectoryRequest(BaseModel):
    """Request payload for discarding a trajectory without counting it as failed."""

    session_id: str
    trajectory_id: str


class EndTrajectoryResponse(BaseModel):
    """Response payload for an ended trajectory."""

    session_id: str
    trajectory_id: str
    status: str


class EndTrajectoriesResponse(BaseModel):
    """Response payload for multiple ended trajectories."""

    session_id: str
    trajectory_ids: list[str]
    status: str


class ExportCheckpointRequest(BaseModel):
    """Request payload for exporting the current train checkpoint."""

    session_id: str


class ExportCheckpointResponse(BaseModel):
    """Response payload for one exported checkpoint."""

    session_id: str
    policy_version: int
    checkpoint_path: str


class EndSessionRequest(BaseModel):
    """Request payload for ending a session."""

    session_id: str


class EndSessionResponse(BaseModel):
    """Response payload for an ended session."""

    session_id: str
    status: str
