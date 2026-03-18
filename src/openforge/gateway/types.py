# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ChatMessageRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "CreateSessionResult",
    "EndSessionRequest",
    "EndSessionResponse",
    "GenerateChoice",
    "GenerateChoiceResponse",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResult",
    "RolloutReward",
    "RolloutRewardRequest",
]


@dataclass(slots=True)
class CreateSessionResult:
    session_id: str
    model: str


@dataclass(slots=True)
class GenerateChoice:
    rollout_id: str
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    rollout_model_version: int


@dataclass(slots=True)
class GenerateResult:
    session_id: str
    choices: list[GenerateChoice]


@dataclass(slots=True)
class RolloutReward:
    rollout_id: str
    reward: float


class CreateSessionRequest(BaseModel):
    model: str


class CreateSessionResponse(BaseModel):
    session_id: str
    model: str


class ChatMessageRequest(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    session_id: str
    messages: list[ChatMessageRequest]
    n: int = Field(default=1, gt=0)
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class GenerateChoiceResponse(BaseModel):
    rollout_id: str
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    rollout_model_version: int


class GenerateResponse(BaseModel):
    session_id: str
    choices: list[GenerateChoiceResponse]


class RolloutRewardRequest(BaseModel):
    rollout_id: str
    reward: float


class EndSessionRequest(BaseModel):
    session_id: str
    rewards: list[RolloutRewardRequest]


class EndSessionResponse(BaseModel):
    session_id: str
    status: str
