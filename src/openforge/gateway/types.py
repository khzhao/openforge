# Copyright 2026 openforge

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field

from openforge.configs.algo import AlgorithmConfig
from openforge.configs.models import ModelConfig
from openforge.configs.rollout import RolloutConfig
from openforge.configs.train import TrainConfig
from openforge.data.types import TrajectoryStatus

__all__ = [
    "AssistantMessage",
    "AssistantToolCall",
    "AssistantToolCallFunction",
    "ChatMessage",
    "ChatCompletionChoice",
    "ChatCompletionCreateRequest",
    "ChatCompletionResponse",
    "ChatCompletionTool",
    "ChatCompletionToolChoice",
    "ChatCompletionToolFunction",
    "CompletionUsage",
    "DiscardTrajectoryRequest",
    "EndTrajectoriesRequest",
    "EndTrajectoriesResponse",
    "EndSessionRequest",
    "EndSessionResponse",
    "EndTrajectoryRequest",
    "EndTrajectoryResponse",
    "ErrorTrajectoriesRequest",
    "ErrorTrajectoryRequest",
    "ExportCheckpointRequest",
    "ExportCheckpointResponse",
    "ModelCard",
    "ModelListResponse",
    "RuntimeConfig",
    "StartSessionRequest",
    "StartSessionResponse",
    "StartTrajectoryGroupsRequest",
    "StartTrajectoryGroupsResponse",
    "StartTrajectoryRequest",
    "StartTrajectoryResponse",
    "ToolMessage",
    "TrajectoryStatusInfo",
    "TrajectoryStatusesRequest",
    "TrajectoryStatusesResponse",
    "WandbConfig",
    "chat_message_payload",
    "tool_payloads",
]


class WandbConfig(BaseModel):
    """Optional session-scoped W&B logging configuration."""

    enabled: bool = True
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    log_interval_seconds: float | None = None


class RuntimeConfig(BaseModel):
    """User-facing runtime config for one started session."""

    algo: AlgorithmConfig
    model: ModelConfig
    train: TrainConfig
    rollout: RolloutConfig
    wandb: WandbConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        """Load RuntimeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)


class StartSessionRequest(BaseModel):
    """Request payload for starting a training session."""

    runtime: RuntimeConfig


class StartSessionResponse(BaseModel):
    """Response payload for a started session."""

    session_id: str
    model: str
    policy_version: int


class TrajectoryStatusInfo(BaseModel):
    """One trajectory status entry for a session-owned trajectory."""

    trajectory_id: str
    status: TrajectoryStatus


class TrajectoryStatusesRequest(BaseModel):
    """Request payload for querying trajectory statuses."""

    session_id: str
    trajectory_ids: list[str]


class TrajectoryStatusesResponse(BaseModel):
    """Response payload for queried trajectory statuses."""

    session_id: str
    trajectories: list[TrajectoryStatusInfo]


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
    """One plain chat message with no tool-specific fields."""

    role: Literal["developer", "system", "user"]
    content: str


class AssistantToolCallFunction(BaseModel):
    """One function call payload in OpenAI chat-completions format."""

    name: str
    arguments: str


class AssistantToolCall(BaseModel):
    """One assistant tool call in OpenAI chat-completions format."""

    id: str
    type: Literal["function"] = "function"
    function: AssistantToolCallFunction


class AssistantMessage(BaseModel):
    """One assistant message in OpenAI chat-completions format."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[AssistantToolCall] | None = None


class ToolMessage(BaseModel):
    """One tool message in OpenAI chat-completions format."""

    role: Literal["tool"]
    content: str
    tool_call_id: str


ChatCompletionMessage = Annotated[
    ChatMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class ChatCompletionToolFunction(BaseModel):
    """One tool function definition in OpenAI chat-completions format."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionTool(BaseModel):
    """One tool definition in OpenAI chat-completions format."""

    type: Literal["function"]
    function: ChatCompletionToolFunction


class ChatCompletionNamedToolChoiceFunction(BaseModel):
    """One named tool target in OpenAI chat-completions format."""

    name: str


class ChatCompletionNamedToolChoice(BaseModel):
    """One explicit tool choice in OpenAI chat-completions format."""

    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


ChatCompletionToolChoice = (
    Literal["none", "auto", "required"] | ChatCompletionNamedToolChoice
)


class _OpenForgeRequestState(BaseModel):
    session_id: str
    trajectory_id: str
    group_id: str | None = None


class ChatCompletionCreateRequest(BaseModel):
    """OpenAI-style request payload for one chat completion."""

    # Framework-injected state for the active session and trajectory.
    openforge: _OpenForgeRequestState = Field(alias="_openforge")
    model: str
    messages: list[ChatCompletionMessage]
    frequency_penalty: float | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    temperature: float | None = None
    tool_choice: ChatCompletionToolChoice | None = None
    tools: list[ChatCompletionTool] | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    """One assistant choice in an OpenAI-style chat completion."""

    finish_reason: str
    index: int
    message: AssistantMessage
    logprobs: None = None


class CompletionUsage(BaseModel):
    """Token usage for one completion."""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-style chat completion returned by the gateway."""

    id: str
    choices: list[ChatCompletionChoice]
    created: int
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    usage: CompletionUsage
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCard(BaseModel):
    """One model exposed by the OpenAI-style models endpoint."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "openforge"


class ModelListResponse(BaseModel):
    """OpenAI-style list-models response."""

    object: Literal["list"] = "list"
    data: list[ModelCard]


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


def chat_message_payload(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert one OpenAI-style message into the tokenizer payload shape."""
    payload = message.model_dump(mode="json", exclude_none=True)
    if payload["role"] != "assistant" or "tool_calls" not in payload:
        return payload

    tool_calls = []
    for tool_call in payload["tool_calls"]:
        function = dict(tool_call["function"])
        function["arguments"] = json.loads(function["arguments"])
        tool_calls.append(
            {
                "id": tool_call["id"],
                "type": tool_call["type"],
                "function": function,
            }
        )

    payload["tool_calls"] = tool_calls
    return payload


def tool_payloads(
    tools: list[ChatCompletionTool] | None,
) -> list[dict[str, Any]] | None:
    """Convert tool definitions into the tokenizer payload shape."""
    if tools is None:
        return None
    return [tool.model_dump(mode="json", exclude_none=True) for tool in tools]
