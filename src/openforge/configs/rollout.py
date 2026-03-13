# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel, Reward
from .topology import ParallelismConfig

RolloutRole = Literal["regular"]


@dataclass(slots=True)
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


@dataclass(slots=True)
class RolloutEndpoint:
    """Routable rollout endpoint or underlying engine endpoint."""

    name: str
    role: RolloutRole
    host: str
    port: int | None
    disaggregation_bootstrap_port: int | None
    url: str | None
    healthy: bool
    policy_version: int | None
    model_path: str | None


class SGLangRequestConfig(OpenForgeBaseModel):
    """Request-time generation defaults for SGLang-backed rollout."""

    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    stop: list[str]
    stop_token_ids: list[int]
    skip_special_tokens: bool
    no_stop_trim: bool
    spaces_between_words: bool


class RolloutServerGroupConfig(OpenForgeBaseModel):
    """A homogeneous rollout server group with shared resources and placement."""

    name: str
    role: RolloutRole
    replicas: int
    num_gpus_per_replica: int
    num_cpus_per_replica: int
    parallelism: ParallelismConfig

    @model_validator(mode="after")
    def _validate_group(self) -> RolloutServerGroupConfig:
        if self.replicas <= 0:
            raise ValueError("replicas must be > 0")
        if self.num_gpus_per_replica <= 0:
            raise ValueError("num_gpus_per_replica must be > 0")
        if self.num_cpus_per_replica < 0:
            raise ValueError("num_cpus_per_replica must be >= 0")
        if self.num_gpus_per_replica != self.parallelism.world_size:
            raise ValueError(
                "rollout.num_gpus_per_replica must match rollout parallelism world "
                f"size ({self.parallelism.world_size})"
            )
        return self

    @property
    def total_gpus(self) -> int:
        return self.replicas * self.num_gpus_per_replica

    @property
    def total_cpus(self) -> int:
        return self.replicas * self.num_cpus_per_replica


class RolloutConfig(OpenForgeBaseModel):
    """Configuration for the rollout process."""

    backend: Literal["sglang"]
    request: SGLangRequestConfig
    server_groups: list[RolloutServerGroupConfig]

    @model_validator(mode="after")
    def _validate_rollout_config(self) -> "RolloutConfig":
        if not self.server_groups:
            raise ValueError("rollout.server_groups must not be empty")

        names = [group.name for group in self.server_groups]
        if len(names) != len(set(names)):
            raise ValueError("rollout.server_groups must have unique names")

        roles = {group.role for group in self.server_groups}
        if not roles.issubset({"regular"}):
            raise ValueError("rollout.server_groups must all use role=regular")
        return self

    @property
    def num_server_replicas(self) -> int:
        return sum(group.replicas for group in self.server_groups)

    @property
    def num_server_groups(self) -> int:
        return len(self.server_groups)

    @property
    def total_gpus(self) -> int:
        return sum(group.total_gpus for group in self.server_groups)

    @property
    def total_cpus(self) -> int:
        return sum(group.total_cpus for group in self.server_groups)
