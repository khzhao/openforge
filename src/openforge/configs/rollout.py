# Copyright 2026 openforge

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel, Reward
from .topology import ParallelismConfig, PlacementConfig

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
    role: str
    host: str
    port: int | None
    bootstrap_port: int | None
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


class RolloutEngineGroupConfig(OpenForgeBaseModel):
    """A homogeneous group of rollout engines with the same role and shape."""

    name: str
    role: RolloutRole
    replicas: int
    num_gpus: int
    num_cpus: int
    parallel: ParallelismConfig
    placement: PlacementConfig

    @model_validator(mode="after")
    def _validate_group(self) -> "RolloutEngineGroupConfig":
        if self.replicas <= 0:
            raise ValueError("replicas must be > 0")
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be > 0")
        if self.num_cpus < 0:
            raise ValueError("num_cpus must be >= 0")
        if self.num_gpus != self.parallel.world_size:
            raise ValueError(
                "rollout.num_gpus must match rollout parallel world size "
                f"({self.parallel.world_size})"
            )
        return self

    @property
    def total_gpus(self) -> int:
        return self.replicas * self.num_gpus

    @property
    def total_cpus(self) -> int:
        return self.replicas * self.num_cpus


class RolloutConfig(OpenForgeBaseModel):
    """Configuration for the rollout process."""

    backend: Literal["sglang"]
    request: SGLangRequestConfig
    engines: list[RolloutEngineGroupConfig]

    @model_validator(mode="after")
    def _validate_rollout_config(self) -> "RolloutConfig":
        if not self.engines:
            raise ValueError("rollout.engines must not be empty")

        names = [engine.name for engine in self.engines]
        if len(names) != len(set(names)):
            raise ValueError("rollout engine names must be unique")

        roles = {engine.role for engine in self.engines}
        if not roles.issubset({"regular"}):
            raise ValueError("rollout engines must all use role=regular")
        return self

    @property
    def num_engines(self) -> int:
        return sum(group.replicas for group in self.engines)

    @property
    def total_gpus(self) -> int:
        return sum(group.total_gpus for group in self.engines)

    @property
    def total_cpus(self) -> int:
        return sum(group.total_cpus for group in self.engines)
