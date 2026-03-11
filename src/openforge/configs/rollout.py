# Copyright 2026 openforge

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel, Reward
from .cluster import ClusterConfig
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


@dataclass(slots=True)
class ResolvedRolloutEngine:
    """Concrete rollout engine instance derived from user intent."""

    engine_id: int
    group_name: str
    role: RolloutRole
    replica_index: int
    num_gpus: int
    num_cpus: int
    parallel: ParallelismConfig
    placement: PlacementConfig


@dataclass(slots=True)
class ResolvedRolloutTopology:
    """Expanded rollout topology for runtime code."""

    engines: list[ResolvedRolloutEngine]

    @property
    def total_gpus(self) -> int:
        return sum(engine.num_gpus for engine in self.engines)

    @property
    def total_cpus(self) -> int:
        return sum(engine.num_cpus for engine in self.engines)


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

    def resolve(self, cluster: ClusterConfig) -> ResolvedRolloutTopology:
        engines: list[ResolvedRolloutEngine] = []
        engine_id = 0
        total_gpus = 0
        total_cpus = 0

        for group in self.engines:
            if group.num_gpus > cluster.gpus_per_node:
                raise ValueError(
                    f"rollout engine {group.name} requests {group.num_gpus} GPUs "
                    f"per engine, but each node only has {cluster.gpus_per_node} GPUs"
                )
            if group.num_cpus > cluster.cpus_per_node:
                raise ValueError(
                    f"rollout engine {group.name} requests {group.num_cpus} CPUs "
                    f"per engine, but each node only has {cluster.cpus_per_node} CPUs"
                )
            total_gpus += group.total_gpus
            total_cpus += group.total_cpus
            if total_gpus > cluster.total_gpus:
                raise ValueError(
                    "rollout topology requests "
                    f"{total_gpus} GPUs, but only {cluster.total_gpus} are available"
                )
            if total_cpus > cluster.total_cpus:
                raise ValueError(
                    "rollout topology requests "
                    f"{total_cpus} CPUs, but only {cluster.total_cpus} are available"
                )

            for replica_index in range(group.replicas):
                engines.append(
                    ResolvedRolloutEngine(
                        engine_id=engine_id,
                        group_name=group.name,
                        role=group.role,
                        replica_index=replica_index,
                        num_gpus=group.num_gpus,
                        num_cpus=group.num_cpus,
                        parallel=group.parallel,
                        placement=group.placement,
                    )
                )
                engine_id += 1

        return ResolvedRolloutTopology(engines=engines)

    @property
    def num_engines(self) -> int:
        return sum(group.replicas for group in self.engines)
