# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel, Reward
from .cluster import ClusterConfig
from .topology import ParallelismConfig, PlacementConfig

RolloutRole = Literal["prefill", "decode", "regular", "placeholder"]
EngineTopology = Literal["regular", "pd"]


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
    gpus_per_engine: int
    cpus_per_engine: int
    parallelism: ParallelismConfig
    placement: PlacementConfig

    @model_validator(mode="after")
    def _validate_group(self) -> RolloutEngineGroupConfig:
        if self.replicas <= 0:
            raise ValueError("replicas must be > 0")
        if self.gpus_per_engine <= 0:
            raise ValueError("gpus_per_engine must be > 0")
        if self.cpus_per_engine < 0:
            raise ValueError("cpus_per_engine must be >= 0")
        if self.gpus_per_engine != self.parallelism.world_size:
            raise ValueError(
                "gpus_per_engine must match rollout engine parallelism world size "
                f"({self.parallelism.world_size})"
            )
        return self

    @property
    def total_gpus(self) -> int:
        return self.replicas * self.gpus_per_engine

    @property
    def total_cpus(self) -> int:
        return self.replicas * self.cpus_per_engine


@dataclass(slots=True)
class ResolvedRolloutEngine:
    """Concrete rollout engine instance derived from user intent."""

    engine_id: int
    group_name: str
    role: RolloutRole
    replica_index: int
    gpus_per_engine: int
    cpus_per_engine: int
    parallelism: ParallelismConfig
    placement: PlacementConfig


@dataclass(slots=True)
class ResolvedRolloutTopology:
    """Expanded rollout topology for runtime code."""

    engine_topology: EngineTopology
    engines: list[ResolvedRolloutEngine]

    @property
    def total_gpus(self) -> int:
        return sum(engine.gpus_per_engine for engine in self.engines)

    @property
    def total_cpus(self) -> int:
        return sum(engine.cpus_per_engine for engine in self.engines)


class RolloutConfig(OpenForgeBaseModel):
    """Configuration for the rollout process."""

    backend: Literal["sglang"]
    request: SGLangRequestConfig
    engine_topology: EngineTopology
    engines: list[RolloutEngineGroupConfig]

    @model_validator(mode="after")
    def _validate_rollout_config(self) -> RolloutConfig:
        if not self.engines:
            raise ValueError("rollout.engines must not be empty")

        names = [engine.name for engine in self.engines]
        if len(names) != len(set(names)):
            raise ValueError("rollout engine names must be unique")

        routable_roles = {
            engine.role for engine in self.engines if engine.role != "placeholder"
        }
        if self.engine_topology == "pd":
            if "prefill" not in routable_roles or "decode" not in routable_roles:
                raise ValueError(
                    "pd rollout must include both prefill and decode engines"
                )
            if not routable_roles.issubset({"prefill", "decode"}):
                raise ValueError(
                    "pd rollout may only contain prefill, decode, or placeholder engines"
                )
        else:
            if not routable_roles.issubset({"regular"}):
                raise ValueError(
                    "regular rollout may only contain regular or placeholder engines"
                )
        return self

    def resolve(self, cluster: ClusterConfig) -> ResolvedRolloutTopology:
        engines: list[ResolvedRolloutEngine] = []
        engine_id = 0
        usage_by_node_pool: dict[str, dict[str, int]] = {}

        for group in self.engines:
            node_pool = cluster.get_pool(group.placement.node_pool)
            usage = usage_by_node_pool.setdefault(
                node_pool.node_pool, {"gpus": 0, "cpus": 0}
            )
            usage["gpus"] += group.total_gpus
            usage["cpus"] += group.total_cpus
            if usage["gpus"] > node_pool.total_gpus:
                raise ValueError(
                    f"rollout topology requests {usage['gpus']} GPUs from node pool "
                    f"{node_pool.node_pool}, but only {node_pool.total_gpus} are available"
                )
            if usage["cpus"] > node_pool.total_cpus:
                raise ValueError(
                    f"rollout topology requests {usage['cpus']} CPUs from node pool "
                    f"{node_pool.node_pool}, but only {node_pool.total_cpus} are available"
                )

            for replica_index in range(group.replicas):
                engines.append(
                    ResolvedRolloutEngine(
                        engine_id=engine_id,
                        group_name=group.name,
                        role=group.role,
                        replica_index=replica_index,
                        gpus_per_engine=group.gpus_per_engine,
                        cpus_per_engine=group.cpus_per_engine,
                        parallelism=group.parallelism,
                        placement=group.placement,
                    )
                )
                engine_id += 1

        return ResolvedRolloutTopology(
            engine_topology=self.engine_topology,
            engines=engines,
        )

    @property
    def num_engines(self) -> int:
        return sum(group.replicas for group in self.engines)
