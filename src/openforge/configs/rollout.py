# Copyright 2026 openforge

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
    gpus_per_engine: int
    cpus_per_engine: int
    parallelism: ParallelismConfig
    placement: PlacementConfig

    @model_validator(mode="after")
    def _validate_group(self) -> "RolloutEngineGroupConfig":
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
    def _validate_rollout_config(self) -> "RolloutConfig":
        if not self.engines:
            raise ValueError("rollout.engines must not be empty")

        names = [engine.name for engine in self.engines]
        if len(names) != len(set(names)):
            raise ValueError("rollout engine names must be unique")

        routable_roles = {
            engine.role for engine in self.engines if engine.role != "placeholder"
        }
        if not routable_roles:
            raise ValueError("rollout must include at least one non-placeholder engine")
        if self.engine_topology == "pd":
            if "prefill" not in routable_roles or "decode" not in routable_roles:
                raise ValueError(
                    "pd rollout must include both prefill and decode engines"
                )
            if not routable_roles.issubset({"prefill", "decode"}):
                raise ValueError(
                    "pd rollout may only contain prefill, decode, or placeholder engines"
                )

            prefill_shapes = {
                self._parallelism_signature(engine.parallelism)
                for engine in self.engines
                if engine.role == "prefill"
            }
            decode_shapes = {
                self._parallelism_signature(engine.parallelism)
                for engine in self.engines
                if engine.role == "decode"
            }
            if len(prefill_shapes) > 1:
                raise ValueError(
                    "pd rollout currently requires all prefill engines to share the same parallelism"
                )
            if len(decode_shapes) > 1:
                raise ValueError(
                    "pd rollout currently requires all decode engines to share the same parallelism"
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
            if group.gpus_per_engine > node_pool.num_gpus_per_node:
                raise ValueError(
                    f"rollout engine {group.name} requests {group.gpus_per_engine} GPUs "
                    f"per engine, but node pool {node_pool.node_pool} only has "
                    f"{node_pool.num_gpus_per_node} GPUs per node"
                )
            if group.cpus_per_engine > node_pool.num_cpus_per_node:
                raise ValueError(
                    f"rollout engine {group.name} requests {group.cpus_per_engine} CPUs "
                    f"per engine, but node pool {node_pool.node_pool} only has "
                    f"{node_pool.num_cpus_per_node} CPUs per node"
                )
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

    @staticmethod
    def _parallelism_signature(parallelism: ParallelismConfig) -> tuple[int, ...]:
        return (
            parallelism.data_parallel_size,
            parallelism.pipeline_parallel_size,
            parallelism.tensor_parallel_size,
            parallelism.context_parallel_size,
            parallelism.expert_parallel_size,
        )
