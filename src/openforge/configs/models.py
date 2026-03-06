# Copyright 2026 openforge

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from .algo import AlgorithmConfig
from .base import OpenForgeBaseModel
from .cluster import ClusterConfig
from .rollout import RolloutConfig
from .train import TrainConfig


class DataConfig(OpenForgeBaseModel):
    """Configuration for data source behavior."""

    backend: str


class GatewayConfig(OpenForgeBaseModel):
    """Configuration for the Gateway server."""

    host: str
    port: int
    backend_url: str


class ModelConfig(OpenForgeBaseModel):
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str
    reference_model_name_or_path: str | None = None


class OpenForgeConfig(OpenForgeBaseModel):
    """Configuration for the OpenForge project."""

    data: DataConfig
    gateway: GatewayConfig
    model: ModelConfig
    algo: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    cluster: ClusterConfig
    train: TrainConfig
    rollout: RolloutConfig

    @model_validator(mode="after")
    def _validate_cluster_allocations(self) -> OpenForgeConfig:
        usage_by_pool: dict[str, dict[str, int]] = {
            pool.node_pool: {"gpus": 0, "cpus": 0} for pool in self.cluster
        }

        train_topology = self.train.resolve(self.cluster)
        usage_by_pool[train_topology.node_pool]["gpus"] += train_topology.total_gpus
        usage_by_pool[train_topology.node_pool]["cpus"] += train_topology.total_cpus

        rollout_topology = self.rollout.resolve(self.cluster)
        for engine in rollout_topology.engines:
            usage_by_pool[engine.placement.node_pool]["gpus"] += engine.gpus_per_engine
            usage_by_pool[engine.placement.node_pool]["cpus"] += engine.cpus_per_engine

        for pool in self.cluster:
            usage = usage_by_pool[pool.node_pool]
            if usage["gpus"] > pool.total_gpus:
                raise ValueError(
                    f"node pool {pool.node_pool} is oversubscribed on GPUs: "
                    f"requested {usage['gpus']}, available {pool.total_gpus}"
                )
            if usage["cpus"] > pool.total_cpus:
                raise ValueError(
                    f"node pool {pool.node_pool} is oversubscribed on CPUs: "
                    f"requested {usage['cpus']}, available {pool.total_cpus}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenForgeConfig:
        """Load OpenForgeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)
