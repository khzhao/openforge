# Copyright 2026 openforge

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
    def _validate_algo_requirements(self) -> "OpenForgeConfig":
        if self.algo.kl_coef > 0.0 and self.model.reference_model_name_or_path is None:
            raise ValueError(
                "model.reference_model_name_or_path must be set when algo.kl_coef > 0.0"
            )
        return self

    @model_validator(mode="after")
    def _validate_cluster_allocations(self) -> "OpenForgeConfig":
        train_topology = self.train.resolve(self.cluster)
        rollout_topology = self.rollout.resolve(self.cluster)
        total_gpus = train_topology.total_gpus + rollout_topology.total_gpus
        total_cpus = train_topology.total_cpus + rollout_topology.total_cpus

        if total_gpus > self.cluster.total_gpus:
            raise ValueError(
                "cluster is oversubscribed on GPUs: "
                f"requested {total_gpus}, available {self.cluster.total_gpus}"
            )
        if total_cpus > self.cluster.total_cpus:
            raise ValueError(
                "cluster is oversubscribed on CPUs: "
                f"requested {total_cpus}, available {self.cluster.total_cpus}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenForgeConfig":
        """Load OpenForgeConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)
