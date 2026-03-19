# Copyright 2026 openforge

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .algo import AlgorithmConfig
from .base import OpenForgeBaseModel
from .cluster import ClusterConfig
from .rollout import RolloutConfig
from .train import TrainConfig


class DataConfig(OpenForgeBaseModel):
    """Configuration for data source behavior."""

    path: str | None = None


class GatewayConfig(OpenForgeBaseModel):
    """Configuration for the Gateway server."""

    host: str
    port: int


class ModelConfig(OpenForgeBaseModel):
    """Configuration for the model."""

    model_name_or_path: str
    tokenizer_name_or_path: str
    reference_model_name_or_path: str | None = None
    attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = "sdpa"


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
        total_gpus = self.train.total_gpus + self.rollout.total_gpus
        total_cpus = self.train.total_cpus + self.rollout.total_cpus

        if self.train.total_gpus > self.cluster.total_gpus:
            raise ValueError(
                "train topology requests "
                f"{self.train.total_gpus} GPUs, but only {self.cluster.total_gpus} are available"
            )
        if self.train.total_cpus > self.cluster.total_cpus:
            raise ValueError(
                "train topology requests "
                f"{self.train.total_cpus} CPUs, but only {self.cluster.total_cpus} are available"
            )

        for engine_group in self.rollout.engine_groups:
            if engine_group.num_gpus_per_replica > self.cluster.gpus_per_node:
                raise ValueError(
                    f"rollout engine group {engine_group.name} requests "
                    f"{engine_group.num_gpus_per_replica} GPUs per replica, "
                    f"but each node only has {self.cluster.gpus_per_node} GPUs"
                )
            if engine_group.num_cpus_per_replica > self.cluster.cpus_per_node:
                raise ValueError(
                    f"rollout engine group {engine_group.name} requests "
                    f"{engine_group.num_cpus_per_replica} CPUs per replica, "
                    f"but each node only has {self.cluster.cpus_per_node} CPUs"
                )

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


class GatewayServerConfig(OpenForgeBaseModel):
    """Server-owned gateway configuration."""

    data: DataConfig
    gateway: GatewayConfig
    cluster: ClusterConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GatewayServerConfig":
        """Load GatewayServerConfig from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.model_validate(raw)
