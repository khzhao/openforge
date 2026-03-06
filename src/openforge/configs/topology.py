# Copyright 2026 openforge

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from .base import OpenForgeBaseModel


class PlacementStrategy(str, Enum):
    """Placement strategy for grouping workers onto nodes."""

    PACK = "PACK"
    SPREAD = "SPREAD"


class ParallelismConfig(OpenForgeBaseModel):
    """Distributed parallelism layout for a train world or rollout engine."""

    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1

    @model_validator(mode="after")
    def _validate_parallelism(self) -> ParallelismConfig:
        if self.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be > 0")
        if self.pipeline_parallel_size <= 0:
            raise ValueError("pipeline_parallel_size must be > 0")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be > 0")
        if self.context_parallel_size <= 0:
            raise ValueError("context_parallel_size must be > 0")
        if self.expert_parallel_size <= 0:
            raise ValueError("expert_parallel_size must be > 0")
        return self

    @property
    def model_parallel_size(self) -> int:
        return (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.context_parallel_size
            * self.expert_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.data_parallel_size * self.model_parallel_size


class PlacementConfig(OpenForgeBaseModel):
    """Placement target for a workload fragment."""

    node_pool: str
    strategy: PlacementStrategy = PlacementStrategy.PACK
