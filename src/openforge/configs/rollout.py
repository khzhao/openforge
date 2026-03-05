# Copyright 2026 openforge

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel, Reward


class RolloutDatum(OpenForgeBaseModel):
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


class SGLangConfig(OpenForgeBaseModel):
    """SGLang backend configuration."""

    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    stop: list[str]
    stop_token_ids: list[int]
    skip_special_tokens: bool
    no_stop_trim: bool
    spaces_between_words: bool


class RolloutConfig(OpenForgeBaseModel):
    """Configuration for the rollout process."""

    # Backend selection and backend-specific settings.
    backend: Literal["sglang"]
    backend_cfg: SGLangConfig

    # Cluster resources.
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int
    num_gpus_per_engine: int
    num_cpus_per_engine: int

    @model_validator(mode="after")
    def _validate_resource_layout(self) -> RolloutConfig:
        total_gpus = self.num_nodes * self.num_gpus_per_node
        if self.num_gpus_per_engine <= 0:
            raise ValueError("num_gpus_per_engine must be > 0")
        if total_gpus % self.num_gpus_per_engine != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by "
                f"num_gpus_per_engine ({self.num_gpus_per_engine})"
            )
        return self

    @property
    def num_engines(self) -> int:
        total_gpus = self.num_nodes * self.num_gpus_per_node
        return int(total_gpus / self.num_gpus_per_engine)
