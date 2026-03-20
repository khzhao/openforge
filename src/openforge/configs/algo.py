# Copyright 2026 openforge

from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel


class AlgorithmConfig(OpenForgeBaseModel):
    """Training algorithm parameters."""

    name: Literal["grpo"] = "grpo"
    clip_range: float = 0.2
    normalize_group_std: bool = True
    entropy_coef: float = 0.0
    kl_coef: float = 0.0

    @model_validator(mode="after")
    def _validate_algorithm(self) -> "AlgorithmConfig":
        if self.clip_range < 0.0:
            raise ValueError("clip_range must be >= 0.0")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be >= 0.0")
        if self.kl_coef < 0.0:
            raise ValueError("kl_coef must be >= 0.0")
        return self
