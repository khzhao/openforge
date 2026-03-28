# Copyright 2026 openforge

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import OpenForgeBaseModel

__all__ = [
    "AlgorithmConfig",
    "GRPOConfig",
    "GRPOTISConfig",
]


class _GRPOBaseConfig(OpenForgeBaseModel):
    clip_range: float = 0.2
    clip_range_low: float | None = None
    clip_range_high: float | None = None
    normalize_group_std: bool = True
    entropy_coef: float = 0.0
    kl_coef: float = 0.0

    @model_validator(mode="after")
    def _validate_grpo(self) -> "_GRPOBaseConfig":
        if self.clip_range < 0.0:
            raise ValueError("clip_range must be >= 0.0")
        if self.clip_range_low is not None and self.clip_range_low < 0.0:
            raise ValueError("clip_range_low must be >= 0.0")
        if self.clip_range_high is not None and self.clip_range_high < 0.0:
            raise ValueError("clip_range_high must be >= 0.0")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be >= 0.0")
        if self.kl_coef < 0.0:
            raise ValueError("kl_coef must be >= 0.0")
        return self


class GRPOConfig(_GRPOBaseConfig):
    """GRPO configuration."""

    name: Literal["grpo"] = "grpo"


class GRPOTISConfig(_GRPOBaseConfig):
    """GRPOTIS configuration."""

    name: Literal["grpo_tis"] = "grpo_tis"
    tis_cap: float

    @model_validator(mode="after")
    def _validate_tis(self) -> "GRPOTISConfig":
        if self.tis_cap <= 0.0:
            raise ValueError("tis_cap must be > 0.0")
        return self


AlgorithmConfig = Annotated[
    GRPOConfig | GRPOTISConfig,
    Field(discriminator="name"),
]
