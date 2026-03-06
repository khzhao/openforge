# Copyright 2026 openforge

from __future__ import annotations

from pydantic import model_validator

from .base import OpenForgeBaseModel


class AlgorithmConfig(OpenForgeBaseModel):
    """Training algorithm parameters."""

    kl_coef: float = 0.0

    @model_validator(mode="after")
    def _validate_algorithm(self) -> AlgorithmConfig:
        if self.kl_coef < 0.0:
            raise ValueError("kl_coef must be >= 0.0")
        return self
