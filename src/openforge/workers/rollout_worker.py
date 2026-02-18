# Copyright 2026 openforge

from collections.abc import Sequence
from typing import Any

from openforge.configs.openforge import OpenForgeConfig


class RolloutWorker:
    """Single worker class for rollout generation."""

    def __init__(self, config: OpenForgeConfig, rank: int = 0) -> None:
        self.config = config
        self.rank = rank

    def generate(
        self, requests: Sequence[dict[str, Any]], is_prefill: bool = False
    ) -> list[dict[str, Any]]:
        """Generate outputs for rollout requests."""
        raise NotImplementedError
