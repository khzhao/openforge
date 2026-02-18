# Copyright 2026 openforge

from collections.abc import Sequence
from typing import Any

from openforge.configs import OpenForgeConfig


class RolloutWorkerGroup:
    """Coordinates rollout rank workers."""

    def __init__(self, config: OpenForgeConfig, workers: list[Any]) -> None:
        if not workers:
            raise ValueError("workers must not be empty")
        self.config = config
        self.workers = workers

    def generate(
        self, requests: Sequence[dict[str, Any]], is_prefill: bool = False
    ) -> list[dict[str, Any]]:
        """Dispatch generation across rollout rank workers."""
        raise NotImplementedError
