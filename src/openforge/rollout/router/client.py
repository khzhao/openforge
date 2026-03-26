# Copyright 2026 openforge

from __future__ import annotations

from typing import Any, Sequence

from openforge.rollout.sglang.client import SGLangClient

__all__ = ["RolloutRouterClient"]


class RolloutRouterClient(SGLangClient):
    """HTTP client for the OpenForge rollout router."""

    @property
    def url(self) -> str:
        return self.base_url

    def health(self, *, timeout: float = 1.0) -> bool:
        return self._ok("GET", "/health", timeout=timeout)

    def release_trajectories(
        self,
        *,
        trajectory_ids: Sequence[str],
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        return self._post_json(
            "/release_trajectories",
            payload={
                "trajectory_ids": [
                    str(trajectory_id) for trajectory_id in trajectory_ids
                ]
            },
            timeout=timeout,
        )
