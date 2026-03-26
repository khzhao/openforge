# Copyright 2026 openforge

from __future__ import annotations

from typing import Any, Sequence

import requests

__all__ = ["TrainServerClient"]


class TrainServerClient:
    """HTTP client for the OpenForge train server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request_weight_sync(
        self,
        *,
        workers: Sequence[dict[str, object]],
        target_version: int,
        mode: str = "distributed",
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/request_weight_sync",
            json={
                "workers": list(workers),
                "target_version": int(target_version),
                "mode": str(mode),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(
                "train /request_weight_sync did not return a JSON object"
            )
        return payload
