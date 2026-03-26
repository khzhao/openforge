# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["RouterSpec"]


@dataclass(slots=True)
class RouterSpec:
    """Configuration parameters for the rollout router."""

    router_name: str
    router_ip: str
    router_port: int
    policy: Literal["cache_aware", "round_robin"]
    worker_urls: list[str]

    request_timeout_secs: int
    worker_startup_timeout_secs: int
    worker_startup_check_interval: int
    health_check_timeout_secs: int
    health_check_interval_secs: int
    log_level: str | None

    @property
    def url(self) -> str:
        return f"http://{self.router_ip}:{self.router_port}"

    @property
    def health_url(self) -> str:
        return f"{self.url}/health"
