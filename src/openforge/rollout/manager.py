# Copyright 2026 openforge

from __future__ import annotations

from typing import Any

from ray.util.placement_group import PlacementGroup

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.router import RolloutRouter
from openforge.rollout.router.types import RouterSpec
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.sglang.engine_group import (
    EngineGroup,
    allocate_engine_addrs,
    start_sglang_engines,
)
from openforge.rollout.sglang.types import EngineAddr, EngineSpec
from openforge.utils.networking import get_free_port, get_host_ip

__all__ = [
    "RolloutManager",
    "EngineGroup",
    "allocate_engine_addrs",
    "start_sglang_engines",
]


class RolloutManager:
    """Thin lifecycle owner for rollout engines and the custom router."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ) -> None:
        self.cfg = cfg
        self.placement_groups = placement_groups

    def initialize(
        self,
        *,
        router_ip: str | None = None,
        router_port: int | None = None,
        engine_addrs: dict[str, EngineAddr] | None = None,
        **router_kwargs: Any,
    ) -> None:
        self.engine_group = EngineGroup(self.cfg, self.placement_groups)
        self.engine_group.initialize(engine_addrs=engine_addrs)

        main_router_kwargs = {
            "router_name": "openforge-router",
            "router_ip": router_ip or get_host_ip(),
            "router_port": router_port or get_free_port(start=30000),
            "policy": router_kwargs.pop("policy", "round_robin"),
            "worker_urls": [
                addr.url for addr in self.engine_group.engine_addrs.values()
            ],
            "request_timeout_secs": router_kwargs.pop("request_timeout_secs", 300),
            "worker_startup_timeout_secs": router_kwargs.pop(
                "worker_startup_timeout_secs",
                300,
            ),
            "worker_startup_check_interval": router_kwargs.pop(
                "worker_startup_check_interval",
                1,
            ),
            "health_check_timeout_secs": router_kwargs.pop(
                "health_check_timeout_secs",
                5,
            ),
            "health_check_interval_secs": router_kwargs.pop(
                "health_check_interval_secs",
                5,
            ),
            "log_level": router_kwargs.pop("log_level", None),
        }
        main_router_kwargs.update(router_kwargs)
        self.router_spec = RouterSpec(**main_router_kwargs)
        self._router = RolloutRouter()
        self._router.initialize(
            spec=self.router_spec,
            engine_workers=self.engine_group.engine_workers,
            engine_specs=self.engine_group.engine_specs,
            engine_addrs=self.engine_group.engine_addrs,
        )
        try:
            self._router.launch()
        except Exception:
            self.engine_group.shutdown()
            self._router.shutdown()
            raise

    def shutdown(self) -> None:
        router = getattr(self, "_router", None)
        try:
            if router is not None:
                router.shutdown()
        finally:
            self.engine_group.shutdown()

    @property
    def router(self) -> RolloutRouter:
        return self._router

    @property
    def router_url(self) -> str:
        return self._router.url

    @property
    def engine_workers(self) -> list[Engine]:
        return self.engine_group.engine_workers

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        return self.engine_group.engine_addrs

    @property
    def engine_specs(self) -> list[EngineSpec]:
        return self.engine_group.engine_specs

    def worker_names(self) -> list[str]:
        return self.router.worker_names()

    def worker_for_name(self, worker_name: str) -> Engine:
        return self.router.worker_for_name(worker_name)

    def health_status(self) -> dict[str, bool]:
        return self.router.health_status()

    def current_policy_version(self) -> int:
        return self.router.current_policy_version()
