# Copyright 2026 openforge

from __future__ import annotations

from typing import Any

from ray.util.placement_group import PlacementGroup

from openforge.configs.models import OpenForgeConfig
from openforge.rollout.router import RolloutRouter
from openforge.rollout.router.types import RouterSpec
from openforge.rollout.sglang.engine import Engine
from openforge.rollout.sglang.engine_group import EngineGroup
from openforge.rollout.sglang.types import EngineAddr, EngineSpec
from openforge.utils.networking import get_free_port, get_host_ip

__all__ = ["RolloutManager"]


class RolloutManager:
    """Thin lifecycle owner for rollout engines and the custom router."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        placement_groups: dict[str, tuple[PlacementGroup, list[int], list[int]]],
    ) -> None:
        self.cfg = cfg
        self.placement_groups = placement_groups
        self.router: RolloutRouter | None = None
        self.router_spec: RouterSpec | None = None

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

        mainrouter_kwargs = {
            "router_name": "openforge-router",
            "router_ip": router_ip or get_host_ip(),
            "router_port": router_port or get_free_port(start=30000),
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
        mainrouter_kwargs.update(router_kwargs)
        self.router_spec = RouterSpec(**mainrouter_kwargs)
        self.router = RolloutRouter(self.router_spec.url)
        try:
            self.router.initialize(
                spec=self.router_spec,
                engine_specs=self.engine_group.engine_specs,
                engine_addrs=self.engine_group.engine_addrs,
            )
        except Exception:
            self.router.shutdown()
            self.engine_group.shutdown()
            raise

    def shutdown(self) -> None:
        try:
            if self.router is not None:
                self.router.shutdown()
        finally:
            self.engine_group.shutdown()

    @property
    def router(self) -> RolloutRouter:
        return self.router

    @property
    def router_spec(self) -> RouterSpec:
        return self.router_spec

    @property
    def router_url(self) -> str:
        return self.router.url

    @property
    def engine_workers(self) -> list[Engine]:
        return self.engine_group.engine_workers

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        return self.engine_group.engine_addrs

    @property
    def engine_specs(self) -> list[EngineSpec]:
        return self.engine_group.engine_specs

    def status(self) -> dict[str, Any]:
        return self.router.status()
