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
from openforge.utils.nccl import apply_nccl_env_defaults
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
        self._router: RolloutRouter | None = None
        self._router_spec: RouterSpec | None = None
        self._engine_group: EngineGroup | None = None

    def initialize(
        self,
        *,
        router_ip: str | None = None,
        router_port: int | None = None,
        engine_addrs: dict[str, EngineAddr] | None = None,
        **router_kwargs: Any,
    ) -> None:
        apply_nccl_env_defaults()
        self._engine_group = EngineGroup(self.cfg, self.placement_groups)
        self._engine_group.initialize(engine_addrs=engine_addrs)

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
        self._router_spec = RouterSpec(**mainrouter_kwargs)
        self._router = RolloutRouter(self._router_spec.url)
        try:
            self._router.initialize(
                spec=self._router_spec,
                engine_specs=self._engine_group.engine_specs,
                engine_addrs=self._engine_group.engine_addrs,
            )
        except Exception:
            self._router.shutdown()
            self._engine_group.shutdown()
            raise

    def shutdown(self) -> None:
        try:
            if self._router is not None:
                self._router.shutdown()
        finally:
            if self._engine_group is not None:
                self._engine_group.shutdown()

    @property
    def router(self) -> RolloutRouter:
        assert self._router is not None, "rollout router has not been initialized"
        return self._router

    @property
    def router_spec(self) -> RouterSpec:
        assert self._router_spec is not None, "rollout router has not been initialized"
        return self._router_spec

    @property
    def router_url(self) -> str:
        return self.router.url

    @property
    def engine_workers(self) -> list[Engine]:
        assert self._engine_group is not None, "engine group has not been initialized"
        return self._engine_group.engine_workers

    @property
    def engine_addrs(self) -> dict[str, EngineAddr]:
        assert self._engine_group is not None, "engine group has not been initialized"
        return self._engine_group.engine_addrs

    @property
    def engine_specs(self) -> list[EngineSpec]:
        assert self._engine_group is not None, "engine group has not been initialized"
        return self._engine_group.engine_specs

    def status(self) -> dict[str, Any]:
        return self.router.status()
