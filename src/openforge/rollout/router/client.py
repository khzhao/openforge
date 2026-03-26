# Copyright 2026 openforge

from __future__ import annotations

import time
from multiprocessing.process import BaseProcess
from typing import Any, Sequence

from openforge.rollout.router.types import RouterSpec
from openforge.rollout.router.utils import launch_openforge_router
from openforge.rollout.sglang.client import SGLangClient
from openforge.rollout.sglang.types import EngineAddr, EngineSpec
from openforge.utils.processes import stop_spawned_process

__all__ = ["RolloutRouter", "RolloutRouterClient"]


class RolloutRouterClient(SGLangClient):
    """HTTP client for the OpenForge rollout router."""

    @property
    def url(self) -> str:
        return self.base_url

    def generate(
        self,
        sampling_params: dict[str, Any],
        *,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Route one generate request through the router."""
        response = self._request(
            "POST",
            "/generate",
            payload={
                **kwargs,
                "sampling_params": sampling_params,
                "stream": False,
            },
            timeout=timeout,
        )
        body = self._decode_body(response.text)
        if not isinstance(body, (dict, list)):
            raise RuntimeError("router /generate did not return JSON")
        return body

    def health(self, *, timeout: float = 1.0) -> bool:
        return self._ok("GET", "/health", timeout=timeout)

    def status(self, *, timeout: float = 5.0) -> dict[str, Any]:
        return self._get_json("/status", timeout=timeout)

    def register_train_server(
        self,
        *,
        train_server_url: str,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        return self._post_json(
            "/register_train_server",
            payload={"train_server_url": str(train_server_url)},
            timeout=timeout,
        )

    def receive_policy_version(
        self,
        *,
        policy_version: int,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        return self._post_json(
            "/receive_policy_version",
            payload={"policy_version": int(policy_version)},
            timeout=timeout,
        )

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


class RolloutRouter(RolloutRouterClient):
    """Lightweight client wrapper for the rollout router service."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    POLL_INTERVAL_SECONDS = 0.1

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self._launched_router: BaseProcess | None = None
        self.spec: RouterSpec | None = None

    def initialize(
        self,
        *,
        spec: RouterSpec,
        engine_specs: Sequence[EngineSpec],
        engine_addrs: dict[str, EngineAddr],
    ) -> None:
        """Store metadata and launch the router server."""
        self.base_url = spec.url.rstrip("/")
        self.spec = spec
        worker_metadata = self._build_worker_metadata(
            engine_specs=engine_specs,
            engine_addrs=engine_addrs,
        )
        if self._launched_router is not None:
            raise RuntimeError(f"rollout router {spec.router_name} is already running")
        self._launched_router = launch_openforge_router(
            spec,
            worker_metadata=worker_metadata,
        )
        self.wait_until_ready(
            startup_timeout_secs=float(spec.worker_startup_timeout_secs),
            request_timeout_secs=float(spec.health_check_timeout_secs),
        )

    def shutdown(self) -> None:
        """Stop the background router server."""
        spec = self.spec
        launched_router = self._launched_router
        if launched_router is not None:
            stop_spawned_process(
                launched_router,
                timeout=(
                    self.REQUEST_TIMEOUT_SECONDS
                    if spec is None
                    else float(spec.request_timeout_secs)
                ),
            )
            if launched_router.is_alive():
                raise RuntimeError(
                    "rollout router did not shut down cleanly"
                    if spec is None
                    else f"rollout router {spec.router_name} did not shut down cleanly"
                )
        self._launched_router = None

    def is_healthy(self) -> bool:
        """Return whether the router health endpoint is responding."""
        return self.health(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def wait_until_ready(
        self,
        *,
        startup_timeout_secs: float,
        request_timeout_secs: float,
    ) -> None:
        """Wait until the router can serve generation traffic."""
        deadline = time.monotonic() + max(
            self.REQUEST_TIMEOUT_SECONDS,
            float(startup_timeout_secs),
        )
        request_timeout = max(self.REQUEST_TIMEOUT_SECONDS, request_timeout_secs)
        while time.monotonic() < deadline:
            if self.health_generate(timeout=request_timeout):
                return
            time.sleep(self.POLL_INTERVAL_SECONDS)
        raise TimeoutError("rollout router did not become ready in time")

    @staticmethod
    def _build_worker_metadata(
        *,
        engine_specs: Sequence[EngineSpec],
        engine_addrs: dict[str, EngineAddr],
    ) -> list[dict[str, Any]]:
        return [
            {
                "worker_name": engine_spec.engine_name,
                "worker_url": engine_addrs[engine_spec.engine_name].url,
                "world_size": engine_spec.parallelism.world_size,
            }
            for engine_spec in engine_specs
        ]
