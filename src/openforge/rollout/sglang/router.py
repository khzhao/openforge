# Copyright 2026 openforge

import time
from multiprocessing.process import BaseProcess
from typing import Any

import requests
from loguru import logger

from openforge.rollout.types import RouterSpec

from .utils import (
    generate_sglang_router_args,
    launch_sglang_router,
    stop_spawned_process,
)

__all__ = ["Router"]


class Router:
    """Managed lifecycle for one SGLang router process."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    HEALTHCHECK_POLL_INTERVAL_SECONDS = 0.5
    PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0

    def __init__(self) -> None:
        self.process: BaseProcess | None = None

    def initialize(self, spec: RouterSpec) -> None:
        """Store the launch contract for this router."""
        self.spec = spec

    @property
    def url(self) -> str:
        return self.spec.url

    def launch(self) -> None:
        """Launch the router process and wait until it is healthy."""
        if self.process is not None and self.process.is_alive():
            raise RuntimeError(f"router {self.spec.router_name} is already running")

        logger.info("Launching SGLang router on {}", self.spec.url)
        self.process = launch_sglang_router(generate_sglang_router_args(self.spec))
        self.wait_until_ready()

    def shutdown(self) -> None:
        """Terminate the router process tree."""
        stop_spawned_process(
            self.process,
            timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS,
        )
        self.process = None

    def is_healthy(self) -> bool:
        """Return ``True`` when the router process is alive and responding on /health."""
        spec = self.spec
        if spec is None or self.process is None or not self.process.is_alive():
            return False

        try:
            response = requests.get(
                spec.health_url,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def _can_route_generate(self) -> bool:
        """Return ``True`` when the router can proxy a minimal generate request."""
        spec = self.spec
        if not spec.worker_urls:
            return True

        payload = {
            "text": "Router readiness probe.",
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": 1,
            },
            "stream": False,
        }
        try:
            response = requests.post(
                f"{spec.url}/generate",
                json=payload,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def generate(
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send one non-streaming generate request through the router."""
        response = requests.post(
            f"{self.url}/generate",
            json={
                **kwargs,
                "sampling_params": sampling_params,
                "stream": False,
            },
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, self.spec.request_timeout_secs),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"sglang router generate failed with status "
                f"{response.status_code}: {response.text!r}"
            )
        return response.json()

    def add_worker(self, worker_url: str) -> None:
        """Register a worker URL before the router is launched."""
        if self.process is not None and self.process.is_alive():
            raise RuntimeError(
                "dynamic worker registration is not implemented for the managed SGLang router"
            )
        if worker_url not in self.spec.worker_urls:
            self.spec.worker_urls.append(worker_url)

    def list_workers(self) -> list[str]:
        """Return the currently configured worker URLs."""
        return list(self.spec.worker_urls)

    def wait_until_ready(self) -> None:
        """Block until the router is healthy and can proxy generate traffic."""
        if self.process is None:
            raise RuntimeError(f"router {self.spec.router_name} has not been launched")

        startup_timeout = max(
            self.REQUEST_TIMEOUT_SECONDS,
            self.spec.worker_startup_timeout_secs,
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"rollout router {self.spec.router_name} exited before becoming ready"
                )
            if self.is_healthy() and self._can_route_generate():
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"rollout router {self.spec.router_name} did not become ready in time"
        )
