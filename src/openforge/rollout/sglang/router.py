# Copyright 2026 openforge

import time
from multiprocessing.process import BaseProcess

import requests
from loguru import logger
from sglang.srt.utils import kill_process_tree

from openforge.rollout.types import RouterSpec

from .utils import (
    generate_sglang_router_args,
    launch_sglang_router,
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
        if self.process is None:
            return

        if self.process.is_alive():
            try:
                kill_process_tree(self.process.pid, include_parent=True)
            except Exception:
                self.process.terminate()
            self.process.join(timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS)

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

    def is_ready(self) -> bool:
        """Return ``True`` when the router is ready to serve model traffic."""
        spec = self.spec
        if not self.is_healthy():
            return False
        if not spec.worker_urls:
            return True

        try:
            response = requests.get(
                f"{spec.url}/v1/models",
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return False

        if response.status_code != 200:
            return False

        try:
            payload = response.json()
        except ValueError:
            return False
        data = payload.get("data")
        return isinstance(data, list) and len(data) > 0

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
        """Block until the router is ready to serve traffic."""
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
            if self.is_ready():
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"rollout router {self.spec.router_name} did not become ready in time"
        )
