# Copyright 2026 openforge

import time
from typing import Any

from loguru import logger

from openforge.rollout.types import EngineAddr, EngineSpec
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

from .client import SGLangClient
from .utils import (
    generate_sglang_server_args,
    kill_sglang_process_tree,
    launch_sglang_process,
)

__all__ = ["Engine"]


class Engine:
    """Managed lifecycle for one SGLang HTTP server."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    HEALTHCHECK_TIMEOUT_SECONDS = 300.0
    HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
    PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0

    def initialize(self, spec: EngineSpec) -> None:
        self.spec = spec

    def get_ip_addr(self) -> str:
        return get_current_ray_node_ip_address()

    def get_free_port(self, start: int, block_size: int) -> int:
        return get_free_port(start=start, block_size=block_size)

    def get_post_launch_addr(self) -> EngineAddr:
        return self.addr

    def get_post_init_spec(self) -> EngineSpec:
        return self.spec

    @property
    def url(self) -> str:
        return f"http://{self.addr.host}:{self.addr.port}"

    def launch(self, addr: EngineAddr) -> None:
        self.addr = addr
        self.client = SGLangClient(self.url)
        logger.info(f"Launching SGLang server on {addr}")
        self.process = launch_sglang_process(
            generate_sglang_server_args(engine_spec=self.spec, engine_addr=addr)
        )
        self._wait_until_ready()

    def stop(self) -> None:
        if self.process.is_alive():
            if not kill_sglang_process_tree(self.process.pid):
                self.process.terminate()
            self.process.join(timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS)
        self.process = None

    def is_healthy(self) -> bool:
        return self.process.is_alive() and self.client.health_generate(
            timeout=self.REQUEST_TIMEOUT_SECONDS
        )

    def get_server_info(self) -> dict[str, Any]:
        return self.client.get_server_info(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def get_model_info(self) -> dict[str, Any]:
        return self.client.get_model_info(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def generate(
        self,
        *,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.client.generate(
            payload=payload,
            timeout=self.REQUEST_TIMEOUT_SECONDS if timeout is None else timeout,
        )

    def pause_generation(self, *, mode: str = "abort") -> dict[str, Any]:
        return self.client.pause_generation(
            mode=mode,
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, 30.0),
        )

    def continue_generation(self) -> dict[str, Any]:
        return self.client.continue_generation(
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, 30.0),
        )

    def flush_cache(self) -> bool:
        return self.client.flush_cache(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.name} exited before becoming healthy"
                )
            if self.client.health_generate(timeout=self.REQUEST_TIMEOUT_SECONDS):
                self._wait_for_flush_cache()
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.name} did not become healthy in time"
        )

    def _wait_for_flush_cache(self) -> None:
        deadline = time.monotonic() + self.HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.name} exited before cache flush"
                )
            if self.client.flush_cache(timeout=self.REQUEST_TIMEOUT_SECONDS):
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.name} never acknowledged flush_cache"
        )
