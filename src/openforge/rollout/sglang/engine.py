# Copyright 2026 openforge

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from loguru import logger

from openforge.rollout.sglang.types import EngineAddr, EngineSpec
from openforge.utils.networking import get_free_port
from openforge.utils.processes import stop_spawned_process
from openforge.utils.ray import get_current_ray_node_ip_address

from .client import SGLangClient
from .utils import (
    generate_sglang_server_args,
    launch_sglang_process,
)

__all__ = ["Engine"]


class Engine:
    """Managed lifecycle for one SGLang HTTP server."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    WEIGHT_UPDATE_TIMEOUT_SECONDS = 300.0
    HEALTHCHECK_TIMEOUT_SECONDS = 300.0
    HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
    PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0

    def initialize(self, spec: EngineSpec) -> None:
        self.spec = spec
        self._runtime_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_runtime_call: Future | None = None

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
        return self.addr.url

    def launch(self, addr: EngineAddr) -> None:
        self.addr = addr
        self.client = SGLangClient(self.url)
        logger.info(f"Launching SGLang server on {addr}")
        self.process = launch_sglang_process(
            generate_sglang_server_args(engine_spec=self.spec, engine_addr=addr)
        )
        self._wait_until_ready()

    def stop(self) -> None:
        try:
            stop_spawned_process(
                self.process,
                timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS,
            )
        finally:
            self.process = None
            self._runtime_executor.shutdown(wait=False, cancel_futures=True)

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
        sampling_params: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.client.generate(
            sampling_params=sampling_params,
            timeout=self.REQUEST_TIMEOUT_SECONDS if timeout is None else timeout,
            **kwargs,
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

    def node_ip_address(self) -> str:
        return get_current_ray_node_ip_address()

    def distributed_world_size(self) -> int:
        return self.spec.parallelism.world_size

    def get_weight_version(self) -> str | None:
        return self.client.get_weight_version(timeout=self.REQUEST_TIMEOUT_SECONDS)

    def update_weights_from_disk(
        self,
        *,
        model_path: str,
        policy_version: int | None = None,
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> dict[str, Any]:
        return self.client.update_weights_from_disk(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
            weight_version="0" if policy_version is None else str(policy_version),
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, 30.0),
        )

    def begin_init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        self._submit_runtime_call(
            lambda: self.client.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
                timeout=self.WEIGHT_UPDATE_TIMEOUT_SECONDS,
            )
        )

    def begin_update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        policy_version: int | None = None,
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> None:
        self._submit_runtime_call(
            lambda: self.client.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                load_format=load_format,
                flush_cache=flush_cache,
                weight_version="0" if policy_version is None else str(policy_version),
                timeout=self.WEIGHT_UPDATE_TIMEOUT_SECONDS,
            )
        )

    def wait_pending_runtime_call(self) -> dict[str, Any] | None:
        if self._pending_runtime_call is None:
            return None
        future = self._pending_runtime_call
        self._pending_runtime_call = None
        return future.result()

    def destroy_weights_update_group(self, *, group_name: str) -> dict[str, Any]:
        return self.client.destroy_weights_update_group(
            group_name=group_name,
            timeout=self.WEIGHT_UPDATE_TIMEOUT_SECONDS,
        )

    def check_weights(self, *, action: str) -> dict[str, Any]:
        return self.client.check_weights(
            action=action,
            timeout=max(self.REQUEST_TIMEOUT_SECONDS, 30.0),
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.engine_name} exited before becoming healthy"
                )
            if self.client.health_generate(timeout=self.REQUEST_TIMEOUT_SECONDS):
                self._wait_for_flush_cache()
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.engine_name} did not become healthy in time"
        )

    def _wait_for_flush_cache(self) -> None:
        deadline = time.monotonic() + self.HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.engine_name} exited before cache flush"
                )
            if self.client.flush_cache(timeout=self.REQUEST_TIMEOUT_SECONDS):
                return
            time.sleep(self.HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.engine_name} never acknowledged flush_cache"
        )

    def _submit_runtime_call(self, fn) -> None:
        if self._pending_runtime_call is not None:
            raise RuntimeError("previous runtime call is still pending")
        self._pending_runtime_call = self._runtime_executor.submit(fn)
