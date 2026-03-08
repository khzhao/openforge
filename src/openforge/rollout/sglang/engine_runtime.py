# Copyright 2026 openforge

import multiprocessing
import os
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any

from .client import SGLangControlClient
from .spec import SGLangEngineSpec

HEALTHCHECK_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0


def _serve_sglang(server_args_payload: dict[str, Any]) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    server_args = ServerArgs(**server_args_payload)
    if getattr(server_args, "host", None) is not None:
        server_args.host = str(server_args.host).strip("[]")

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def _launch_sglang_process(server_args_payload: dict[str, Any]) -> BaseProcess:
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_serve_sglang, args=(server_args_payload,))
    process.start()
    return process


def _kill_process_tree(pid: int) -> bool:
    try:
        from sglang.srt.utils import kill_process_tree
    except Exception:
        return False
    kill_process_tree(pid, include_parent=True)
    return True


class SGLangEngineRuntime:
    """Managed lifecycle for one in-worker SGLang HTTP server."""

    def __init__(
        self,
        spec: SGLangEngineSpec,
        *,
        request_timeout_seconds: float = 5.0,
        process_launcher: Callable[
            [dict[str, Any]], BaseProcess
        ] = _launch_sglang_process,
    ) -> None:
        self.spec = spec
        self.request_timeout_seconds = request_timeout_seconds
        self._process_launcher = process_launcher
        self.process: BaseProcess | None = None
        self.client = SGLangControlClient(spec.url)

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.process = self._process_launcher(dict(self.spec.server_args))
        self._wait_until_ready()

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            if not _kill_process_tree(process.pid):
                process.terminate()
            process.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        self.process = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def is_healthy(self) -> bool:
        process = self.process
        if process is None or not process.is_alive():
            return False
        return self.client.health_generate(timeout=self.request_timeout_seconds)

    def get_server_info(self) -> dict[str, Any]:
        return self.client.get_server_info(timeout=self.request_timeout_seconds)

    def get_weight_version(self) -> str | None:
        return self.client.get_weight_version(timeout=self.request_timeout_seconds)

    def check_weights(self, *, action: str) -> dict[str, Any]:
        return self.client.check_weights(
            action=action,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def flush_cache(self) -> bool:
        return self.client.flush_cache(timeout=self.request_timeout_seconds)

    def pause_generation(self, *, mode: str = "abort") -> Any:
        return self.client.pause_generation(
            mode=mode,
            timeout=self.request_timeout_seconds,
        )

    def continue_generation(self) -> Any:
        return self.client.continue_generation(timeout=self.request_timeout_seconds)

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> dict[str, Any]:
        return self.client.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def destroy_weights_update_group(self, *, group_name: str) -> dict[str, Any]:
        return self.client.destroy_weights_update_group(
            group_name=group_name,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
        load_format: str | None = None,
    ) -> dict[str, Any]:
        policy_version = self._require_policy_version(weight_version)
        payload = self.client.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            load_format=load_format,
            timeout=max(self.request_timeout_seconds, 30.0),
        )
        self.spec.policy_version = policy_version
        return payload

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        policy_version = self._require_policy_version(weight_version)
        payload = self.client.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            timeout=max(self.request_timeout_seconds, 30.0),
        )
        self.spec.policy_version = policy_version
        return payload

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.name} exited before becoming healthy"
                )
            if self.client.health_generate(timeout=self.request_timeout_seconds):
                self._wait_for_flush_cache()
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.name} did not become healthy in time"
        )

    def _wait_for_flush_cache(self) -> None:
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"rollout engine {self.spec.name} exited before cache flush"
                )
            if self.client.flush_cache(timeout=self.request_timeout_seconds):
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self.spec.name} never acknowledged flush_cache"
        )

    @staticmethod
    def _require_policy_version(weight_version: str | None) -> int:
        if weight_version is None:
            raise ValueError("weight_version is required for SGLang weight updates")
        try:
            return int(weight_version)
        except ValueError as exc:
            raise ValueError(
                f"weight_version must be an integer string, got {weight_version!r}"
            ) from exc
