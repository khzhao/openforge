# Copyright 2026 openforge

import multiprocessing
import os
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from openforge.policy.types import PolicyArtifactRef

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

    def is_healthy(self) -> bool:
        process = self.process
        if process is None or not process.is_alive():
            return False
        return self.client.health_generate(timeout=self.request_timeout_seconds)

    def get_server_info(self) -> dict[str, Any]:
        return self.client.get_server_info(timeout=self.request_timeout_seconds)

    def get_model_info(self) -> dict[str, Any]:
        return self.client.get_model_info(timeout=self.request_timeout_seconds)

    def get_weight_version(self) -> str | None:
        return self.client.get_weight_version(timeout=self.request_timeout_seconds)

    def generate(
        self,
        *,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.client.generate(
            payload=payload,
            timeout=self.request_timeout_seconds if timeout is None else timeout,
        )

    def update_weights_from_disk(
        self,
        *,
        model_path: str,
        load_format: str | None = None,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        payload = self.client.update_weights_from_disk(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=weight_version,
            timeout=max(self.request_timeout_seconds, 30.0),
        )
        self.spec.model_path = model_path
        self.spec.server_args["model_path"] = model_path
        self.spec.server_args["served_model_name"] = Path(model_path).name
        if weight_version is not None:
            try:
                self.spec.policy_version = int(weight_version)
                self.spec.server_args["weight_version"] = weight_version
            except ValueError:
                pass
        return payload

    def load_policy_artifact(
        self,
        artifact: PolicyArtifactRef,
        *,
        flush_cache: bool = True,
        abort_all_requests: bool = False,
    ) -> dict[str, Any]:
        artifact_path = Path(artifact.path)
        if not artifact_path.is_dir():
            raise FileNotFoundError(
                f"policy artifact path does not exist or is not a directory: {artifact.path}"
            )

        payload = self.update_weights_from_disk(
            model_path=str(artifact_path),
            load_format=artifact.load_format,
            flush_cache=flush_cache,
            abort_all_requests=abort_all_requests,
            weight_version=str(artifact.policy_version),
        )
        model_info = self.get_model_info()
        loaded_weight_version = model_info.get("weight_version")
        if loaded_weight_version is not None and (
            str(loaded_weight_version) != str(artifact.policy_version)
        ):
            raise RuntimeError(
                "sglang reported weight_version "
                f"{loaded_weight_version!r} after loading artifact version "
                f"{artifact.policy_version!r}"
            )
        return payload

    def check_weights(self, *, action: str) -> dict[str, Any]:
        return self.client.check_weights(
            action=action,
            timeout=max(self.request_timeout_seconds, 30.0),
        )

    def flush_cache(self) -> bool:
        return self.client.flush_cache(timeout=self.request_timeout_seconds)

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
