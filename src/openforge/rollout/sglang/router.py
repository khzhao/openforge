# Copyright 2026 openforge

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import ray
import requests

from openforge.configs.rollout import RolloutEndpoint
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

__all__ = ["PDRouterRuntime", "PDRouterWorker"]

HEALTHCHECK_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0


class PDRouterRuntime:
    """Lifecycle manager for one SGLang PD router process."""

    def __init__(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
        checkpoints_dir: str,
    ) -> None:
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints = decode_endpoints
        self.checkpoints_dir = checkpoints_dir
        self.host = _resolve_node_ip_address()
        self.port = get_free_port(start=38000)
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def start(self) -> None:
        command = self._build_command()
        log_path = Path(self.checkpoints_dir) / "rollout_logs" / "pd-router.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        self.process = subprocess.Popen(
            command,
            env=os.environ.copy(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_healthy()

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)

        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def restart(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
    ) -> None:
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints = decode_endpoints
        self.stop()
        self.start()

    def endpoint(self) -> RolloutEndpoint:
        return RolloutEndpoint(
            name="pd-router",
            role="router",
            host=self.host,
            port=self.port,
            bootstrap_port=None,
            url=f"http://{self.host}:{self.port}",
            healthy=self.is_healthy(),
            policy_version=None,
            model_path=None,
        )

    def is_healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        return _check_health(self.port)

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("pd router exited before becoming healthy")
            if self.is_healthy():
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError("pd router did not become healthy in time")

    def _build_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--pd-disaggregation",
            "--prefill",
        ]
        for endpoint in self.prefill_endpoints:
            assert endpoint.url is not None
            assert endpoint.bootstrap_port is not None
            command.extend([endpoint.url, str(endpoint.bootstrap_port)])

        command.append("--decode")
        for endpoint in self.decode_endpoints:
            assert endpoint.url is not None
            command.append(endpoint.url)

        return command


class PDRouterWorker:
    """Ray-friendly wrapper over one PD router runtime."""

    def initialize(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
        checkpoints_dir: str,
    ) -> RolloutEndpoint:
        self.runtime = PDRouterRuntime(
            prefill_endpoints=prefill_endpoints,
            decode_endpoints=decode_endpoints,
            checkpoints_dir=checkpoints_dir,
        )
        self.runtime.start()
        return self.runtime.endpoint()

    def endpoint(self) -> RolloutEndpoint:
        return self._runtime().endpoint()

    def restart(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
    ) -> RolloutEndpoint:
        runtime = self._runtime()
        runtime.restart(
            prefill_endpoints=prefill_endpoints,
            decode_endpoints=decode_endpoints,
        )
        return runtime.endpoint()

    def shutdown(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.stop()
            self.runtime = None

    def is_healthy(self) -> bool:
        runtime = getattr(self, "runtime", None)
        return runtime is not None and runtime.is_healthy()

    def _runtime(self) -> PDRouterRuntime:
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            raise RuntimeError("pd router has not been initialized")
        return runtime


def _resolve_node_ip_address() -> str:
    if ray.is_initialized():
        return get_current_ray_node_ip_address()
    return socket.gethostbyname(socket.gethostname())


def _check_health(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    try:
        response = requests.get(url, timeout=1.0)
        return response.status_code == 200
    except requests.RequestException:
        return False
