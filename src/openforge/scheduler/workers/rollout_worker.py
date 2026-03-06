# Copyright 2026 openforge

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import ray

from openforge.configs import (
    ExportedPolicy,
    OpenForgeConfig,
    ResolvedRolloutEngine,
    RolloutEndpoint,
)
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

HEALTHCHECK_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_POLL_INTERVAL_SECONDS = 1.0
PROCESS_TERMINATION_TIMEOUT_SECONDS = 30.0


class RolloutWorker:
    """Ray-friendly worker that owns a single SGLang engine process."""

    def initialize(
        self,
        cfg: OpenForgeConfig,
        *,
        engine: ResolvedRolloutEngine,
    ) -> RolloutEndpoint:
        self.cfg = cfg
        self.engine = engine
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self.host = self._resolve_node_ip_address()
        self.port: int | None = None
        self.bootstrap_port: int | None = None
        self.model_path: str | None = None
        self.policy_version: int | None = None
        self._set_initial_model_path()

        if engine.role == "placeholder":
            return self.endpoint()

        self.port = self._allocate_port()
        if self.cfg.rollout.engine_topology == "pd" and engine.role == "prefill":
            self.bootstrap_port = self._allocate_port(start=self.port + 1)

        self._launch_process()
        return self.endpoint()

    def endpoint(self) -> RolloutEndpoint:
        url = None
        if self.port is not None:
            url = f"http://{self.host}:{self.port}"

        return RolloutEndpoint(
            name=self._engine_name(),
            role=self.engine.role,
            host=self.host,
            port=self.port,
            bootstrap_port=self.bootstrap_port,
            url=url,
            healthy=self.is_healthy(),
            policy_version=self.policy_version,
            model_path=self.model_path,
        )

    def load_weights_from_policy(self, policy: ExportedPolicy) -> RolloutEndpoint:
        """Restart the engine from a serving artifact exported by training."""
        self.model_path = policy.model_path
        self.policy_version = policy.policy_version

        if self.engine.role == "placeholder":
            return self.endpoint()

        self._restart_process()
        return self.endpoint()

    def load_latest_weights_from_train(self) -> RolloutEndpoint:
        """Load the most recent training export if one exists."""
        latest = self.cfg.train.read_exported_policy(latest=True)
        if latest is None:
            raise FileNotFoundError(
                "no rollout policy export found under train.checkpoints_dir"
            )
        return self.load_weights_from_policy(latest)

    def shutdown(self) -> None:
        self._stop_process()

    def is_healthy(self) -> bool:
        if self.engine.role == "placeholder":
            return False
        if self.process is None or self.process.poll() is not None:
            return False
        assert self.port is not None
        return self._check_health(self.port)

    def _engine_name(self) -> str:
        return f"{self.engine.group_name}-{self.engine.replica_index}"

    def _set_initial_model_path(self) -> None:
        latest = self.cfg.train.read_exported_policy(latest=True)
        if latest is None:
            self.model_path = self.cfg.model.model_name_or_path
            self.policy_version = None
            return

        self.model_path = latest.model_path
        self.policy_version = latest.policy_version

    def _launch_process(self) -> None:
        command = self._build_command()
        env = self._build_env()
        log_path = self._log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_healthy()

    def _restart_process(self) -> None:
        self._stop_process()
        self._launch_process()

    def _stop_process(self) -> None:
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

    def _wait_until_healthy(self) -> None:
        assert self.port is not None
        deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"rollout engine {self._engine_name()} exited before becoming healthy"
                )
            if self._check_health(self.port):
                return
            time.sleep(HEALTHCHECK_POLL_INTERVAL_SECONDS)
        raise TimeoutError(
            f"rollout engine {self._engine_name()} did not become healthy in time"
        )

    def _build_command(self) -> list[str]:
        assert self.model_path is not None
        assert self.port is not None

        parallelism = self.engine.parallelism
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tokenizer-path",
            self.cfg.model.tokenizer_name_or_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--data-parallel-size",
            str(parallelism.data_parallel_size),
            "--pipeline-parallel-size",
            str(parallelism.pipeline_parallel_size),
            "--tensor-parallel-size",
            str(parallelism.tensor_parallel_size),
            "--context-parallel-size",
            str(parallelism.context_parallel_size),
            "--expert-parallel-size",
            str(parallelism.expert_parallel_size),
        ]

        if self.engine.role in {"prefill", "decode"}:
            command.extend(["--disaggregation-mode", self.engine.role])

        if self.engine.role == "prefill":
            decode_parallelism = self._role_parallelism("decode")
            assert self.bootstrap_port is not None
            command.extend(
                [
                    "--bootstrap-port",
                    str(self.bootstrap_port),
                    "--disaggregation-decode-tp",
                    str(decode_parallelism.tensor_parallel_size),
                    "--disaggregation-decode-dp",
                    str(decode_parallelism.data_parallel_size),
                ]
            )

        if self.engine.role == "decode":
            prefill_parallelism = self._role_parallelism("prefill")
            command.extend(
                [
                    "--disaggregation-prefill-pp",
                    str(prefill_parallelism.pipeline_parallel_size),
                ]
            )

        return command

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if ray.is_initialized():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(int(gpu_id)) for gpu_id in gpu_ids
                )
        return env

    def _log_path(self) -> Path:
        return (
            Path(self.cfg.train.checkpoints_dir)
            / "rollout_logs"
            / f"{self._engine_name()}.log"
        )

    def _allocate_port(self, start: int | None = None) -> int:
        if start is None:
            start = 30000 + (self.engine.engine_id * 10)
        return get_free_port(start=start)

    def _role_parallelism(self, role: str):
        for group in self.cfg.rollout.engines:
            if group.role == role:
                return group.parallelism
        raise ValueError(f"Missing rollout role {role} required for pd topology")

    def _resolve_node_ip_address(self) -> str:
        if ray.is_initialized():
            return get_current_ray_node_ip_address()
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def _check_health(port: int) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        try:
            with urlopen(url, timeout=1.0) as response:
                return response.status == 200
        except (OSError, URLError):
            return False


class PDRouterWorker:
    """CPU-only worker that owns the SGLang PD router process."""

    def initialize(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
        checkpoints_dir: str,
    ) -> RolloutEndpoint:
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self.host = self._resolve_node_ip_address()
        self.port = get_free_port(start=38000)
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints = decode_endpoints
        self.checkpoints_dir = checkpoints_dir

        self._launch_process()
        return self.endpoint()

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

    def restart(
        self,
        *,
        prefill_endpoints: list[RolloutEndpoint],
        decode_endpoints: list[RolloutEndpoint],
    ) -> RolloutEndpoint:
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints = decode_endpoints
        self._stop_process()
        self._launch_process()
        return self.endpoint()

    def shutdown(self) -> None:
        self._stop_process()

    def is_healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        return RolloutWorker._check_health(self.port)

    def _launch_process(self) -> None:
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

    def _stop_process(self) -> None:
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

    def _resolve_node_ip_address(self) -> str:
        if ray.is_initialized():
            return get_current_ray_node_ip_address()
        return socket.gethostbyname(socket.gethostname())
