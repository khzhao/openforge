# Copyright 2026 openforge

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import ray
import requests

from openforge.configs import (
    DistributedPolicyWeightBucket,
    OpenForgeConfig,
    ResolvedRolloutEngine,
    RolloutEndpoint,
    SerializedPolicyWeights,
)
from openforge.engines.sglang import SGLangEngineRuntime, SGLangEngineSpec
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

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
        self.runtime: SGLangEngineRuntime | None = None
        self.host = self._resolve_node_ip_address()
        self.port: int | None = None
        self.bootstrap_port: int | None = None
        self.model_path: str = self.cfg.model.model_name_or_path
        self.policy_version: int | None = None
        self._active_update_group_name: str | None = None
        self._paused_for_weight_update = False

        if engine.role == "placeholder":
            return self.endpoint()

        self.port = self._allocate_port()
        if self.cfg.rollout.engine_topology == "pd" and engine.role == "prefill":
            self.bootstrap_port = self._allocate_port(start=self.port + 1)

        spec = self._build_runtime_spec()
        self.runtime = SGLangEngineRuntime(spec)
        self.runtime.start()
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

    def update_weights_from_train(
        self,
        weights: SerializedPolicyWeights,
    ) -> RolloutEndpoint:
        """Update the live engine weights using serialized SGLang tensor buckets."""
        self.policy_version = weights.policy_version

        if self.engine.role == "placeholder":
            return self.endpoint()

        assert self.runtime is not None
        paused = False
        try:
            self.runtime.pause_generation(mode="abort")
            paused = True
            if not self.runtime.flush_cache():
                raise RuntimeError("sglang engine did not flush cache before update")

            tp_size = self.engine.parallelism.tensor_parallel_size
            for serialized_bucket in weights.serialized_weight_buckets:
                # SGLang expects one payload per TP rank. We serialize full HF
                # weights, and each TP rank loads/shards the same bucket locally.
                self.runtime.update_weights_from_tensor(
                    serialized_named_tensors=[serialized_bucket] * tp_size,
                    load_format=weights.load_format,
                    flush_cache=False,
                    weight_version=str(weights.policy_version),
                )
        finally:
            if paused:
                self.runtime.continue_generation()
        return self.endpoint()

    def begin_tensor_weight_update(self, *, policy_version: int) -> None:
        if self.engine.role == "placeholder":
            return

        assert self.runtime is not None
        self.runtime.pause_generation(mode="abort")
        self._paused_for_weight_update = True
        try:
            if not self.runtime.flush_cache():
                raise RuntimeError("sglang engine did not flush cache before update")
            self.policy_version = policy_version
        except Exception:
            if self._paused_for_weight_update:
                self.runtime.continue_generation()
                self._paused_for_weight_update = False
            raise

    def apply_tensor_weight_bucket(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str,
        policy_version: int,
    ) -> None:
        if self.engine.role == "placeholder":
            return

        assert self.runtime is not None
        self.runtime.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=False,
            weight_version=str(policy_version),
        )
        self.policy_version = policy_version

    def finish_tensor_weight_update(self) -> RolloutEndpoint:
        if self.engine.role == "placeholder":
            return self.endpoint()

        assert self.runtime is not None
        try:
            if not self.runtime.flush_cache():
                raise RuntimeError("sglang engine did not flush cache after update")
        finally:
            if self._paused_for_weight_update:
                self.runtime.continue_generation()
                self._paused_for_weight_update = False
        return self.endpoint()

    def shutdown(self) -> None:
        runtime = self.runtime
        if runtime is not None:
            if self._active_update_group_name is not None:
                runtime.destroy_weights_update_group(
                    group_name=self._active_update_group_name
                )
                self._active_update_group_name = None
            if self._paused_for_weight_update:
                runtime.continue_generation()
                self._paused_for_weight_update = False
            runtime.stop()
            self.runtime = None

    def begin_distributed_weight_update(
        self,
        *,
        master_addr: str,
        master_port: int,
        rank_offset: int = 1,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        if self.engine.role == "placeholder":
            return

        assert self.runtime is not None
        self.runtime.pause_generation(mode="abort")
        self._paused_for_weight_update = True
        try:
            if not self.runtime.flush_cache():
                raise RuntimeError("sglang engine did not flush cache before update")
            self.runtime.init_weights_update_group(
                master_address=master_addr,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )
            self._active_update_group_name = group_name
        except Exception:
            if self._paused_for_weight_update:
                self.runtime.continue_generation()
                self._paused_for_weight_update = False
            raise

    def apply_distributed_weight_bucket(
        self,
        *,
        bucket: DistributedPolicyWeightBucket,
        policy_version: int,
        load_format: str,
        group_name: str,
    ) -> None:
        if self.engine.role == "placeholder":
            return

        assert self.runtime is not None
        self.runtime.update_weights_from_distributed(
            names=bucket.names,
            dtypes=bucket.dtypes,
            shapes=bucket.shapes,
            group_name=group_name,
            flush_cache=False,
            weight_version=str(policy_version),
            load_format=load_format,
        )
        self.policy_version = policy_version

    def finish_distributed_weight_update(self, *, group_name: str) -> RolloutEndpoint:
        if self.engine.role == "placeholder":
            return self.endpoint()

        assert self.runtime is not None
        try:
            self.runtime.destroy_weights_update_group(group_name=group_name)
        finally:
            self._active_update_group_name = None
            if self._paused_for_weight_update:
                self.runtime.continue_generation()
                self._paused_for_weight_update = False
        return self.endpoint()

    def is_healthy(self) -> bool:
        if self.engine.role == "placeholder":
            return False
        runtime = self.runtime
        if runtime is None:
            return False
        return runtime.is_healthy()

    def _engine_name(self) -> str:
        return f"{self.engine.group_name}-{self.engine.replica_index}"

    def _build_runtime_spec(self) -> SGLangEngineSpec:
        assert self.port is not None

        parallelism = self.engine.parallelism
        server_args: dict[str, object] = {
            "model_path": self.model_path,
            "tokenizer_path": self.cfg.model.tokenizer_name_or_path,
            "host": "0.0.0.0",
            "port": self.port,
            "trust_remote_code": True,
            "skip_server_warmup": True,
            "served_model_name": self.cfg.model.model_name_or_path,
            "weight_version": (
                "default" if self.policy_version is None else str(self.policy_version)
            ),
            "dp_size": parallelism.data_parallel_size,
            "pp_size": parallelism.pipeline_parallel_size,
            "tp_size": parallelism.tensor_parallel_size,
            "attn_cp_size": parallelism.context_parallel_size,
            "ep_size": parallelism.expert_parallel_size,
        }

        if self.engine.role in {"prefill", "decode"}:
            server_args["disaggregation_mode"] = self.engine.role

        if self.engine.role == "prefill":
            decode_parallelism = self._role_parallelism("decode")
            assert self.bootstrap_port is not None
            server_args["disaggregation_bootstrap_port"] = self.bootstrap_port
            server_args["disaggregation_decode_tp"] = (
                decode_parallelism.tensor_parallel_size
            )
            server_args["disaggregation_decode_dp"] = (
                decode_parallelism.data_parallel_size
            )

        if self.engine.role == "decode":
            prefill_parallelism = self._role_parallelism("prefill")
            server_args["disaggregation_prefill_pp"] = (
                prefill_parallelism.pipeline_parallel_size
            )

        return SGLangEngineSpec(
            engine_id=self.engine.engine_id,
            name=self._engine_name(),
            role=self.engine.role,
            host=self.host,
            port=self.port,
            bootstrap_port=self.bootstrap_port,
            model_path=self.model_path,
            tokenizer_path=self.cfg.model.tokenizer_name_or_path,
            policy_version=self.policy_version,
            server_args=server_args,
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
            response = requests.get(url, timeout=1.0)
            return response.status_code == 200
        except requests.RequestException:
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
