# Copyright 2026 openforge

import socket
from typing import cast

import ray
import requests

from openforge.configs.rollout import RolloutEndpoint
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
    WeightBucketMeta,
)
from openforge.rollout.runtime import RolloutRuntime
from openforge.rollout.sglang.runtime import SGLangRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.rollout.types import RolloutWorkerSpec
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

__all__ = ["RolloutWorker"]


class RolloutWorker:
    """Ray-friendly worker that owns a single rollout runtime."""

    def initialize(self, spec: RolloutWorkerSpec) -> RolloutEndpoint:
        self.spec = spec
        self.engine = spec.engine
        self.host = spec.host
        self.port = spec.port
        self.bootstrap_port = spec.bootstrap_port
        self.model_path = spec.cfg.model.model_name_or_path
        self.policy_version = spec.policy_version
        self.runtime: RolloutRuntime | None = None

        runtime_spec = self._build_runtime_spec()
        self.runtime = SGLangRuntime(runtime_spec)
        self.runtime.start()
        self._sync_runtime_state()
        return self.endpoint()

    def endpoint(self) -> RolloutEndpoint:
        runtime = self._runtime()
        self._sync_runtime_state()
        return runtime.endpoint()

    def is_healthy(self) -> bool:
        return self._runtime().is_healthy()

    def pause_generation(self, *, mode: str = "abort") -> None:
        self._runtime().pause_generation(mode=mode)

    def continue_generation(self) -> None:
        self._runtime().continue_generation()

    def flush_cache(self) -> bool:
        return self._runtime().flush_cache()

    def get_model_info(self) -> dict[str, object]:
        return cast(dict[str, object], self._runtime().get_model_info())

    def get_server_info(self) -> dict[str, object]:
        return cast(dict[str, object], self._runtime().get_server_info())

    def get_weight_version(self) -> str | None:
        return self._runtime().get_weight_version()

    def load_policy_artifact(self, artifact: PolicyArtifactRef) -> RolloutEndpoint:
        self.model_path = artifact.path
        self.policy_version = artifact.policy_version
        self._runtime().load_policy_artifact(artifact)
        self._sync_runtime_state()
        return self.endpoint()

    def begin_tensor_update(self, session: TensorUpdateSession) -> None:
        self._runtime().begin_tensor_update(session)
        self.policy_version = session.policy_version

    def apply_tensor_bucket(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str,
        policy_version: int,
    ) -> None:
        self._runtime().apply_tensor_bucket(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            policy_version=policy_version,
        )
        self.policy_version = policy_version

    def finish_tensor_update(self, session: TensorUpdateSession) -> RolloutEndpoint:
        self._runtime().finish_tensor_update(session)
        self._sync_runtime_state()
        return self.endpoint()

    def begin_distributed_update(self, session: DistributedUpdateSession) -> None:
        self._runtime().begin_distributed_update(session)

    def apply_distributed_bucket(
        self,
        *,
        bucket: WeightBucketMeta,
        policy_version: int,
        load_format: str,
        group_name: str,
    ) -> None:
        self._runtime().apply_distributed_bucket(
            bucket=bucket,
            policy_version=policy_version,
            load_format=load_format,
            group_name=group_name,
        )
        self.policy_version = policy_version

    def finish_distributed_update(
        self,
        session: DistributedUpdateSession,
    ) -> RolloutEndpoint:
        self._runtime().finish_distributed_update(session)
        self.policy_version = session.policy_version
        self._sync_runtime_state()
        return self.endpoint()

    def abort_update(self, *, session_id: str) -> None:
        self._runtime().abort_update(session_id=session_id)

    def shutdown(self) -> None:
        runtime = self.runtime
        if runtime is not None:
            runtime.stop()
            self.runtime = None

    @staticmethod
    def resolve_node_ip_address() -> str:
        if ray.is_initialized():
            return get_current_ray_node_ip_address()
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def allocate_port(*, start: int) -> int:
        return get_free_port(start=start)

    @staticmethod
    def _check_health(port: int) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        try:
            response = requests.get(url, timeout=1.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _build_runtime_spec(self) -> SGLangEngineSpec:
        parallelism = self.engine.parallelism
        server_args: dict[str, object] = {
            "model_path": self.model_path,
            "tokenizer_path": self.spec.cfg.model.tokenizer_name_or_path,
            "host": "0.0.0.0",
            "port": self.port,
            "trust_remote_code": True,
            "skip_server_warmup": True,
            "served_model_name": self.spec.cfg.model.model_name_or_path,
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
            tokenizer_path=self.spec.cfg.model.tokenizer_name_or_path,
            policy_version=self.policy_version,
            server_args=server_args,
        )

    def _runtime(self) -> RolloutRuntime:
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("rollout runtime has not been initialized")
        return runtime

    def _sync_runtime_state(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        endpoint = runtime.endpoint()
        if endpoint.model_path is None:
            raise RuntimeError("rollout runtime endpoint did not report model_path")
        self.model_path = endpoint.model_path
        self.policy_version = endpoint.policy_version

    def _engine_name(self) -> str:
        return f"{self.engine.group_name}-{self.engine.replica_index}"

    def _role_parallelism(self, role: str):
        for group in self.spec.cfg.rollout.engines:
            if group.role == role:
                return group.parallelism
        raise ValueError(f"Missing rollout role {role} required for pd topology")
