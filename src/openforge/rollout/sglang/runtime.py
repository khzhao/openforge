# Copyright 2026 openforge

from pathlib import Path
from typing import Any

from openforge.configs.rollout import RolloutEndpoint
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
    WeightBucketMeta,
)
from openforge.rollout.runtime import RolloutRuntime

from .engine_runtime import SGLangEngineRuntime
from .spec import SGLangEngineSpec

__all__ = ["SGLangRuntime"]


class SGLangRuntime(RolloutRuntime):
    """RolloutRuntime implementation backed by the in-process SGLang runtime."""

    def __init__(self, spec: SGLangEngineSpec) -> None:
        self.spec = spec
        self._runtime = SGLangEngineRuntime(spec)
        self._active_update_group_name: str | None = None
        self._active_session_id: str | None = None
        self._paused_for_weight_update = False

    def start(self) -> None:
        self._runtime.start()

    def stop(self) -> None:
        self.abort_update(session_id=self._active_session_id or "")
        self._runtime.stop()

    def restart(
        self,
        *,
        model_path: str | None = None,
        policy_version: int | None = None,
    ) -> None:
        if model_path is not None:
            self.spec.model_path = model_path
            self.spec.server_args["model_path"] = model_path
            self.spec.server_args["served_model_name"] = Path(model_path).name
        if policy_version is not None:
            self.spec.policy_version = policy_version
            self.spec.server_args["weight_version"] = str(policy_version)
        self._runtime.restart()

    def endpoint(self) -> RolloutEndpoint:
        return RolloutEndpoint(
            name=self.spec.name,
            role=self.spec.role,
            host=self.spec.host,
            port=self.spec.port,
            bootstrap_port=self.spec.bootstrap_port,
            url=self.spec.url,
            healthy=self.is_healthy(),
            policy_version=self.spec.policy_version,
            model_path=self.spec.model_path,
        )

    def is_healthy(self) -> bool:
        return self._runtime.is_healthy()

    def pause_generation(self, *, mode: str = "abort") -> None:
        self._runtime.pause_generation(mode=mode)

    def continue_generation(self) -> None:
        self._runtime.continue_generation()

    def flush_cache(self) -> bool:
        return self._runtime.flush_cache()

    def get_model_info(self) -> dict[str, Any]:
        return self._runtime.client.get_model_info(
            timeout=self._runtime.request_timeout_seconds
        )

    def get_server_info(self) -> dict[str, Any]:
        return self._runtime.get_server_info()

    def get_weight_version(self) -> str | None:
        return self._runtime.get_weight_version()

    def load_policy_artifact(self, artifact: PolicyArtifactRef) -> None:
        self.restart(
            model_path=artifact.path,
            policy_version=artifact.policy_version,
        )

    def begin_tensor_update(self, session: TensorUpdateSession) -> None:
        self.pause_generation(mode="abort")
        self._paused_for_weight_update = True
        try:
            if not self.flush_cache():
                raise RuntimeError("sglang engine did not flush cache before update")
            self.spec.policy_version = session.policy_version
            self._active_session_id = session.session_id
        except Exception:
            self.abort_update(session_id=session.session_id)
            raise

    def apply_tensor_bucket(
        self,
        *,
        serialized_named_tensors: list[str],
        load_format: str,
        policy_version: int,
    ) -> None:
        self._runtime.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=False,
            weight_version=str(policy_version),
        )
        self.spec.policy_version = policy_version

    def finish_tensor_update(self, session: TensorUpdateSession) -> None:
        try:
            if not self.flush_cache():
                raise RuntimeError("sglang engine did not flush cache after update")
            self.spec.policy_version = session.policy_version
        finally:
            if self._paused_for_weight_update:
                self.continue_generation()
                self._paused_for_weight_update = False
            self._active_session_id = None

    def begin_distributed_update(self, session: DistributedUpdateSession) -> None:
        self.pause_generation(mode="abort")
        self._paused_for_weight_update = True
        try:
            if not self.flush_cache():
                raise RuntimeError("sglang engine did not flush cache before update")
            self._runtime.init_weights_update_group(
                master_address=session.master_addr,
                master_port=session.master_port,
                rank_offset=session.rank_offsets[self.spec.engine_id],
                world_size=session.world_size,
                group_name=session.group_name,
                backend=session.backend,
            )
            self._active_update_group_name = session.group_name
            self._active_session_id = session.session_id
        except Exception:
            self.abort_update(session_id=session.session_id)
            raise

    def apply_distributed_bucket(
        self,
        *,
        bucket: WeightBucketMeta,
        load_format: str,
        policy_version: int,
        group_name: str,
    ) -> None:
        self._runtime.update_weights_from_distributed(
            names=bucket.names,
            dtypes=bucket.dtypes,
            shapes=bucket.shapes,
            group_name=group_name,
            flush_cache=False,
            weight_version=str(policy_version),
            load_format=load_format,
        )
        self.spec.policy_version = policy_version

    def finish_distributed_update(self, session: DistributedUpdateSession) -> None:
        try:
            if self._active_update_group_name is not None:
                self._runtime.destroy_weights_update_group(
                    group_name=self._active_update_group_name
                )
            self.spec.policy_version = session.policy_version
        finally:
            self._active_update_group_name = None
            if self._paused_for_weight_update:
                self.continue_generation()
                self._paused_for_weight_update = False
            self._active_session_id = None

    def abort_update(self, *, session_id: str) -> None:
        if (
            self._active_session_id is not None
            and session_id
            and self._active_session_id != session_id
        ):
            return
        if self._active_update_group_name is not None:
            try:
                self._runtime.destroy_weights_update_group(
                    group_name=self._active_update_group_name
                )
            except Exception:
                pass
            self._active_update_group_name = None
        if self._paused_for_weight_update:
            try:
                self.continue_generation()
            except Exception:
                pass
            self._paused_for_weight_update = False
        self._active_session_id = None
