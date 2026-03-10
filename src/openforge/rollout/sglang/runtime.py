# Copyright 2026 openforge

from pathlib import Path
from typing import Any

from openforge.configs.rollout import RolloutEndpoint
from openforge.policy.types import PolicyArtifactRef
from openforge.rollout.runtime import RolloutRuntime

from .engine_runtime import SGLangEngineRuntime
from .spec import SGLangEngineSpec

__all__ = ["SGLangRuntime"]


class SGLangRuntime(RolloutRuntime):
    """RolloutRuntime implementation backed by the in-process SGLang runtime."""

    def __init__(self, spec: SGLangEngineSpec) -> None:
        self.spec = spec
        self._runtime = SGLangEngineRuntime(spec)

    def start(self) -> None:
        self._runtime.start()

    def stop(self) -> None:
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
        self._runtime.update_weights_from_disk(
            model_path=artifact.path,
            weight_version=str(artifact.policy_version),
        )
        self.spec.model_path = artifact.path
        self.spec.tokenizer_path = artifact.path
        self.spec.policy_version = artifact.policy_version
        self.spec.server_args["model_path"] = artifact.path
        self.spec.server_args["tokenizer_path"] = artifact.path
        self.spec.server_args["served_model_name"] = Path(artifact.path).name
        self.spec.server_args["weight_version"] = str(artifact.policy_version)
