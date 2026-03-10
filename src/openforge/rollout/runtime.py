# Copyright 2026 openforge

from abc import ABC, abstractmethod
from typing import Any

from openforge.configs.rollout import RolloutEndpoint
from openforge.policy.types import PolicyArtifactRef

__all__ = ["RolloutRuntime"]


class RolloutRuntime(ABC):
    """Abstract engine-local rollout runtime interface."""

    @abstractmethod
    def start(self) -> None:
        """Start the engine-local runtime."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the engine-local runtime."""

    @abstractmethod
    def restart(
        self,
        *,
        model_path: str | None = None,
        policy_version: int | None = None,
    ) -> None:
        """Restart the runtime with optional updated model metadata."""

    @abstractmethod
    def endpoint(self) -> RolloutEndpoint:
        """Return the current endpoint view for this runtime."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return whether the engine is healthy."""

    @abstractmethod
    def pause_generation(self, *, mode: str = "abort") -> None:
        """Pause in-flight generation."""

    @abstractmethod
    def continue_generation(self) -> None:
        """Resume generation after a pause."""

    @abstractmethod
    def flush_cache(self) -> bool:
        """Flush any engine-side cache state."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from the engine."""

    @abstractmethod
    def get_server_info(self) -> dict[str, Any]:
        """Fetch server metadata from the engine."""

    @abstractmethod
    def get_weight_version(self) -> str | None:
        """Fetch the currently loaded weight version."""

    @abstractmethod
    def load_policy_artifact(self, artifact: PolicyArtifactRef) -> None:
        """Load or restart from an exported policy artifact."""
