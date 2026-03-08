# Copyright 2026 openforge

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from tensordict import TensorDict

from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.train.types import CheckpointInfo, TrainStepResult, TrainWorkerSpec

__all__ = ["TrainBackend"]


class TrainBackend(ABC):
    """Abstract rank-local backend interface for train workers."""

    @abstractmethod
    def initialize(self, spec: TrainWorkerSpec) -> None:
        """Initialize distributed state, models, optimizer, and runtime resources."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients before a new update or accumulation cycle."""

    @abstractmethod
    def forward(self, batch: TensorDict) -> TensorDict:
        """Run forward pass and return backend-defined outputs."""

    @abstractmethod
    def backward(self, forward_out: TensorDict) -> None:
        """Run backward pass from forward outputs."""

    @abstractmethod
    def no_sync(self) -> AbstractContextManager[None]:
        """Context manager to disable synchronization during accumulation."""

    @abstractmethod
    def step_optimizer(self, *, global_step: int | None = None) -> TrainStepResult:
        """Apply optimizer step and return per-rank step metadata."""

    @abstractmethod
    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> CheckpointInfo:
        """Save a checkpoint and return metadata for the created artifact."""

    @abstractmethod
    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> CheckpointInfo | None:
        """Load a checkpoint and return metadata for the checkpoint that was loaded."""

    @abstractmethod
    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef | None:
        """Publish a train checkpoint as an artifact reference for rollout loading."""

    @abstractmethod
    def push_tensor_update(
        self,
        session: TensorUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        """Push a direct tensor update into an already-open rollout session."""

    @abstractmethod
    def push_distributed_update(
        self,
        session: DistributedUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        """Push a direct distributed update into an already-open rollout session."""

    @abstractmethod
    def sleep(self) -> None:
        """Move runtime state to a low-memory idle state."""

    @abstractmethod
    def wakeup(self) -> None:
        """Restore runtime state from idle to active training."""

    @abstractmethod
    def clear_memory(self) -> None:
        """Release temporary allocator state and other non-persistent memory."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down backend runtime resources."""
