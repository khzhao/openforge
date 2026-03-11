# Copyright 2026 openforge

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from tensordict import TensorDict

from openforge.train.types import TrainStepResult, TrainWorkerSpec

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
    def sleep(self) -> None:
        """Move runtime state to a low-memory idle state."""

    @abstractmethod
    def wakeup(self) -> None:
        """Restore runtime state from idle to active training."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down backend runtime resources."""
