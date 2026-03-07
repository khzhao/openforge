# Copyright 2026 openforge

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from tensordict import TensorDict

from openforge.configs import (
    DistributedPolicyWeights,
    OpenForgeConfig,
    SerializedPolicyWeights,
)


class TrainBackend(ABC):
    """Abstract training backend interface shared by FSDP2/Megatron backends."""

    @abstractmethod
    def initialize(
        self,
        cfg: OpenForgeConfig,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        """Initialize distributed state, models, optimizer, and runtime resources."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients before a new update or accumulation cycle."""

    @abstractmethod
    def forward(self, batch: TensorDict) -> TensorDict:
        """Run forward pass and return backend-defined outputs.

        Contract:
        - returned TensorDict must contain a scalar tensor under key `loss`.
        - backend may include additional tensors/metrics for logging.
        """

    @abstractmethod
    def backward(self, forward_out: TensorDict) -> None:
        """Run backward pass from forward outputs."""

    @abstractmethod
    def no_sync(self) -> AbstractContextManager[None]:
        """Context manager to disable gradient synchronization for accumulation."""

    @abstractmethod
    def step_optimizer(self, *, global_step: int | None = None) -> dict[str, float]:
        """Apply optimizer/scheduler step and return scalar metrics."""

    @abstractmethod
    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> str:
        """Save checkpoint to cfg.train.checkpoints_dir."""

    @abstractmethod
    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> tuple[int, int] | None:
        """Load checkpoint from cfg.train.checkpoints_dir.

        If `latest=True`, backend should load the most recent checkpoint.
        If `step` is provided, backend should load that explicit checkpoint.
        Returns `(step, policy_version)` if a checkpoint is found, otherwise `None`.
        """

    @abstractmethod
    def export_policy_weights_for_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> SerializedPolicyWeights | None:
        """Export a live SGLang weight-update payload for rollout consumption."""

    @abstractmethod
    def prepare_policy_weights_for_distributed_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> DistributedPolicyWeights | None:
        """Prepare metadata and cached buckets for SGLang distributed sync."""

    @abstractmethod
    def init_policy_weights_update_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        """Initialize the train-side source rank of a rollout weight-update group."""

    @abstractmethod
    def broadcast_prepared_policy_weights_bucket(
        self,
        *,
        bucket_index: int,
        group_name: str,
    ) -> None:
        """Broadcast one previously prepared weight bucket to rollout consumers."""

    @abstractmethod
    def destroy_policy_weights_update_group(self, *, group_name: str) -> None:
        """Destroy a previously initialized rollout weight-update group."""

    @abstractmethod
    def clear_prepared_policy_weights_for_rollout(self) -> None:
        """Release any cached rollout sync buckets after a sync completes."""

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
