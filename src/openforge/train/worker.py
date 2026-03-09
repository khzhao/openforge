# Copyright 2026 openforge

from contextlib import nullcontext
from dataclasses import replace
from typing import Sequence

from tensordict import TensorDict

from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.train.fsdp2.backend import FSDP2Backend
from openforge.train.types import (
    CheckpointInfo,
    TrainStepResult,
    TrainWorkerSpec,
    TrainWorkerState,
)

__all__ = ["TrainWorker"]


class TrainWorker:
    """Rank-local Ray worker facade over one train backend instance."""

    def initialize(
        self,
        spec: TrainWorkerSpec,
    ) -> TrainWorkerState:
        self.spec = spec

        if spec.cfg.train.backend == "fsdp2":
            self.backend = FSDP2Backend()
        else:
            raise ValueError(f"Unsupported backend: {spec.cfg.train.backend}")

        self.backend.initialize(spec)
        device = str(getattr(self.backend, "device", "unknown"))
        self._state = TrainWorkerState(
            rank=spec.rank,
            world_size=spec.world_size,
            backend=spec.cfg.train.backend,
            device=device,
            initialized=True,
            sleeping=False,
        )
        return self.status()

    def step(
        self,
        microbatches: Sequence[TensorDict],
        *,
        global_step: int | None = None,
    ) -> TrainStepResult:
        if not microbatches:
            raise ValueError("step requires at least one microbatch")

        backend = self.backend
        backend.zero_grad()
        last_index = len(microbatches) - 1
        for index, batch in enumerate(microbatches):
            context = backend.no_sync() if index < last_index else nullcontext()
            with context:
                forward_out = backend.forward(batch)
                backend.backward(forward_out)

        result = backend.step_optimizer(global_step=global_step)
        if not isinstance(result, TrainStepResult):
            raise TypeError(
                "TrainBackend.step_optimizer must return TrainStepResult, "
                f"got {type(result).__name__}"
            )
        return result

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> CheckpointInfo:
        result = self.backend.save_checkpoint(
            step=step,
            policy_version=policy_version,
            save_optimizer=save_optimizer,
        )
        if not isinstance(result, CheckpointInfo):
            raise TypeError(
                "TrainBackend.save_checkpoint must return CheckpointInfo, "
                f"got {type(result).__name__}"
            )
        return result

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> CheckpointInfo | None:
        result = self.backend.load_checkpoint(
            latest=latest,
            step=step,
            load_optimizer=load_optimizer,
        )
        if result is None or isinstance(result, CheckpointInfo):
            return result
        raise TypeError(
            "TrainBackend.load_checkpoint must return CheckpointInfo | None, "
            f"got {type(result).__name__}"
        )

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef | None:
        artifact = self.backend.export_policy_artifact(
            step=step,
            policy_version=policy_version,
        )
        if artifact is not None and not isinstance(artifact, PolicyArtifactRef):
            raise TypeError(
                "TrainBackend.export_policy_artifact must return "
                f"PolicyArtifactRef | None, got {type(artifact).__name__}"
            )
        return artifact

    def push_tensor_update(
        self,
        session: TensorUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        self.backend.push_tensor_update(
            session,
            step=step,
            policy_version=policy_version,
        )

    def push_distributed_update(
        self,
        session: DistributedUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        self.backend.push_distributed_update(
            session,
            step=step,
            policy_version=policy_version,
        )

    def sleep(self) -> None:
        self.backend.sleep()
        self._state = replace(self._state, sleeping=True)

    def wakeup(self) -> None:
        self.backend.wakeup()
        self._state = replace(self._state, sleeping=False)

    def clear_memory(self) -> None:
        self.backend.clear_memory()

    def status(self) -> TrainWorkerState:
        return replace(self._state)

    def shutdown(self) -> None:
        self.backend.shutdown()
        self._state = replace(self._state, initialized=False)
