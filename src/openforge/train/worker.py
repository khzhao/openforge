# Copyright 2026 openforge

from contextlib import nullcontext
from dataclasses import replace
from typing import Sequence

from tensordict import TensorDict

from openforge.policy.types import PolicyArtifactRef
from openforge.train.fsdp2.base import FSDP2Engine
from openforge.train.fsdp2.checkpoint import (
    load_backend_checkpoint,
    save_backend_checkpoint,
)
from openforge.train.fsdp2.export import (
    export_policy_artifact as export_fsdp2_policy_artifact,
)
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
            self.backend = FSDP2Engine(spec)
        else:
            raise ValueError(f"Unsupported backend: {spec.cfg.train.backend}")

        self._state = TrainWorkerState(
            rank=spec.rank,
            world_size=spec.world_size,
            backend=spec.cfg.train.backend,
            device=str(self.backend.device),
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

        metrics = backend.step_optimizer()
        if not isinstance(metrics, dict):
            raise TypeError(
                "FSDP2Engine.step_optimizer must return dict[str, float], "
                f"got {type(metrics).__name__}"
            )
        return TrainStepResult(
            rank=self.spec.rank,
            global_step=global_step,
            metrics={
                **metrics,
                "global_step": -1.0 if global_step is None else float(global_step),
            },
        )

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> CheckpointInfo:
        result = save_backend_checkpoint(
            self.backend,
            step=step,
            policy_version=policy_version,
            save_optimizer=save_optimizer,
        )
        if not isinstance(result, CheckpointInfo):
            raise TypeError(
                "save_backend_checkpoint must return CheckpointInfo, "
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
        result = load_backend_checkpoint(
            self.backend,
            latest=latest,
            step=step,
            load_optimizer=load_optimizer,
        )
        if result is None or isinstance(result, CheckpointInfo):
            return result
        raise TypeError(
            "load_backend_checkpoint must return CheckpointInfo | None, "
            f"got {type(result).__name__}"
        )

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef | None:
        artifact = export_fsdp2_policy_artifact(
            self.backend,
            step=step,
            policy_version=policy_version,
        )
        if artifact is not None and not isinstance(artifact, PolicyArtifactRef):
            raise TypeError(
                "export_policy_artifact helper must return "
                f"PolicyArtifactRef | None, got {type(artifact).__name__}"
            )
        return artifact

    def sleep(self) -> None:
        self.backend.sleep()
        self._state = replace(self._state, sleeping=True)

    def wakeup(self) -> None:
        self.backend.wakeup()
        self._state = replace(self._state, sleeping=False)

    def status(self) -> TrainWorkerState:
        return replace(self._state)

    def shutdown(self) -> None:
        self.backend.shutdown()
        self._state = replace(self._state, initialized=False)
