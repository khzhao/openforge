# Copyright 2026 openforge

from contextlib import nullcontext
from dataclasses import replace
from typing import Sequence

from tensordict import TensorDict

from openforge.train.fsdp2.base import FSDP2Engine
from openforge.train.types import TrainStepResult, TrainWorkerSpec, TrainWorkerState

__all__ = ["TrainWorker"]


class TrainWorker:
    """Thin Ray-facing wrapper over one FSDP2 engine."""

    def initialize(self, spec: TrainWorkerSpec) -> TrainWorkerState:
        backend_name = spec.cfg.train.backend
        if backend_name == "megatron":
            raise NotImplementedError("Megatron train workers are not implemented yet")
        if backend_name != "fsdp2":
            raise ValueError(f"Unsupported backend: {backend_name}")

        self.spec = spec
        self.engine = FSDP2Engine(spec)
        self.state = TrainWorkerState(
            rank=spec.rank,
            world_size=spec.world_size,
            backend=backend_name,
            device=str(self.engine.device),
            initialized=True,
            sleeping=False,
        )
        return self.state

    def step(
        self,
        microbatches: Sequence[TensorDict],
        *,
        global_step: int | None = None,
    ) -> TrainStepResult:
        if not microbatches:
            raise ValueError("step requires at least one microbatch")

        self.engine.zero_grad()
        last_index = len(microbatches) - 1
        for index, batch in enumerate(microbatches):
            context = self.engine.no_sync() if index < last_index else nullcontext()
            with context:
                outputs = self.engine.forward(batch)
                self.engine.backward(outputs)

        metrics = self.engine.step_optimizer()
        return TrainStepResult(
            rank=self.spec.rank,
            global_step=global_step,
            metrics={
                **metrics,
                "global_step": -1.0 if global_step is None else float(global_step),
            },
        )

    def sleep(self) -> None:
        self.engine.sleep()
        self.state = replace(self.state, sleeping=True)

    def wakeup(self) -> None:
        self.engine.wakeup()
        self.state = replace(self.state, sleeping=False)

    def status(self) -> TrainWorkerState:
        return self.state

    def shutdown(self) -> None:
        self.engine.shutdown()
        self.state = replace(self.state, initialized=False)
