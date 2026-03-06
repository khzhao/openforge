# Copyright 2026 openforge

from __future__ import annotations

from tensordict import TensorDict

from openforge.configs import OpenForgeConfig
from openforge.engines.fsdp2 import FSDP2Backend


class ActorRefWorker:
    """Colocated actor-ref worker (one per GPU).

    Not wrapped with ray.remote so it can be tested without Ray.
    ActorRefGroup wraps it as a remote actor at runtime.
    """

    def initialize(
        self,
        cfg: OpenForgeConfig,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self.cfg = cfg

        if cfg.train.backend == "fsdp2":
            self.backend = FSDP2Backend()
        else:
            raise ValueError(f"Unsupported backend: {cfg.train.backend}")

        self.backend.initialize(
            cfg,
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
        )

    def train_step(
        self,
        batch: TensorDict,
        *,
        global_step: int | None = None,
    ) -> dict[str, float]:
        backend = self.backend
        backend.zero_grad()
        forward_out = backend.forward(batch)
        backend.backward(forward_out)
        return backend.step_optimizer(global_step=global_step)

    def save_checkpoint(self, *, step: int, policy_version: int) -> str:
        return self.backend.save_checkpoint(step=step, policy_version=policy_version)

    def load_checkpoint(self) -> tuple[int, int] | None:
        return self.backend.load_checkpoint(latest=True)

    def sleep(self) -> None:
        self.backend.sleep()

    def wakeup(self) -> None:
        self.backend.wakeup()

    def clear_memory(self) -> None:
        self.backend.clear_memory()

    def shutdown(self) -> None:
        self.backend.shutdown()
