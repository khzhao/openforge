# Copyright 2026 openforge

from __future__ import annotations

from tensordict import TensorDict

from openforge.configs import (
    DistributedPolicyWeights,
    OpenForgeConfig,
    SerializedPolicyWeights,
)
from openforge.engines.fsdp2 import FSDP2Backend
from openforge.utils.networking import get_free_port


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

    def export_policy_weights_for_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> SerializedPolicyWeights | None:
        return self.backend.export_policy_weights_for_rollout(
            step=step,
            policy_version=policy_version,
        )

    def prepare_policy_weights_for_distributed_rollout(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> DistributedPolicyWeights | None:
        return self.backend.prepare_policy_weights_for_distributed_rollout(
            step=step,
            policy_version=policy_version,
        )

    def init_policy_weights_update_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        self.backend.init_policy_weights_update_group(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

    def broadcast_prepared_policy_weights_bucket(
        self,
        *,
        bucket_index: int,
        group_name: str,
    ) -> None:
        self.backend.broadcast_prepared_policy_weights_bucket(
            bucket_index=bucket_index,
            group_name=group_name,
        )

    def destroy_policy_weights_update_group(self, *, group_name: str) -> None:
        self.backend.destroy_policy_weights_update_group(group_name=group_name)

    def clear_prepared_policy_weights_for_rollout(self) -> None:
        self.backend.clear_prepared_policy_weights_for_rollout()

    def allocate_free_port(self, *, start: int = 40000) -> int:
        return get_free_port(start=start)

    def sync_policy_weights_to_rollout(
        self,
        *,
        rollout_workers: list[object],
        rollout_engines: list[object],
        policy_version: int,
        sync_mode: str,
    ) -> None:
        self.backend.sync_policy_weights_to_rollout(
            rollout_workers=rollout_workers,
            rollout_engines=rollout_engines,
            policy_version=policy_version,
            sync_mode=sync_mode,
        )

    def sleep(self) -> None:
        self.backend.sleep()

    def wakeup(self) -> None:
        self.backend.wakeup()

    def clear_memory(self) -> None:
        self.backend.clear_memory()

    def shutdown(self) -> None:
        self.backend.shutdown()
