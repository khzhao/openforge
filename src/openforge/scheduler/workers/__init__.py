# Copyright 2026 openforge

from __future__ import annotations

from typing import Any

__all__ = ["ActorRefWorker", "PDRouterWorker", "RolloutWorker"]


def __getattr__(name: str) -> Any:
    if name == "ActorRefWorker":
        from .actor_ref_worker import ActorRefWorker

        return ActorRefWorker
    if name in {"PDRouterWorker", "RolloutWorker"}:
        from .rollout_worker import PDRouterWorker, RolloutWorker

        return {
            "PDRouterWorker": PDRouterWorker,
            "RolloutWorker": RolloutWorker,
        }[name]
    raise AttributeError(name)
