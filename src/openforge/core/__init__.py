# Copyright 2026 openforge

from enum import StrEnum


class Role(StrEnum):
    """Role of a worker in the system."""

    ActorRef = "actor_ref"
    Rollout = "rollout"
    Actor = "actor"
    Ref = "ref"


__all__ = ["Role"]
