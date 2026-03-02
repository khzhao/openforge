# Copyright 2026 openforge

from enum import StrEnum


class Role(StrEnum):
    """Role of a worker in the system."""

    ActorRefCritic = "actor_ref_critic"
    Rollout = "rollout"
    Actor = "actor"
    Ref = "ref"
    Critic = "critic"


__all__ = ["Role"]
