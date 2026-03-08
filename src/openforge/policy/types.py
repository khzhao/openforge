# Copyright 2026 openforge

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "DistributedUpdateSession",
    "LoadFormat",
    "PolicyArtifactRef",
    "TensorUpdateSession",
    "WeightBucketMeta",
]

LoadFormat: TypeAlias = Literal["flattened_bucket"]


@dataclass(slots=True)
class PolicyArtifactRef:
    """Reference to a train-exported policy artifact on shared storage."""

    step: int
    policy_version: int
    path: str


@dataclass(slots=True)
class WeightBucketMeta:
    """Metadata describing one named-tensor bucket."""

    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]


@dataclass(slots=True)
class TensorUpdateSession:
    """Controller-coordinated session for direct tensor bucket updates."""

    session_id: str
    policy_version: int
    load_format: LoadFormat
    engine_ids: list[int]
    transport_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DistributedUpdateSession:
    """Controller-coordinated session for direct distributed updates."""

    session_id: str
    policy_version: int
    load_format: LoadFormat
    engine_ids: list[int]
    master_addr: str
    master_port: int
    group_name: str
    world_size: int
    backend: str
    rank_offsets: dict[int, int]
    transport_metadata: dict[str, Any] = field(default_factory=dict)
