# Copyright 2026 openforge

from dataclasses import dataclass

__all__ = [
    "PolicyArtifactRef",
]


@dataclass(slots=True)
class PolicyArtifactRef:
    """Reference to a train-exported policy artifact on shared storage."""

    step: int
    policy_version: int
    path: str
    load_format: str | None = "safetensors"
