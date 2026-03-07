# Copyright 2026 openforge

from typing import Any

__all__ = [
    "TrainBackend",
    "FSDP2Backend",
    "SGLangEngineRuntime",
    "SGLangEngineSpec",
    "SGLangServerClient",
]


def __getattr__(name: str) -> Any:
    if name == "TrainBackend":
        from .abcs import TrainBackend

        return TrainBackend
    if name == "FSDP2Backend":
        from .fsdp2 import FSDP2Backend

        return FSDP2Backend
    if name in {"SGLangEngineRuntime", "SGLangEngineSpec", "SGLangServerClient"}:
        from .sglang import (
            SGLangEngineRuntime,
            SGLangEngineSpec,
            SGLangServerClient,
        )

        return {
            "SGLangEngineRuntime": SGLangEngineRuntime,
            "SGLangEngineSpec": SGLangEngineSpec,
            "SGLangServerClient": SGLangServerClient,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
