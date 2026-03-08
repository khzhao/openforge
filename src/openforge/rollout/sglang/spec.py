# Copyright 2026 openforge

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SGLangEngineSpec:
    """Concrete launch and control specification for one SGLang engine."""

    engine_id: int
    name: str
    role: str
    host: str
    port: int
    bootstrap_port: int | None
    model_path: str
    tokenizer_path: str
    policy_version: int | None
    server_args: dict[str, Any]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
