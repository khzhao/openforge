# Copyright 2026 openforge

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from openforge.gateway.types import RuntimeConfig


class _GatewayState(BaseModel):
    host: str
    port: int
    pid: int


class _SessionState(BaseModel):
    session_id: str
    runtime: RuntimeConfig


class _ActiveState(BaseModel):
    version: Literal[1] = 1
    gateway: _GatewayState
    session: _SessionState | None = None


def active_state_path() -> Path:
    """Return the machine-local shared state path."""
    cache_root = Path(os.environ.get("OPENFORGE_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "openforge" / "active_gateway.json"


def load_active_state() -> _ActiveState:
    """Load the current shared gateway/session state."""
    path = active_state_path()
    assert path.exists(), (
        "no active gateway recorded; run `uv run openforge gateway start --config ...` first"
    )
    return _ActiveState.model_validate_json(path.read_text(encoding="utf-8"))


def save_active_gateway(*, host: str, port: int, pid: int) -> None:
    """Record the active gateway target."""
    _write_state(
        _ActiveState(
            gateway=_GatewayState(host=host, port=port, pid=pid),
        )
    )


def clear_active_gateway(expected_pid: int | None = None) -> None:
    """Clear the active gateway target."""
    path = active_state_path()
    if not path.exists():
        return
    if expected_pid is not None and load_active_state().gateway.pid != expected_pid:
        return
    path.unlink()


def load_active_gateway_target() -> tuple[str, int]:
    """Return the active gateway host and port."""
    gateway = load_active_state().gateway
    return gateway.host, gateway.port


def load_active_gateway_pid() -> int:
    """Return the active gateway process id."""
    return load_active_state().gateway.pid


def save_active_session(*, session_id: str, runtime: RuntimeConfig) -> None:
    """Record the active session and its runtime config."""
    state = load_active_state()
    _write_state(
        _ActiveState(
            gateway=state.gateway,
            session=_SessionState(session_id=session_id, runtime=runtime),
        )
    )


def clear_active_session() -> None:
    """Clear the active session while keeping the gateway target."""
    path = active_state_path()
    if not path.exists():
        return
    state = load_active_state()
    _write_state(_ActiveState(gateway=state.gateway))


def load_active_runtime_config() -> RuntimeConfig:
    """Return the runtime config for the active session."""
    session = load_active_state().session
    assert session is not None, (
        "no active session recorded; run `uv run openforge session start --runtime-config ...` first"
    )
    return session.runtime


def _write_state(state: _ActiveState) -> None:
    path = active_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(state.model_dump_json(), encoding="utf-8")
    tmp_path.replace(path)
