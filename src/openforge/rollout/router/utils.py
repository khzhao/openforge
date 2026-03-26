# Copyright 2026 openforge

from __future__ import annotations

import threading
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI

__all__ = [
    "LaunchedRouterServer",
    "launch_openforge_router",
    "stop_openforge_router",
]


class _ThreadedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return


@dataclass(slots=True)
class LaunchedRouterServer:
    """Started router server and thread."""

    server: uvicorn.Server
    thread: threading.Thread


def launch_openforge_router(
    app: FastAPI,
    *,
    host: str,
    port: int,
    log_level: str | None,
    name: str,
) -> LaunchedRouterServer:
    """Start the rollout router server."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level or "warning",
        access_log=False,
        lifespan="off",
    )
    server = _ThreadedUvicornServer(config)
    thread = threading.Thread(
        target=server.run,
        name=name,
        daemon=True,
    )
    thread.start()
    return LaunchedRouterServer(server=server, thread=thread)


def stop_openforge_router(
    launched: LaunchedRouterServer | None,
    *,
    timeout: float,
    name: str,
) -> None:
    """Stop the rollout router server."""
    if launched is None:
        return
    launched.server.should_exit = True
    launched.thread.join(timeout=timeout)
    if launched.thread.is_alive():
        raise RuntimeError(f"rollout router {name} did not shut down cleanly")
