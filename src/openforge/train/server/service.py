# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

if TYPE_CHECKING:
    from openforge.train.manager import TrainManager

__all__ = [
    "create_train_app",
    "launch_train_http_server",
    "start_train_http_server",
    "stop_train_http_server",
    "wait_until_train_http_server_ready",
]


class _ThreadedUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return


class _WeightSyncWorker(BaseModel):
    worker_url: str
    world_size: int


class _WeightSyncRequest(BaseModel):
    workers: list[_WeightSyncWorker]
    target_version: int
    mode: Literal["distributed", "disk"] = "distributed"


def create_train_app(manager: TrainManager) -> FastAPI:
    """Create the train HTTP control-plane app."""
    app = FastAPI(title="OpenForge Train Server")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/request_weight_sync")
    async def respond_to_weight_sync_request(
        payload: _WeightSyncRequest,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            manager.respond_to_weight_sync_request,
            workers=[
                (worker.worker_url, worker.world_size) for worker in payload.workers
            ],
            target_version=payload.target_version,
            mode=payload.mode,
        )

    return app


def launch_train_http_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    log_level: str | None,
    name: str,
) -> tuple[uvicorn.Server, threading.Thread, str]:
    """Launch the train HTTP server in a background thread."""
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
    return server, thread, f"http://{host}:{port}"


def start_train_http_server(
    manager: TrainManager,
    *,
    host: str,
    port: int,
    log_level: str | None,
    name: str,
    request_timeout: float,
    poll_interval: float,
) -> tuple[uvicorn.Server, threading.Thread, str]:
    """Create, launch, and wait for the train HTTP server."""
    server, thread, url = launch_train_http_server(
        create_train_app(manager),
        host=host,
        port=port,
        log_level=log_level,
        name=name,
    )
    try:
        wait_until_train_http_server_ready(
            url=url,
            server=server,
            thread=thread,
            request_timeout=request_timeout,
            poll_interval=poll_interval,
        )
    except Exception:
        stop_train_http_server(
            server,
            thread,
            timeout=request_timeout,
            name=name,
        )
        raise
    return server, thread, url


def stop_train_http_server(
    server: uvicorn.Server,
    thread: threading.Thread,
    *,
    timeout: float,
    name: str,
) -> None:
    """Stop the train HTTP server."""
    server.should_exit = True
    thread.join(timeout=timeout)
    assert not thread.is_alive(), f"train server {name} did not shut down cleanly"


def wait_until_train_http_server_ready(
    *,
    url: str,
    server: uvicorn.Server,
    thread: threading.Thread,
    request_timeout: float,
    poll_interval: float,
) -> None:
    """Wait until the train HTTP server starts serving requests."""
    deadline = time.monotonic() + request_timeout
    while time.monotonic() < deadline:
        assert thread.is_alive(), "train server exited before becoming ready"
        try:
            response = requests.get(
                f"{url}/health",
                timeout=request_timeout,
            )
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    raise TimeoutError("train server did not become ready in time")
