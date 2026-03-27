# Copyright 2026 openforge

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import uvicorn

from openforge.data import OpenForgeStore
from openforge.train.loop import TrainLoop
from openforge.train.server import start_train_http_server, stop_train_http_server
from openforge.utils.networking import get_free_port, get_host_ip

if TYPE_CHECKING:
    from openforge.train.manager import TrainManager

__all__ = ["TrainRuntime"]


class TrainRuntime:
    """Train-side owner for the manager, server, and background train loop."""

    REQUEST_TIMEOUT_SECONDS = 5.0
    POLL_INTERVAL_SECONDS = 0.1
    SHUTDOWN_TIMEOUT_SECONDS = 30.0

    def __init__(self, manager: TrainManager) -> None:
        self._manager = manager
        self._session_id: str | None = None
        self._train_loop: TrainLoop | None = None
        self._update_callback = None
        self._server_name = "openforge-train-server"
        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._server_url: str | None = None
        self._server, self._server_thread, self._server_url = start_train_http_server(
            manager,
            host=get_host_ip(),
            port=get_free_port(start=25000),
            log_level=None,
            name=self._server_name,
            request_timeout=self.REQUEST_TIMEOUT_SECONDS,
            poll_interval=self.POLL_INTERVAL_SECONDS,
        )

    @property
    def manager(self) -> TrainManager:
        return self._manager

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def policy_version(self) -> int:
        train_loop = self._train_loop
        return (
            self._manager.latest_stable_policy_version
            if train_loop is None
            else train_loop.policy_version
        )

    @property
    def server_url(self) -> str:
        server_url = self._server_url
        assert server_url is not None
        return server_url

    def register_rollout(self, rollout_router_url: str) -> None:
        self._manager.register_rollout(
            rollout_router_url,
            train_server_url=self.server_url,
        )

    def set_update_callback(self, callback) -> None:
        self._update_callback = callback

    def start_session(
        self,
        *,
        session_id: str,
        store: OpenForgeStore,
    ) -> None:
        assert self._train_loop is None
        self._session_id = session_id
        self._train_loop = TrainLoop(
            session_id=session_id,
            store=store,
            train_manager=self._manager,
            update_callback=self._update_callback,
        )
        self._train_loop.start()

    async def end_session(self) -> None:
        train_loop = self._train_loop
        if train_loop is None:
            return
        await train_loop.stop()
        self._train_loop = None
        self._session_id = None

    def export_checkpoint(self) -> tuple[int, str]:
        policy_version = self.policy_version
        checkpoint_path = self._manager.export_checkpoint(
            policy_version=policy_version,
        )
        return policy_version, checkpoint_path

    def status(self) -> dict[str, object]:
        train_loop = self._train_loop
        if train_loop is None:
            return {
                "active": False,
                "policy_version": self.policy_version,
                "server_url": self.server_url,
            }

        now = time.monotonic()
        last_update_monotonic = train_loop.last_update_monotonic
        return {
            "active": True,
            "global_step": train_loop.global_step,
            "policy_version": train_loop.policy_version,
            "server_url": self.server_url,
            "heartbeat_age_s": max(0.0, now - train_loop.last_heartbeat_monotonic),
            "last_update_age_s": (
                None
                if last_update_monotonic is None
                else max(0.0, now - last_update_monotonic)
            ),
        }

    async def shutdown(self) -> None:
        await self.end_session()
        server = self._server
        server_thread = self._server_thread
        assert server is not None
        assert server_thread is not None
        self._server = None
        self._server_thread = None
        self._server_url = None
        try:
            stop_train_http_server(
                server,
                server_thread,
                timeout=self.SHUTDOWN_TIMEOUT_SECONDS,
                name=self._server_name,
            )
        finally:
            self._manager.shutdown()
