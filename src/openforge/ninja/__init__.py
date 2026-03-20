# Copyright 2026 openforge

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from functools import wraps
from multiprocessing.process import BaseProcess
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

import httpx

from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import RuntimeConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = [
    "Client",
    "Gateway",
    "NinjaRunner",
    "Session",
    "register",
]


class GlobalState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.gateway: Gateway | None = None
        self.session: Session | None = None


STATE = GlobalState()


class Gateway:
    """Own the gateway server lifecycle for one local process."""

    def __init__(self, gateway_config: GatewayServerConfig) -> None:
        self.config = gateway_config
        host = gateway_config.gateway.host
        port = gateway_config.gateway.port
        self.base_url = f"http://{host}:{port}"
        self._process: BaseProcess | None = None

    def start(self) -> Gateway:
        return self

    def __enter__(self) -> Gateway:
        with STATE.lock:
            assert STATE.gateway is None, "a gateway is already active"

        ctx = mp.get_context("spawn")
        self._process = ctx.Process(
            target=_run_gateway_server,
            args=(self.config,),
            # The gateway may start Ray workers and SGLang subprocesses.
            daemon=False,
        )
        self._process.start()
        self._wait_until_ready()

        with STATE.lock:
            STATE.gateway = self
        return self

    def __exit__(self, *_: object) -> None:
        with STATE.lock:
            assert STATE.session is None, (
                "cannot close the gateway while a session is active"
            )
            STATE.gateway = None

        process = self._process
        self._process = None
        if process is not None:
            process.terminate()
            process.join(timeout=10)

    def _wait_until_ready(self, *, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        client = httpx.Client(base_url=self.base_url, timeout=300.0)
        try:
            while time.monotonic() < deadline:
                try:
                    if client.get("/health").status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadError):
                    pass
                time.sleep(0.5)
        finally:
            client.close()
        raise TimeoutError("gateway server did not start in time")


class Session:
    """Own one active gateway session."""

    REQUEST_TIMEOUT_SECONDS = 300.0

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._http: httpx.Client | None = None
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        assert self._session_id is not None, "no active session"
        return self._session_id

    @property
    def http(self) -> httpx.Client:
        assert self._http is not None, "no active session"
        return self._http

    def client(
        self,
        *,
        group_id: str | None = None,
    ) -> Client:
        return Client(
            base_url=str(self.http.base_url),
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            session_id=self.session_id,
            trajectory_id=f"traj_{uuid4().hex}",
            group_id=group_id,
        )

    def trajectory_groups(
        self,
        *,
        counts: list[int],
        group_ids: list[str | None],
    ) -> list[list[Client]]:
        if len(counts) != len(group_ids):
            raise ValueError("counts must align with group_ids")
        if not counts:
            return []
        if any(count <= 0 for count in counts):
            raise ValueError("count must be >= 1")
        response = self.http.post(
            "/start_trajectory_groups",
            json={
                "session_id": self.session_id,
                "counts": counts,
                "group_ids": group_ids,
            },
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        trajectory_id_groups = payload["trajectory_ids"]
        assert isinstance(trajectory_id_groups, list)
        client_groups: list[list[Client]] = []
        for trajectory_ids, group_id in zip(
            trajectory_id_groups,
            group_ids,
            strict=True,
        ):
            assert isinstance(trajectory_ids, list)
            client_groups.append(
                [
                    Client(
                        base_url=str(self.http.base_url),
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                        session_id=self.session_id,
                        trajectory_id=str(trajectory_id),
                        group_id=group_id,
                    )
                    for trajectory_id in trajectory_ids
                ]
            )
        return client_groups

    def end_clients(
        self,
        clients: list["Client"],
        *,
        rewards: list[float],
    ) -> None:
        if len(clients) != len(rewards):
            raise ValueError("rewards must align with clients")
        if not clients:
            return
        response = self.http.post(
            "/end_trajectories",
            json={
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
                "final_rewards": rewards,
            },
        )
        response.raise_for_status()
        for client in clients:
            client.close()

    def fail_clients(self, clients: list["Client"]) -> None:
        if not clients:
            return
        response = self.http.post(
            "/error_trajectories",
            json={
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
            },
        )
        response.raise_for_status()
        for client in clients:
            client.close()

    def export_checkpoint(self) -> dict[str, Any]:
        response = self.http.post(
            "/export_checkpoint",
            json={"session_id": self.session_id},
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return payload

    def start(self) -> Session:
        return self

    def __enter__(self) -> Session:
        with STATE.lock:
            gateway = STATE.gateway
            session = STATE.session
        assert gateway is not None, "no active gateway"
        assert session is None, "a session is already active"

        self._http = httpx.Client(
            base_url=gateway.base_url,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        try:
            response = self._http.post(
                "/start_session",
                json={"runtime": self.runtime_config.model_dump(mode="json")},
            )
            response.raise_for_status()
            self._session_id = str(response.json()["session_id"])
        except Exception:
            self._http.close()
            self._http = None
            raise

        with STATE.lock:
            STATE.session = self
        return self

    def __exit__(self, *_: object) -> None:
        http = self.http
        session_id = self.session_id

        with STATE.lock:
            STATE.session = None
        self._session_id = None
        self._http = None

        try:
            response = http.post("/end_session", json={"session_id": session_id})
            response.raise_for_status()
        finally:
            http.close()


class Client:
    """Own one trajectory underneath the active session."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._http: httpx.Client | None = None
        self._session_id = session_id
        self._trajectory_id = trajectory_id
        self._group_id = group_id
        self._used = False

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def group_id(self) -> str | None:
        return self._group_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._used = True
        response = self._ensure_http().post(
            "/generate",
            json={
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "group_id": self._group_id,
                "messages": messages,
                "sampling_params": {} if sampling_params is None else sampling_params,
            },
        )
        response.raise_for_status()
        return response.json()

    def finish(self, reward: float) -> None:
        try:
            if not self._used:
                self._start()
            response = self._ensure_http().post(
                "/end_trajectory",
                json={
                    "session_id": self._session_id,
                    "trajectory_id": self._trajectory_id,
                    "final_reward": reward,
                },
            )
            response.raise_for_status()
        finally:
            self.close()

    def fail(self) -> None:
        try:
            if not self._used:
                self._start()
            response = self._ensure_http().post(
                "/error_trajectory",
                json={
                    "session_id": self._session_id,
                    "trajectory_id": self._trajectory_id,
                },
            )
            response.raise_for_status()
        finally:
            self.close()

    def discard(self) -> None:
        try:
            if not self._used:
                self._start()
            response = self._ensure_http().post(
                "/discard_trajectory",
                json={
                    "session_id": self._session_id,
                    "trajectory_id": self._trajectory_id,
                },
            )
            response.raise_for_status()
        finally:
            self.close()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def _start(self) -> None:
        response = self._ensure_http().post(
            "/start_trajectory",
            json={
                "session_id": self._session_id,
                "group_id": self._group_id,
            },
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        self._trajectory_id = str(payload["trajectory_id"])
        self._used = True

    def _ensure_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._http


def register(
    gateway_config: GatewayServerConfig,
    runtime_config: RuntimeConfig,
) -> Callable[[Callable[..., Any]], Callable[..., float]]:
    """Register a function as an OpenForge agent."""

    def decorator(func: Callable[..., Any]) -> Callable[..., float]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> float:
            with STATE.lock:
                gateway = STATE.gateway
                session = STATE.session
            assert gateway is not None, "no active gateway"
            assert session is not None, "no active session"

            client = session.client()
            try:
                reward = float(func(client, *args, **kwargs))
            except Exception:
                client.fail()
                raise

            client.finish(reward)
            return reward

        wrapped.gateway = lambda: Gateway(gateway_config)
        wrapped.session = lambda: Session(runtime_config)
        return wrapped

    return decorator


def _run_gateway_server(gateway_config: GatewayServerConfig) -> None:
    import uvicorn

    from openforge.gateway.server import create_app

    app = create_app(gateway_config)
    uvicorn.run(
        app,
        host=gateway_config.gateway.host,
        port=gateway_config.gateway.port,
        access_log=False,
        log_level="warning",
    )


class NinjaRunner:
    """Run an OpenForge agent function in a loop."""

    @staticmethod
    def run(
        agent_func: Callable[..., float],
        *,
        episodes: int = 1,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        payload = {} if kwargs is None else kwargs
        with agent_func.gateway():
            with agent_func.session():
                for _ in range(episodes):
                    agent_func(*args, **payload)
