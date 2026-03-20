# Copyright 2026 openforge

from __future__ import annotations

import time
from functools import wraps
from multiprocessing import Process
from threading import Lock
from typing import Any, Callable

import httpx

from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import RuntimeConfig

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
        self._process: Process | None = None

    def start(self) -> Gateway:
        return self

    def __enter__(self) -> Gateway:
        with STATE.lock:
            assert STATE.gateway is None, "a gateway is already active"

        self._process = Process(
            target=_run_gateway_server,
            args=(self.config,),
            daemon=True,
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

    def start(self) -> Session:
        return self

    def __enter__(self) -> Session:
        with STATE.lock:
            gateway = STATE.gateway
            session = STATE.session
        assert gateway is not None, "no active gateway"
        assert session is None, "a session is already active"

        self._http = httpx.Client(base_url=gateway.base_url, timeout=300.0)
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

    def __init__(self, *, http: httpx.Client, session_id: str) -> None:
        self._http = http
        self._session_id = session_id
        response = self._http.post(
            "/start_trajectory",
            json={"session_id": self._session_id},
        )
        response.raise_for_status()
        self._trajectory_id = str(response.json()["trajectory_id"])

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._http.post(
            "/generate",
            json={
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "messages": messages,
                "sampling_params": {} if sampling_params is None else sampling_params,
            },
        )
        response.raise_for_status()
        return response.json()

    def finish(self, reward: float) -> None:
        response = self._http.post(
            "/end_trajectory",
            json={
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "final_reward": reward,
            },
        )
        response.raise_for_status()

    def fail(self) -> None:
        response = self._http.post(
            "/error_trajectory",
            json={
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
            },
        )
        response.raise_for_status()


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

            client = Client(http=session.http, session_id=session.session_id)
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
        log_level="info",
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
