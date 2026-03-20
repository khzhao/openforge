# Copyright 2026 openforge

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from multiprocessing import Process
from typing import Any, Callable

import httpx
from loguru import logger

from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import ChatMessage, RuntimeConfig

__all__ = ["AgentClient", "ChatMessage", "NinjaRunner", "register"]


@dataclass(frozen=True, slots=True)
class NinjaSpec:
    gateway_config: GatewayServerConfig
    runtime_config: RuntimeConfig
    num_episodes: int


class AgentClient:
    """Synchronous HTTP client for the OpenForge gateway."""

    def __init__(
        self,
        *,
        gateway_config: GatewayServerConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        host = gateway_config.gateway.host
        port = gateway_config.gateway.port
        self._base_url = f"http://{host}:{port}"
        self._runtime_config = runtime_config
        self._http: httpx.Client | None = None
        self.session_id: str | None = None
        self._active_trajectory_id: str | None = None

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(base_url=self._base_url, timeout=300.0)
        return self._http

    def wait_until_ready(self, *, timeout: float = 120.0) -> None:
        """Poll /health until the gateway is up."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._client().get("/health").status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError):
                pass
            time.sleep(0.5)
        raise TimeoutError("gateway server did not start in time")

    def start_session(self) -> str:
        resp = self._client().post(
            "/start_session",
            json={"runtime": self._runtime_config.model_dump()},
        )
        resp.raise_for_status()
        self.session_id = resp.json()["session_id"]
        return self.session_id

    def start_trajectory(
        self,
        *,
        parent_trajectory_id: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {"session_id": self.session_id}
        if parent_trajectory_id is not None:
            payload["parent_trajectory_id"] = parent_trajectory_id
        resp = self._client().post("/start_trajectory", json=payload)
        resp.raise_for_status()
        self._active_trajectory_id = resp.json()["trajectory_id"]
        return self._active_trajectory_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "trajectory_id": self._active_trajectory_id,
            "messages": messages,
        }
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        resp = self._client().post("/generate", json=payload)
        resp.raise_for_status()
        return resp.json()

    def end_trajectory(self, *, reward: float) -> None:
        resp = self._client().post(
            "/end_trajectory",
            json={
                "session_id": self.session_id,
                "trajectory_id": self._active_trajectory_id,
                "final_reward": reward,
            },
        )
        resp.raise_for_status()
        self._active_trajectory_id = None

    def error_trajectory(self) -> None:
        resp = self._client().post(
            "/error_trajectory",
            json={
                "session_id": self.session_id,
                "trajectory_id": self._active_trajectory_id,
            },
        )
        resp.raise_for_status()
        self._active_trajectory_id = None

    def end_session(self) -> None:
        resp = self._client().post(
            "/end_session",
            json={"session_id": self.session_id},
        )
        resp.raise_for_status()
        self.session_id = None

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


class AgentWrapper:
    """Manages trajectory lifecycle around an agent function."""

    def __init__(
        self,
        agent_func: Callable[..., Any],
        *,
        gateway_config: GatewayServerConfig,
        runtime_config: RuntimeConfig,
        num_episodes: int,
    ) -> None:
        self._agent_func = agent_func
        self.openforge_ninja_spec = NinjaSpec(
            gateway_config=gateway_config,
            runtime_config=runtime_config,
            num_episodes=num_episodes,
        )
        self.client = AgentClient(
            gateway_config=gateway_config,
            runtime_config=runtime_config,
        )
        wraps(agent_func)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        self.client.start_trajectory()
        try:
            reward = self._agent_func(self.client, *args, **kwargs)
        except Exception:
            self.client.error_trajectory()
            raise
        self.client.end_trajectory(reward=float(reward))
        return reward


def register(
    *,
    gateway_config: GatewayServerConfig,
    runtime_config: RuntimeConfig,
    num_episodes: int = 1,
):
    def decorator(agent_func: Callable[..., Any]) -> AgentWrapper:
        return AgentWrapper(
            agent_func,
            gateway_config=gateway_config,
            runtime_config=runtime_config,
            num_episodes=num_episodes,
        )

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

    @staticmethod
    def run(agent_func: Callable[..., Any]) -> None:
        spec: NinjaSpec = agent_func.openforge_ninja_spec
        client: AgentClient = agent_func.client

        server_process = Process(
            target=_run_gateway_server,
            args=(spec.gateway_config,),
            daemon=True,
        )
        server_process.start()

        try:
            client.wait_until_ready()
            client.start_session()
            try:
                for episode in range(spec.num_episodes):
                    logger.info("episode {} / {}", episode + 1, spec.num_episodes)
                    try:
                        agent_func()
                    except Exception:
                        logger.exception("episode {} failed", episode + 1)
            finally:
                if client.session_id is not None:
                    try:
                        client.end_session()
                    except Exception:
                        logger.exception("failed to end session cleanly")
        finally:
            client.close()
            server_process.terminate()
            server_process.join(timeout=10)
