# Copyright 2026 openforge

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
from multiprocessing.process import BaseProcess
from threading import Lock, Thread
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


@dataclass(slots=True)
class _PendingFinish:
    client: "Client"
    future: Future[None]
    reward: float


@dataclass(slots=True)
class _HttpLane:
    client: httpx.Client
    lock: Lock


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
    HTTP_LANES = 64
    MAX_HTTP_CONNECTIONS = 1
    END_BATCH_RETRIES = 3
    END_BATCH_MAX_SIZE = 320
    END_BATCH_MAX_WAIT_SECONDS = 0.02
    END_BATCH_IDLE_SECONDS = 0.001

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._base_url: str | None = None
        self._http_lanes: list[_HttpLane] = []
        self._http_lane_index = 0
        self._http_lane_lock = Lock()
        self._session_id: str | None = None
        self._finish_lock = Lock()
        self._finish_thread: Thread | None = None
        self._finish_stop = False
        self._pending_finishes: list[_PendingFinish] = []

    @property
    def session_id(self) -> str:
        assert self._session_id is not None, "no active session"
        return self._session_id

    def client(
        self,
        *,
        group_id: str | None = None,
    ) -> Client:
        return Client(
            finish=self._finish_client,
            post=self._post,
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
        response = self._post(
            "/start_trajectory_groups",
            {
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
                        finish=self._finish_client,
                        http=self.http,
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
        payload = {
            "session_id": self.session_id,
            "trajectory_ids": [client.trajectory_id for client in clients],
            "final_rewards": rewards,
        }
        for attempt in range(self.END_BATCH_RETRIES):
            try:
                response = self._post("/end_trajectories", payload)
                response.raise_for_status()
                break
            except httpx.ReadError:
                if attempt + 1 == self.END_BATCH_RETRIES:
                    raise
                time.sleep(self.END_BATCH_MAX_WAIT_SECONDS)
        for client in clients:
            client.close()

    def fail_clients(self, clients: list["Client"]) -> None:
        if not clients:
            return
        response = self._post(
            "/error_trajectories",
            {
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
            },
        )
        response.raise_for_status()
        for client in clients:
            client.close()

    def export_checkpoint(self) -> dict[str, Any]:
        response = self._post(
            "/export_checkpoint",
            {"session_id": self.session_id},
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return payload

    def current_policy_version(self) -> int:
        response = self._get("/current_session")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return int(payload["policy_version"])

    def start(self) -> Session:
        return self

    def __enter__(self) -> Session:
        with STATE.lock:
            gateway = STATE.gateway
            session = STATE.session
        assert gateway is not None, "no active gateway"
        assert session is None, "a session is already active"

        self._base_url = gateway.base_url
        self._http_lanes = [
            _HttpLane(client=self._build_http_client(), lock=Lock())
            for _ in range(self.HTTP_LANES)
        ]
        try:
            response = self._post(
                "/start_session",
                {"runtime": self.runtime_config.model_dump(mode="json")},
            )
            response.raise_for_status()
            self._session_id = str(response.json()["session_id"])
        except Exception:
            self._close_http_clients()
            self._base_url = None
            raise
        self._finish_stop = False
        self._pending_finishes = []
        self._finish_thread = Thread(target=self._flush_finishes, daemon=True)
        self._finish_thread.start()

        with STATE.lock:
            STATE.session = self
        return self

    def __exit__(self, *_: object) -> None:
        session_id = self.session_id
        finish_thread = self._finish_thread

        with STATE.lock:
            STATE.session = None
        self._finish_stop = True
        if finish_thread is not None:
            finish_thread.join()
        self._finish_thread = None
        self._session_id = None

        try:
            response = self._post("/end_session", {"session_id": session_id})
            response.raise_for_status()
        finally:
            self._base_url = None
            self._close_http_clients()

    def _finish_client(self, client: "Client", reward: float) -> None:
        future: Future[None] = Future()
        with self._finish_lock:
            self._pending_finishes.append(
                _PendingFinish(
                    client=client,
                    future=future,
                    reward=reward,
                )
            )
        future.result()

    def _flush_finishes(self) -> None:
        while True:
            if self._finish_stop and not self._pending_finishes:
                return
            if not self._pending_finishes:
                time.sleep(self.END_BATCH_IDLE_SECONDS)
                continue
            time.sleep(self.END_BATCH_MAX_WAIT_SECONDS)
            with self._finish_lock:
                batch = self._pending_finishes[: self.END_BATCH_MAX_SIZE]
                del self._pending_finishes[: len(batch)]
            try:
                self.end_clients(
                    [item.client for item in batch],
                    rewards=[item.reward for item in batch],
                )
            except Exception as exc:
                for item in batch:
                    item.future.set_exception(exc)
            else:
                for item in batch:
                    item.future.set_result(None)

    def _close_http_clients(self) -> None:
        lanes = self._http_lanes
        self._http_lanes = []
        self._http_lane_index = 0
        for lane in lanes:
            lane.client.close()

    def _build_http_client(self) -> httpx.Client:
        assert self._base_url is not None, "no active session"
        return httpx.Client(
            base_url=self._base_url,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            limits=httpx.Limits(
                max_connections=self.MAX_HTTP_CONNECTIONS,
                max_keepalive_connections=self.MAX_HTTP_CONNECTIONS,
            ),
        )

    def _next_http_lane(self) -> _HttpLane:
        assert self._http_lanes, "no active session"
        with self._http_lane_lock:
            lane = self._http_lanes[self._http_lane_index]
            self._http_lane_index = (self._http_lane_index + 1) % len(
                self._http_lanes
            )
        return lane

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        lane = self._next_http_lane()
        with lane.lock:
            return lane.client.post(path, json=payload)

    def _get(self, path: str) -> httpx.Response:
        lane = self._next_http_lane()
        with lane.lock:
            return lane.client.get(path)


class Client:
    """Own one trajectory underneath the active session."""

    def __init__(
        self,
        *,
        finish: Callable[["Client", float], None],
        post: Callable[[str, dict[str, Any]], httpx.Response],
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
    ) -> None:
        self._finish = finish
        self._post = post
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
        response = self._post(
            "/generate",
            {
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "group_id": self._group_id,
                "messages": messages,
                "sampling_params": {} if sampling_params is None else sampling_params,
            },
        )
        response.raise_for_status()
        self._used = True
        return response.json()

    def finish(self, reward: float) -> None:
        try:
            if not self._used:
                self._start()
            self._finish(self, reward)
        finally:
            self.close()

    def fail(self) -> None:
        try:
            if not self._used:
                return
            response = self._post(
                "/error_trajectory",
                {
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
                return
            response = self._post(
                "/discard_trajectory",
                {
                    "session_id": self._session_id,
                    "trajectory_id": self._trajectory_id,
                },
            )
            response.raise_for_status()
        finally:
            self.close()

    def close(self) -> None:
        return

    def _start(self) -> None:
        response = self._post(
            "/start_trajectory",
            {
                "session_id": self._session_id,
                "group_id": self._group_id,
            },
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        self._trajectory_id = str(payload["trajectory_id"])
        self._used = True


def register(
    gateway_config: GatewayServerConfig,
    runtime_config: RuntimeConfig,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as an OpenForge agent."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with STATE.lock:
                gateway = STATE.gateway
                session = STATE.session
            assert gateway is not None, "no active gateway"
            assert session is not None, "no active session"
            group_id = kwargs.pop("__group_id", None)
            discard = bool(kwargs.pop("__discard", False))

            client = session.client(group_id=group_id)
            try:
                result = func(client, *args, **kwargs)
            except Exception:
                client.fail()
                raise

            if discard:
                client.discard()
                return result

            reward = float(result)
            client.finish(reward)
            return reward

        def run(fn: Callable[[], Any]) -> Any:
            with Gateway(gateway_config).start():
                with Session(runtime_config).start():
                    return fn()

        def export_checkpoint() -> dict[str, Any]:
            with STATE.lock:
                session = STATE.session
            assert session is not None, "no active session"
            return session.export_checkpoint()

        def current_policy_version() -> int:
            with STATE.lock:
                session = STATE.session
            assert session is not None, "no active session"
            return session.current_policy_version()

        wrapped.run = run
        wrapped.export_checkpoint = export_checkpoint
        wrapped.current_policy_version = current_policy_version
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
        agent_func: Callable[..., Any],
        *,
        episodes: int = 1,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        def execute() -> None:
            payload = {} if kwargs is None else kwargs
            for _ in range(episodes):
                agent_func(*args, **payload)

        agent_func.run(execute)
