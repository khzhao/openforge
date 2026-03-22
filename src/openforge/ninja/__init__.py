# Copyright 2026 openforge

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import update_wrapper
from typing import Any, Callable
from uuid import uuid4

import httpx

from openforge.configs.models import GatewayServerConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = [
    "Client",
    "register",
]


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2.0, 0.5 * (2**attempt)))


def _map_parallel(
    items: list[Any],
    *,
    max_parallelism: int,
    fn: Callable[[Any], Any],
) -> list[Any]:
    if not items:
        return []
    max_workers = max(1, min(max_parallelism, len(items)))
    if max_workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, items))


def _resolve_max_parallelism(
    *,
    max_parallelism: int | None,
    job_count: int,
) -> int:
    if max_parallelism is None:
        return max(1, job_count)
    if max_parallelism <= 0:
        raise ValueError("max_parallelism must be > 0")
    return max_parallelism


def _normalize_requests(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    requests: list[dict[str, Any]] | None,
) -> tuple[list[tuple[tuple[Any, ...], dict[str, Any]]], bool]:
    if requests is None:
        return [(tuple(args), dict(kwargs))], True
    if args or kwargs:
        raise ValueError("direct args/kwargs cannot be combined with requests")

    normalized: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError("each request must be a dict")
        normalized.append(((), dict(request)))
    return normalized, False


class _ActiveSession:
    REQUEST_TIMEOUT_SECONDS = 300.0
    END_RETRIES = 3
    END_RETRY_DELAY_SECONDS = 0.02

    def __init__(self, gateway_config: GatewayServerConfig) -> None:
        host = gateway_config.gateway.host
        port = gateway_config.gateway.port
        self._base_url = f"http://{host}:{port}"
        self._http: httpx.Client | None = None
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        assert self._session_id is not None, "no active session"
        return self._session_id

    def __enter__(self) -> _ActiveSession:
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        response = self.get("/current_session")
        if response.status_code == 404:
            self.__exit__(None, None, None)
            raise AssertionError("no active session")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        self._session_id = str(payload["session_id"])
        return self

    def __exit__(self, *_: object) -> None:
        client = self._http
        self._http = None
        self._session_id = None
        if client is not None:
            client.close()

    def client(
        self,
        *,
        group_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> Client:
        return Client(
            post=self.post,
            session_id=self.session_id,
            trajectory_id=trajectory_id or f"traj_{uuid4().hex}",
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
        if any(count <= 0 for count in counts):
            raise ValueError("count must be >= 1")
        if not counts:
            return []

        response = self.post(
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
        return [
            [
                self.client(
                    group_id=group_id,
                    trajectory_id=str(trajectory_id),
                )
                for trajectory_id in trajectory_ids
            ]
            for trajectory_ids, group_id in zip(
                trajectory_id_groups,
                group_ids,
                strict=True,
            )
        ]

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
        self._retry_post(
            "/end_trajectories",
            {
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
                "final_rewards": rewards,
            },
        )

    def fail_clients(self, clients: list["Client"]) -> None:
        if not clients:
            return
        self._retry_post(
            "/error_trajectories",
            {
                "session_id": self.session_id,
                "trajectory_ids": [client.trajectory_id for client in clients],
            },
        )

    def export_checkpoint(self) -> dict[str, Any]:
        response = self.post(
            "/export_checkpoint",
            {"session_id": self.session_id},
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return payload

    def current_policy_version(self) -> int:
        response = self.get("/current_session")
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload, dict)
        return int(payload["policy_version"])

    def post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        client = self._http_client()
        return client.post(path, json=payload)

    def get(self, path: str) -> httpx.Response:
        client = self._http_client()
        return client.get(path)

    def _http_client(self) -> httpx.Client:
        assert self._http is not None, "gateway is not connected"
        return self._http

    def _retry_post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(self.END_RETRIES):
            try:
                response = self.post(path, payload)
                response.raise_for_status()
                return response
            except httpx.ReadError:
                if attempt + 1 == self.END_RETRIES:
                    raise
                time.sleep(self.END_RETRY_DELAY_SECONDS)
        raise AssertionError("unreachable")


class Client:
    """Own one trajectory underneath the gateway's active session."""

    def __init__(
        self,
        *,
        post: Callable[[str, dict[str, Any]], httpx.Response],
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
    ) -> None:
        self._post = post
        self._session_id = session_id
        self._trajectory_id = trajectory_id
        self._group_id = group_id
        self._used = False

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

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
        if not self._used:
            self._start()
        response = self._post(
            "/end_trajectory",
            {
                "session_id": self._session_id,
                "trajectory_id": self._trajectory_id,
                "final_reward": reward,
            },
        )
        response.raise_for_status()

    def fail(self) -> None:
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


class _RegisteredAgent:
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        gateway_config: GatewayServerConfig,
    ) -> None:
        self._func = func
        self._gateway_config = gateway_config
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._session() as session:
            return self._invoke(session, args, kwargs)

    def execute(
        self,
        *args: Any,
        requests: list[dict[str, Any]] | None = None,
        max_parallelism: int | None = None,
        group_size: int = 1,
        retries: int = 0,
        **kwargs: Any,
    ) -> Any:
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if retries < 0:
            raise ValueError("retries must be >= 0")

        call_specs, single_request = _normalize_requests(
            args=args,
            kwargs=kwargs,
            requests=requests,
        )
        resolved_max_parallelism = _resolve_max_parallelism(
            max_parallelism=max_parallelism,
            job_count=len(call_specs) * group_size,
        )
        with self._session() as session:
            if group_size == 1:
                results = _execute_many(
                    self,
                    session,
                    call_specs,
                    max_parallelism=resolved_max_parallelism,
                    retries=retries,
                )
            else:
                results = _execute_grouped(
                    self,
                    session,
                    call_specs,
                    group_size=group_size,
                    max_parallelism=resolved_max_parallelism,
                    retries=retries,
                )
        if single_request:
            return results[0]
        return results

    def export_checkpoint(self) -> dict[str, Any]:
        with self._session() as session:
            return session.export_checkpoint()

    def current_policy_version(self) -> int:
        with self._session() as session:
            return session.current_policy_version()

    def _call_body(
        self,
        client: Client,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> float:
        return float(self._func(client, *call_args, **call_kwargs))

    def _invoke(
        self,
        session: _ActiveSession,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Any:
        client = session.client()
        try:
            result = self._func(client, *call_args, **call_kwargs)
        except Exception:
            client.fail()
            raise

        reward = float(result)
        client.finish(reward)
        return reward

    def _session(self) -> _ActiveSession:
        return _ActiveSession(self._gateway_config)


def _execute_many(
    agent: _RegisteredAgent,
    session: _ActiveSession,
    call_specs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    max_parallelism: int,
    retries: int,
) -> list[float]:
    def run_once(call_spec: tuple[tuple[Any, ...], dict[str, Any]]) -> float:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return float(agent._invoke(session, call_spec[0], call_spec[1]))
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                _sleep_before_retry(attempt)
        assert last_error is not None
        raise last_error

    return _map_parallel(
        call_specs,
        max_parallelism=max_parallelism,
        fn=run_once,
    )


def _execute_grouped(
    agent: _RegisteredAgent,
    session: _ActiveSession,
    call_specs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    group_size: int,
    max_parallelism: int,
    retries: int,
) -> list[list[float]]:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        client_groups: list[list[Client]] = []
        try:
            client_groups = session.trajectory_groups(
                counts=[group_size for _ in call_specs],
                group_ids=[f"group_{uuid4().hex}" for _ in call_specs],
            )
            jobs = [
                (request_index, rollout_index, call_args, call_kwargs, client)
                for request_index, ((call_args, call_kwargs), clients) in enumerate(
                    zip(call_specs, client_groups, strict=True)
                )
                for rollout_index, client in enumerate(clients)
            ]

            def run_job(
                job: tuple[int, int, tuple[Any, ...], dict[str, Any], Client]
            ) -> tuple[int, int, Client, float]:
                request_index, rollout_index, call_args, call_kwargs, client = job
                reward = agent._call_body(client, call_args, call_kwargs)
                return request_index, rollout_index, client, reward

            results = _map_parallel(
                jobs,
                max_parallelism=max_parallelism,
                fn=run_job,
            )
            session.end_clients(
                [client for _request_index, _rollout_index, client, _reward in results],
                rewards=[
                    reward
                    for _request_index, _rollout_index, _client, reward in results
                ],
            )

            rewards_by_request = [
                [0.0 for _ in range(group_size)] for _ in call_specs
            ]
            for request_index, rollout_index, _client, reward in results:
                rewards_by_request[request_index][rollout_index] = reward
            return rewards_by_request
        except Exception as exc:
            last_error = exc
            pending_clients = [
                client for clients in client_groups for client in clients
            ]
            if pending_clients:
                try:
                    session.fail_clients(pending_clients)
                except Exception:
                    for client in pending_clients:
                        try:
                            client.fail()
                        except Exception:
                            pass
            if attempt >= retries:
                raise
            _sleep_before_retry(attempt)

    assert last_error is not None
    raise last_error


def register(
    gateway_config: GatewayServerConfig,
) -> Callable[[Callable[..., Any]], _RegisteredAgent]:
    """Register a function as an OpenForge agent."""

    def decorator(func: Callable[..., Any]) -> _RegisteredAgent:
        return _RegisteredAgent(
            func=func,
            gateway_config=gateway_config,
        )

    return decorator
