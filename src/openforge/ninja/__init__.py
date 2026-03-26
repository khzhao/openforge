# Copyright 2026 openforge

from __future__ import annotations

import inspect
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Callable
from uuid import uuid4

import httpx

from openforge import active_state
from openforge.configs.models import GatewayServerConfig
from openforge.gateway.types import ChatCompletionResponse, ModelListResponse

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = ["agent", "train"]

_AUTO_CONCURRENCY_CAP = 512
_AUTO_CONCURRENCY_CPU_MULTIPLIER = 16
_AUTO_CONCURRENCY_FLOOR = 32
_TRAIN_STATUS_POLL_INTERVAL_SECONDS = 0.25


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2.0, 0.5 * (2**attempt)))


def _fail_clients_best_effort(
    session: "_ActiveSession",
    clients: list["_TrajectoryClient"],
) -> None:
    if not clients:
        return
    try:
        session.fail_clients(clients)
    except Exception:
        for client in clients:
            try:
                client.fail()
            except Exception:
                pass


def _map_parallel(
    items: list[Any],
    *,
    concurrency: int,
    fn: Callable[[Any], Any],
) -> list[Any]:
    if not items:
        return []
    max_workers = max(1, min(concurrency, len(items)))
    if max_workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, items))


def _resolve_concurrency(
    *,
    concurrency: int | None,
    job_count: int,
) -> int:
    if concurrency is None:
        cpu_count = os.cpu_count() or 1
        auto_limit = min(
            _AUTO_CONCURRENCY_CAP,
            max(
                _AUTO_CONCURRENCY_FLOOR,
                cpu_count * _AUTO_CONCURRENCY_CPU_MULTIPLIER,
            ),
        )
        return max(1, min(job_count, auto_limit))
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    return concurrency


def _function_expects_client(func: Callable[..., Any]) -> bool:
    parameters = list(inspect.signature(func).parameters.values())
    if not parameters:
        return False
    first = parameters[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return False
    return first.name == "client"


def _validate_registered_function(func: Callable[..., Any]) -> None:
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            "ninja.agent does not support async functions; "
            "define a synchronous agent function instead"
        )


def _coerce_reward(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be a finite real number, got bool")
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{context} must be a finite real number, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(reward):
        raise ValueError(f"{context} must be finite, got {reward!r}")
    return reward


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


def _try_active_global_batch_size() -> int | None:
    try:
        runtime_config = active_state.load_active_runtime_config()
    except AssertionError:
        return None
    return int(runtime_config.train.global_batch_size)


class _ActiveSession:
    REQUEST_TIMEOUT_SECONDS = 300.0
    END_RETRIES = 3
    END_RETRY_DELAY_SECONDS = 0.02

    def __init__(self, gateway_target: tuple[str, int]) -> None:
        host, port = gateway_target
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
            raise AssertionError(
                "no active session recorded; "
                "run `python -m openforge.cli.main session start --runtime-config ...` first"
            )
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
        used: bool = False,
    ) -> _TrajectoryClient:
        return _TrajectoryClient(
            post=self.post,
            retry_post=self._retry_post,
            session_id=self.session_id,
            trajectory_id=trajectory_id or f"traj_{uuid4().hex}",
            group_id=group_id,
            used=used,
        )

    def trajectory_groups(
        self,
        *,
        counts: list[int],
        group_ids: list[str | None],
    ) -> list[list[_TrajectoryClient]]:
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
                    used=True,
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
        clients: list["_TrajectoryClient"],
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

    def fail_clients(self, clients: list["_TrajectoryClient"]) -> None:
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

    def trajectory_statuses(self, trajectory_ids: list[str]) -> dict[str, str]:
        response = self._retry_post(
            "/trajectory_statuses",
            {
                "session_id": self.session_id,
                "trajectory_ids": trajectory_ids,
            },
        )
        payload = response.json()
        assert isinstance(payload, dict)
        trajectories = payload["trajectories"]
        assert isinstance(trajectories, list)
        return {
            str(trajectory["trajectory_id"]): str(trajectory["status"])
            for trajectory in trajectories
        }

    def agent_client(self, client: "_TrajectoryClient") -> _AgentClient:
        return _AgentClient(self, client)

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


class _TrajectoryClient:
    """Own one trajectory underneath the gateway's active session."""

    def __init__(
        self,
        *,
        post: Callable[[str, dict[str, Any]], httpx.Response],
        retry_post: Callable[[str, dict[str, Any]], httpx.Response],
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
        used: bool = False,
    ) -> None:
        self._post = post
        self._retry_post = retry_post
        self._session_id = session_id
        self._trajectory_id = trajectory_id
        self._group_id = group_id
        self._used = used

    @property
    def trajectory_id(self) -> str:
        return self._trajectory_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def group_id(self) -> str | None:
        return self._group_id

    def mark_used(self) -> None:
        self._used = True

    def finish(self, reward: float) -> None:
        if not self._used:
            self._start()
        response = self._retry_post(
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
        response = self._retry_post(
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


class _ChatCompletionsClient:
    def __init__(
        self,
        session: _ActiveSession,
        trajectory: _TrajectoryClient,
    ) -> None:
        self._session = session
        self._trajectory = trajectory

    def create(self, **kwargs: Any) -> ChatCompletionResponse:
        self._trajectory.mark_used()
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None:
            extra_body = {}
        else:
            extra_body = dict(extra_body)
        payload = dict(kwargs)
        payload.update(extra_body)
        payload["_openforge"] = {
            "session_id": self._trajectory.session_id,
            "trajectory_id": self._trajectory.trajectory_id,
            "group_id": self._trajectory.group_id,
        }
        response = self._session.post("/v1/chat/completions", payload)
        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())


class _ChatClient:
    def __init__(self, session: _ActiveSession, trajectory: _TrajectoryClient) -> None:
        self.completions = _ChatCompletionsClient(session, trajectory)


class _ModelsClient:
    def __init__(self, session: _ActiveSession) -> None:
        self._session = session

    def list(self) -> ModelListResponse:
        response = self._session.get("/v1/models")
        response.raise_for_status()
        return ModelListResponse.model_validate(response.json())


class _AgentClient:
    def __init__(self, session: _ActiveSession, trajectory: _TrajectoryClient) -> None:
        self.chat = _ChatClient(session, trajectory)
        self.models = _ModelsClient(session)

    def close(self) -> None:
        return None


class _RegisteredAgent:
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        gateway_config: GatewayServerConfig | None,
    ) -> None:
        _validate_registered_function(func)
        self._func = func
        self._gateway_config = gateway_config
        self._expects_client = _function_expects_client(func)
        self._signature = inspect.signature(func)
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._validate_call(args, kwargs)
        with self._session() as session:
            return self._invoke(session, args, kwargs)

    def sample(
        self,
        *args: Any,
        requests: list[dict[str, Any]] | None = None,
        concurrency: int | None = None,
        group_size: int = 1,
        num_rollouts: int | None = None,
        retries: int = 0,
        **kwargs: Any,
    ) -> Any:
        if num_rollouts is not None:
            if group_size != 1:
                raise ValueError("group_size cannot be combined with num_rollouts")
            group_size = num_rollouts
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if retries < 0:
            raise ValueError("retries must be >= 0")

        call_specs, single_request = _normalize_requests(
            args=args,
            kwargs=kwargs,
            requests=requests,
        )
        for request_index, (call_args, call_kwargs) in enumerate(call_specs):
            self._validate_call(
                call_args,
                call_kwargs,
                request_index=request_index if not single_request else None,
            )
        resolved_concurrency = _resolve_concurrency(
            concurrency=concurrency,
            job_count=len(call_specs) * group_size,
        )
        with self._session() as session:
            if group_size == 1:
                results = _execute_many(
                    self,
                    session,
                    call_specs,
                    concurrency=resolved_concurrency,
                    retries=retries,
                )
            else:
                results = _execute_grouped(
                    self,
                    session,
                    call_specs,
                    group_size=group_size,
                    concurrency=resolved_concurrency,
                    retries=retries,
                )
        if single_request:
            return results[0]
        return results

    def save(self) -> dict[str, Any]:
        with self._session() as session:
            return session.export_checkpoint()

    def policy_version(self) -> int:
        with self._session() as session:
            return session.current_policy_version()

    def _call_body(
        self,
        session: _ActiveSession,
        client: _TrajectoryClient,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> float:
        return _coerce_reward(
            self._call_func(session, client, call_args, call_kwargs),
            context=f"{self.__name__} return value",
        )

    def _invoke(
        self,
        session: _ActiveSession,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Any:
        client = session.client()
        try:
            reward = self._call_body(session, client, call_args, call_kwargs)
        except Exception:
            client.fail()
            raise

        client.finish(reward)
        return reward

    def _call_func(
        self,
        session: _ActiveSession,
        client: _TrajectoryClient,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Any:
        if not self._expects_client:
            return self._func(*call_args, **call_kwargs)

        agent_client = session.agent_client(client)
        try:
            return self._func(agent_client, *call_args, **call_kwargs)
        finally:
            agent_client.close()

    def _session(self) -> _ActiveSession:
        return _ActiveSession(_resolve_gateway_target(self._gateway_config))

    def _validate_call(
        self,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        *,
        request_index: int | None = None,
    ) -> None:
        bind_args = call_args
        if self._expects_client:
            bind_args = (object(), *call_args)
        try:
            self._signature.bind(*bind_args, **call_kwargs)
        except TypeError as exc:
            if request_index is None:
                raise TypeError(f"{self.__name__} call is invalid: {exc}") from exc
            raise TypeError(
                f"{self.__name__} request[{request_index}] is invalid: {exc}"
            ) from exc


@dataclass(slots=True)
class _GroupedExecutionResult:
    request_index: int
    trajectory_ids: list[str]
    rewards: list[float]


def _resolve_gateway_target(
    gateway_config: GatewayServerConfig | None,
) -> tuple[str, int]:
    if gateway_config is not None:
        return gateway_config.gateway.host, gateway_config.gateway.port
    return active_state.load_active_gateway_target()


def _execute_many(
    agent: _RegisteredAgent,
    session: _ActiveSession,
    call_specs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    concurrency: int,
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
        concurrency=concurrency,
        fn=run_once,
    )


def _execute_grouped(
    agent: _RegisteredAgent,
    session: _ActiveSession,
    call_specs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    group_size: int,
    concurrency: int,
    retries: int,
) -> list[list[float]]:
    grouped_results = _execute_grouped_results(
        agent,
        session,
        call_specs,
        group_size=group_size,
        concurrency=concurrency,
        retries=retries,
    )
    rewards_by_request = [[0.0 for _ in range(group_size)] for _ in call_specs]
    for result in grouped_results:
        rewards_by_request[result.request_index] = result.rewards
    return rewards_by_request


def _execute_grouped_results(
    agent: _RegisteredAgent,
    session: _ActiveSession,
    call_specs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    group_size: int,
    concurrency: int,
    retries: int,
) -> list[_GroupedExecutionResult]:
    rollout_concurrency = max(1, min(group_size, concurrency))
    group_concurrency = max(1, concurrency // rollout_concurrency)

    def run_group(
        group_job: tuple[int, tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> _GroupedExecutionResult:
        request_index, (call_args, call_kwargs) = group_job
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            clients: list[_TrajectoryClient] = []
            try:
                clients = session.trajectory_groups(
                    counts=[group_size],
                    group_ids=[f"group_{uuid4().hex}"],
                )[0]

                def run_rollout(
                    rollout_job: tuple[int, _TrajectoryClient],
                ) -> tuple[int, _TrajectoryClient, float]:
                    rollout_index, client = rollout_job
                    reward = agent._call_body(
                        session,
                        client,
                        call_args,
                        call_kwargs,
                    )
                    return rollout_index, client, reward

                results = _map_parallel(
                    list(enumerate(clients)),
                    concurrency=rollout_concurrency,
                    fn=run_rollout,
                )
                ordered_results = sorted(results, key=lambda item: item[0])
                ordered_clients = [
                    client for _rollout_index, client, _reward in ordered_results
                ]
                ordered_rewards = [
                    reward for _rollout_index, _client, reward in ordered_results
                ]
                session.end_clients(
                    ordered_clients,
                    rewards=ordered_rewards,
                )
                return _GroupedExecutionResult(
                    request_index=request_index,
                    trajectory_ids=[client.trajectory_id for client in ordered_clients],
                    rewards=ordered_rewards,
                )
            except Exception as exc:
                last_error = exc
                _fail_clients_best_effort(session, clients)
                if attempt >= retries:
                    raise RuntimeError(
                        f"grouped execute failed for request index {request_index}"
                    ) from exc
                _sleep_before_retry(attempt)

        assert last_error is not None
        raise last_error

    return _map_parallel(
        list(enumerate(call_specs)),
        concurrency=group_concurrency,
        fn=run_group,
    )


def _wait_for_trained_trajectories(
    session: _ActiveSession,
    trajectory_ids: list[str],
    *,
    timeout: float,
) -> None:
    if timeout <= 0:
        raise ValueError("wait_timeout must be > 0")

    pending = set(trajectory_ids)
    deadline = time.monotonic() + timeout
    while pending:
        statuses = session.trajectory_statuses(sorted(pending))
        failed = {
            trajectory_id: status
            for trajectory_id, status in statuses.items()
            if status in {"failed", "discarded"}
        }
        if failed:
            details = ", ".join(
                f"{trajectory_id}={status}"
                for trajectory_id, status in sorted(failed.items())
            )
            raise RuntimeError(
                f"training failed before trajectories completed: {details}"
            )

        pending = {
            trajectory_id
            for trajectory_id, status in statuses.items()
            if status != "trained"
        }
        if not pending:
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(
            min(
                _TRAIN_STATUS_POLL_INTERVAL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
        )

    raise TimeoutError(
        f"{len(pending)} trajectories were not trained within {timeout} seconds"
    )


def train(
    agent: _RegisteredAgent,
    *,
    inputs: list[dict[str, Any]],
    group_size: int,
    concurrency: int | None = None,
    retries: int = 0,
    wait_timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run grouped trajectories through the gateway and wait until they are trained."""
    if not isinstance(agent, _RegisteredAgent):
        raise TypeError("agent must be a function registered with @ninja.agent")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    call_specs, _single_request = _normalize_requests(
        args=(),
        kwargs={},
        requests=inputs,
    )
    if not call_specs:
        raise ValueError("inputs must not be empty")
    for request_index, (call_args, call_kwargs) in enumerate(call_specs):
        agent._validate_call(
            call_args,
            call_kwargs,
            request_index=request_index,
        )

    resolved_concurrency = _resolve_concurrency(
        concurrency=concurrency,
        job_count=len(call_specs) * group_size,
    )
    global_batch_size: int | None = None
    if agent._gateway_config is None:
        global_batch_size = _try_active_global_batch_size()
        if global_batch_size is not None and group_size > global_batch_size:
            raise ValueError(
                "group_size must be <= active runtime global_batch_size: "
                f"{group_size} > {global_batch_size}"
            )
    with agent._session() as session:
        initial_policy_version = session.current_policy_version()
        grouped_results = _execute_grouped_results(
            agent,
            session,
            call_specs,
            group_size=group_size,
            concurrency=resolved_concurrency,
            retries=retries,
        )
        trajectory_ids = [
            trajectory_id
            for result in grouped_results
            for trajectory_id in result.trajectory_ids
        ]
        try:
            _wait_for_trained_trajectories(
                session,
                trajectory_ids,
                timeout=wait_timeout,
            )
        except TimeoutError as exc:
            if global_batch_size is None:
                raise
            raise TimeoutError(
                f"{exc} (call produced {len(trajectory_ids)} trajectories; "
                f"active runtime global_batch_size={global_batch_size})"
            ) from exc
        final_policy_version = session.current_policy_version()

    rewards = [reward for result in grouped_results for reward in result.rewards]
    group_mean_rewards = [
        sum(result.rewards) / len(result.rewards) for result in grouped_results
    ]
    return {
        "group_size": group_size,
        "prompt_groups": len(grouped_results),
        "samples": len(rewards),
        "initial_policy_version": initial_policy_version,
        "final_policy_version": final_policy_version,
        "max_group_reward": max(max(result.rewards) for result in grouped_results),
        "mean_group_reward": sum(group_mean_rewards) / len(group_mean_rewards),
        "sample_mean_reward": sum(rewards) / len(rewards),
    }


def agent(
    gateway_config: GatewayServerConfig | None = None,
) -> Callable[[Callable[..., Any]], _RegisteredAgent]:
    """Register a function as an OpenForge agent."""

    def decorator(func: Callable[..., Any]) -> _RegisteredAgent:
        return _RegisteredAgent(
            func=func,
            gateway_config=gateway_config,
        )

    return decorator
