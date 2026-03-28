# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from .common import _LOG, _map_parallel, _sleep_before_retry

if TYPE_CHECKING:
    from .registered import _RegisteredAgent
    from .session import _ActiveSession, _TrajectoryClient


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


@dataclass(slots=True)
class _GroupedExecutionResult:
    request_index: int
    trajectory_ids: list[str]
    rewards: list[float]


@dataclass(slots=True)
class _GroupedExecutionFailure:
    request_index: int
    error: str


def _execute_many(
    agent: "_RegisteredAgent",
    session: "_ActiveSession",
    call_specs: list[tuple[tuple[object, ...], dict[str, object]]],
    *,
    concurrency: int,
    retries: int,
    purpose: str,
) -> list[float]:
    def run_once(call_spec: tuple[tuple[object, ...], dict[str, object]]) -> float:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return float(
                    agent._invoke(
                        session,
                        call_spec[0],
                        call_spec[1],
                        purpose=purpose,
                    )
                )
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
    agent: "_RegisteredAgent",
    session: "_ActiveSession",
    call_specs: list[tuple[tuple[object, ...], dict[str, object]]],
    *,
    group_size: int,
    concurrency: int,
    retries: int,
    purpose: str,
) -> list[list[float]]:
    grouped_results, _failures = _execute_grouped_results(
        agent,
        session,
        call_specs,
        group_size=group_size,
        concurrency=concurrency,
        retries=retries,
        purpose=purpose,
        raise_on_failure=True,
    )
    rewards_by_request = [[0.0 for _ in range(group_size)] for _ in call_specs]
    for result in grouped_results:
        rewards_by_request[result.request_index] = result.rewards
    return rewards_by_request


def _execute_grouped_results(
    agent: "_RegisteredAgent",
    session: "_ActiveSession",
    call_specs: list[tuple[tuple[object, ...], dict[str, object]]],
    *,
    group_size: int,
    concurrency: int,
    retries: int,
    purpose: str = "train",
    raise_on_failure: bool = True,
) -> tuple[list[_GroupedExecutionResult], list[_GroupedExecutionFailure]]:
    rollout_concurrency = max(1, min(group_size, concurrency))
    group_concurrency = max(1, concurrency // rollout_concurrency)

    def run_group(
        group_job: tuple[int, tuple[tuple[object, ...], dict[str, object]]],
    ) -> _GroupedExecutionResult | _GroupedExecutionFailure:
        request_index, (call_args, call_kwargs) = group_job
        last_error: Exception | None = None
        group_id = f"group_{uuid4().hex}"

        for attempt in range(retries + 1):
            clients: list[_TrajectoryClient] = []
            try:
                clients = session.trajectory_groups(
                    counts=[group_size],
                    group_ids=[group_id],
                    purpose=purpose,
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
                _LOG.exception(
                    "grouped execute attempt failed: request_index=%s attempt=%s/%s group_id=%s trajectories=%s",
                    request_index,
                    attempt + 1,
                    retries + 1,
                    group_id,
                    [client.trajectory_id for client in clients],
                )
                _fail_clients_best_effort(session, clients)
                if attempt >= retries:
                    if not raise_on_failure:
                        return _GroupedExecutionFailure(
                            request_index=request_index,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    raise RuntimeError(
                        f"grouped execute failed for request index {request_index}"
                    ) from exc
                _sleep_before_retry(attempt)

        assert last_error is not None
        raise last_error

    outputs = _map_parallel(
        list(enumerate(call_specs)),
        concurrency=group_concurrency,
        fn=run_group,
    )
    grouped_results: list[_GroupedExecutionResult] = []
    failures: list[_GroupedExecutionFailure] = []
    for output in outputs:
        if isinstance(output, _GroupedExecutionResult):
            grouped_results.append(output)
        else:
            failures.append(output)
    return grouped_results, failures
