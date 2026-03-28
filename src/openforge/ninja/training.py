# Copyright 2026 openforge

from __future__ import annotations

import time
from typing import Any

from .common import (
    _TRAIN_STATUS_POLL_INTERVAL_SECONDS,
    _normalize_requests,
    _resolve_concurrency,
    _try_active_global_batch_size,
)
from .execute import _execute_grouped_results
from .registered import _RegisteredAgent
from .session import _ActiveSession


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
    wait_timeout: float = 3600.0,
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
        grouped_results, _failures = _execute_grouped_results(
            agent,
            session,
            call_specs,
            group_size=group_size,
            concurrency=resolved_concurrency,
            retries=retries,
            purpose="train",
            raise_on_failure=True,
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
