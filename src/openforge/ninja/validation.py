# Copyright 2026 openforge

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .common import _normalize_requests, _resolve_concurrency
from .execute import _execute_many
from .registered import _RegisteredAgent
from .session import _ActiveSession


def _resolve_request_data_path(path: str, *, split_name: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        for suffix in (".jsonl", ".json", ".parquet"):
            split_path = candidate / f"{split_name}{suffix}"
            if split_path.exists():
                return split_path
        raise ValueError(f"{path} is missing {split_name}.jsonl/.json/.parquet")
    return candidate


def _load_request_rows(
    *,
    file_path: str,
    split_name: str,
    max_examples: int | None,
) -> tuple[Path, list[dict[str, Any]]]:
    resolved_path = _resolve_request_data_path(file_path, split_name=split_name)
    suffix = resolved_path.suffix
    rows: list[dict[str, Any]] = []

    def append_row(row: object, *, context: str) -> None:
        if not isinstance(row, dict):
            raise TypeError(f"{context} must decode to an object")
        rows.append(dict(row))

    if suffix == ".jsonl":
        with resolved_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                append_row(
                    json.loads(stripped),
                    context=f"{resolved_path}:{line_number}",
                )
                if max_examples is not None and len(rows) >= max_examples:
                    break
        return resolved_path, rows

    if suffix == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{resolved_path} must decode to a list of objects")
        for index, row in enumerate(payload, start=1):
            append_row(row, context=f"{resolved_path}[{index}]")
            if max_examples is not None and len(rows) >= max_examples:
                break
        return resolved_path, rows

    if suffix == ".parquet":
        import datasets

        dataset = datasets.load_dataset("parquet", data_files=str(resolved_path))[
            "train"
        ]
        for index, row in enumerate(dataset, start=1):
            append_row(row, context=f"{resolved_path}[{index}]")
            if max_examples is not None and len(rows) >= max_examples:
                break
        return resolved_path, rows

    raise ValueError("validation file must be a .jsonl, .json, or .parquet file")


def _build_validation_update(
    *,
    rewards: list[float],
    duration_seconds: float,
    policy_version: int,
) -> dict[str, object]:
    if not rewards:
        raise ValueError("validation file produced no requests")
    reward_mean = sum(rewards) / len(rewards)
    reward_variance = sum((reward - reward_mean) ** 2 for reward in rewards) / len(
        rewards
    )
    return {
        "policy_version": policy_version,
        "sample_count": len(rewards),
        "reward_mean": reward_mean,
        "reward_std": math.sqrt(reward_variance),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "validation_time_s": duration_seconds,
        "samples_per_second": (
            len(rewards) / duration_seconds if duration_seconds > 0.0 else 0.0
        ),
    }


def validate(
    agent: _RegisteredAgent,
    *,
    file_path: str,
    concurrency: int | None = None,
    retries: int = 0,
    max_examples: int | None = None,
    wait_timeout: float = _ActiveSession.VALIDATION_WAIT_TIMEOUT_SECONDS,
    log_to_wandb: bool = True,
) -> dict[str, object]:
    """Run file-backed validation requests against the latest active policy."""
    if not isinstance(agent, _RegisteredAgent):
        raise TypeError("agent must be a function registered with @ninja.agent")
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be > 0")

    resolved_path, requests = _load_request_rows(
        file_path=file_path,
        split_name="validation",
        max_examples=max_examples,
    )
    call_specs, _single_request = _normalize_requests(
        args=(),
        kwargs={},
        requests=requests,
    )
    if not call_specs:
        raise ValueError("validation file produced no requests")
    for request_index, (call_args, call_kwargs) in enumerate(call_specs):
        agent._validate_call(
            call_args,
            call_kwargs,
            request_index=request_index,
        )

    resolved_concurrency = _resolve_concurrency(
        concurrency=concurrency,
        job_count=len(call_specs),
    )
    with agent._session() as session:
        policy_version = session.current_train_policy_version()
        session.wait_for_rollout_policy_version(
            policy_version=policy_version,
            timeout=wait_timeout,
        )
        validation_started_monotonic = time.monotonic()
        rewards = _execute_many(
            agent,
            session,
            call_specs,
            concurrency=resolved_concurrency,
            retries=retries,
            purpose="validation",
        )
        payload = _build_validation_update(
            rewards=rewards,
            duration_seconds=time.monotonic() - validation_started_monotonic,
            policy_version=policy_version,
        )
        if log_to_wandb:
            session.log_validation_update(payload)

    return {
        **payload,
        "file_path": str(resolved_path),
    }
