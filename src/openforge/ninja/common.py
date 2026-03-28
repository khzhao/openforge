# Copyright 2026 openforge

from __future__ import annotations

import inspect
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from openforge import active_state
from openforge.configs.models import GatewayServerConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
_LOG = logging.getLogger(__name__)

_AUTO_CONCURRENCY_CAP = 512
_AUTO_CONCURRENCY_CPU_MULTIPLIER = 16
_AUTO_CONCURRENCY_FLOOR = 32
_TRAIN_STATUS_POLL_INTERVAL_SECONDS = 0.25


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2.0, 0.5 * (2**attempt)))


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


def _resolve_gateway_target(
    gateway_config: GatewayServerConfig | None,
) -> tuple[str, int]:
    if gateway_config is not None:
        return gateway_config.gateway.host, gateway_config.gateway.port
    return active_state.load_active_gateway_target()
