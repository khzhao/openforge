# Copyright 2026 openforge

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

__all__ = ["fan_out", "submit_fan_out", "wait_futures"]

_T = TypeVar("_T")


def fan_out(
    calls: Sequence[Callable[[], _T]],
    *,
    max_workers: int | None = None,
) -> list[_T]:
    """Run blocking calls in parallel and return their results in order."""
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=max_workers or len(calls)) as executor:
        return wait_futures(submit_fan_out(executor, calls))


def submit_fan_out(
    executor: Executor,
    calls: Sequence[Callable[[], _T]],
) -> list[Future[_T]]:
    """Submit a batch of blocking calls onto an existing executor."""
    return [executor.submit(call) for call in calls]


def wait_futures(futures: Sequence[Future[_T]]) -> list[_T]:
    """Wait for futures and return their results in order."""
    return [future.result() for future in futures]
