#!/usr/bin/env python3
# Copyright 2026 openforge

"""Helpers for standalone test scripts."""

import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar


ExcT = TypeVar("ExcT", bound=BaseException)


def assert_raises(
    exc_type: type[ExcT],
    fn: Callable[..., Any],
    *args: Any,
    match: str | None = None,
    **kwargs: Any,
) -> ExcT:
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"Expected {exc_type.__name__} matching {match!r}, got {exc!r}"
            ) from exc
        return exc
    except Exception as exc:
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def run_named_tests(
    test_cases: Sequence[tuple[str, Callable[[], None]]],
    *,
    summary_label: str,
) -> int:
    for name, test_fn in test_cases:
        print(f"[RUN ] {name}")
        test_fn()
        print(f"[PASS] {name}")

    print()
    print(f"All {len(test_cases)} {summary_label} tests passed.")
    return 0
