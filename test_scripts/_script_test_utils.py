# Copyright 2026 openforge

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Iterator


@contextmanager
def expect_raises(
    exc_type: type[BaseException],
    match: str | None = None,
) -> Iterator[None]:
    try:
        yield
    except exc_type as exc:
        if match is not None:
            assert re.search(match, str(exc)), str(exc)
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def run_tests(tests: list[Callable[[], None]]) -> int:
    for test in tests:
        print(f"RUN {test.__name__}", flush=True)
        test()
    print(f"PASS {len(tests)} tests", flush=True)
    return 0


def install_test_stubs() -> None:
    _install_sglang_stubs()
    _install_sglang_router_stubs()


def _install_sglang_stubs() -> None:
    if "sglang" in sys.modules:
        return

    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    utils = ModuleType("sglang.srt.utils")
    entrypoints = ModuleType("sglang.srt.entrypoints")
    http_server = ModuleType("sglang.srt.entrypoints.http_server")

    class ServerArgs:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def kill_process_tree(*args, **kwargs) -> None:
        return None

    def launch_server(*args, **kwargs) -> None:
        return None

    server_args.ServerArgs = ServerArgs
    utils.kill_process_tree = kill_process_tree
    http_server.launch_server = launch_server

    sglang.srt = srt
    srt.server_args = server_args
    srt.utils = utils
    srt.entrypoints = entrypoints
    entrypoints.http_server = http_server

    sys.modules["sglang"] = sglang
    sys.modules["sglang.srt"] = srt
    sys.modules["sglang.srt.server_args"] = server_args
    sys.modules["sglang.srt.utils"] = utils
    sys.modules["sglang.srt.entrypoints"] = entrypoints
    sys.modules["sglang.srt.entrypoints.http_server"] = http_server


def _install_sglang_router_stubs() -> None:
    if "sglang_router" in sys.modules:
        return

    sglang_router = ModuleType("sglang_router")
    router_args = ModuleType("sglang_router.router_args")
    launch_router = ModuleType("sglang_router.launch_router")

    class RouterArgs:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def launch_router_fn(*args, **kwargs) -> None:
        return None

    router_args.RouterArgs = RouterArgs
    launch_router.launch_router = launch_router_fn

    sglang_router.router_args = router_args
    sglang_router.launch_router = launch_router

    sys.modules["sglang_router"] = sglang_router
    sys.modules["sglang_router.router_args"] = router_args
    sys.modules["sglang_router.launch_router"] = launch_router
