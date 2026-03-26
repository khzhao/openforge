# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

from _script_test_utils import run_tests

import openforge.train.runtime as train_runtime_module
from openforge.train.runtime import TrainRuntime


class _FakeTrainLoop:
    instances: list["_FakeTrainLoop"] = []
    drain_results: list[bool] = []

    def __init__(self, *, session_id: str, store, train_manager) -> None:
        self.session_id = session_id
        self.store = store
        self.train_manager = train_manager
        self.started = False
        self.stopped = False
        self.train_once_calls = 0
        self.policy_version = 3
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def train_once(self) -> bool:
        self.train_once_calls += 1
        if self.__class__.drain_results:
            return self.__class__.drain_results.pop(0)
        return False


class _FakeTrainManager:
    def __init__(self) -> None:
        self.latest_stable_policy_version = 0
        self.register_rollout_calls: list[tuple[str, str]] = []
        self.export_checkpoint_calls: list[int] = []
        self.shutdown_calls = 0

    def register_rollout(
        self,
        rollout_router_url: str,
        *,
        train_server_url: str,
    ) -> None:
        self.register_rollout_calls.append((rollout_router_url, train_server_url))

    def export_checkpoint(self, *, policy_version: int) -> str:
        self.export_checkpoint_calls.append(policy_version)
        return f"/tmp/checkpoint-{policy_version}"

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@contextmanager
def _patched_train_loop() -> Iterator[None]:
    _FakeTrainLoop.instances.clear()
    _FakeTrainLoop.drain_results = []
    with (
        patch.object(train_runtime_module, "TrainLoop", _FakeTrainLoop),
        patch.object(
            train_runtime_module,
            "start_train_http_server",
            lambda manager, **kwargs: (object(), object(), "http://127.0.0.1:26000"),
        ),
        patch.object(train_runtime_module, "stop_train_http_server", lambda *args, **kwargs: None),
    ):
        yield


def test_train_runtime_starts_session_loop() -> None:
    with _patched_train_loop():
        runtime = TrainRuntime(_FakeTrainManager())
        store = object()

        runtime.start_session(session_id="sess_0", store=store)

        assert runtime.session_id == "sess_0"
        assert runtime.policy_version == 3
        assert len(_FakeTrainLoop.instances) == 1
        assert _FakeTrainLoop.instances[0].store is store
        assert _FakeTrainLoop.instances[0].started is True


def test_train_runtime_registers_rollout_and_exports_checkpoint() -> None:
    with _patched_train_loop():
        manager = _FakeTrainManager()
        runtime = TrainRuntime(manager)
        runtime.register_rollout("http://127.0.0.1:31000")
        runtime.start_session(session_id="sess_0", store=object())

        policy_version, checkpoint_path = runtime.export_checkpoint()

        assert manager.register_rollout_calls == [
            ("http://127.0.0.1:31000", "http://127.0.0.1:26000")
        ]
        assert manager.export_checkpoint_calls == [3]
        assert policy_version == 3
        assert checkpoint_path == "/tmp/checkpoint-3"


def test_train_runtime_end_session_stops_without_draining() -> None:
    async def run() -> None:
        with _patched_train_loop():
            runtime = TrainRuntime(_FakeTrainManager())
            runtime.start_session(session_id="sess_0", store=object())

            await runtime.end_session()

            assert runtime.session_id is None
            assert runtime.policy_version == 0
            assert _FakeTrainLoop.instances[0].stopped is True
            assert _FakeTrainLoop.instances[0].train_once_calls == 0

    asyncio.run(run())


def test_train_runtime_shutdown_stops_session_and_manager() -> None:
    async def run() -> None:
        with _patched_train_loop():
            manager = _FakeTrainManager()
            runtime = TrainRuntime(manager)
            runtime.start_session(session_id="sess_0", store=object())

            await runtime.shutdown()

            assert _FakeTrainLoop.instances[0].stopped is True
            assert _FakeTrainLoop.instances[0].train_once_calls == 0
            assert manager.shutdown_calls == 1

    asyncio.run(run())


def main() -> int:
    return run_tests(
        [
            test_train_runtime_starts_session_loop,
            test_train_runtime_registers_rollout_and_exports_checkpoint,
            test_train_runtime_end_session_stops_without_draining,
            test_train_runtime_shutdown_stops_session_and_manager,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
