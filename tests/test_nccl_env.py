# Copyright 2026 openforge

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from _script_test_utils import install_test_stubs

install_test_stubs()

import openforge.rollout.sglang.engine_group as engine_group_module
import openforge.rollout.manager as rollout_manager_module
import openforge.train.manager as train_manager_module
from openforge.rollout.sglang.engine_group import start_sglang_engines
from openforge.rollout.sglang.types import EngineAddr
from openforge.rollout.manager import RolloutManager
from openforge.train.manager import TrainManager
from openforge.utils.nccl import NCCL_ENV_DEFAULTS, apply_nccl_env_defaults


def test_apply_nccl_env_defaults_preserves_existing_values() -> None:
    env = {"NCCL_CUMEM_ENABLE": "1"}

    applied = apply_nccl_env_defaults(env)

    assert env == {
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
    }
    assert applied == env


def test_train_manager_initialize_sets_nccl_defaults() -> None:
    class _FakeWorkerHandle:
        def __init__(self) -> None:
            self.initialize = SimpleNamespace(remote=lambda spec: spec)

    class _FakeRayTrainWorker:
        option_calls: list[dict[str, object]] = []

        @classmethod
        def options(cls, **kwargs):
            cls.option_calls.append(kwargs)
            return cls

        @classmethod
        def remote(cls):
            return _FakeWorkerHandle()

    manager = TrainManager()
    _FakeRayTrainWorker.option_calls = []
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            cpus_per_worker=1,
            num_workers=2,
        )
    )
    placement_group = {"actor": ("pg", [0, 1], [2, 3])}

    with (
        patch.object(train_manager_module, "RayTrainWorker", _FakeRayTrainWorker),
        patch.object(train_manager_module.ray, "get", lambda payload: payload),
        patch.dict("os.environ", {}, clear=True),
    ):
        manager.initialize(
            cfg,
            master_addr="127.0.0.1",
            master_port=20000,
            placement_group=placement_group,
        )

        for key, value in NCCL_ENV_DEFAULTS.items():
            assert os.environ[key] == value
        assert _FakeRayTrainWorker.option_calls
        assert _FakeRayTrainWorker.option_calls[0]["runtime_env"] == {
            "env_vars": dict(NCCL_ENV_DEFAULTS)
        }


def test_rollout_manager_initialize_sets_nccl_defaults() -> None:
    class _FakeEngineGroup:
        def __init__(self, cfg, placement_groups) -> None:
            self.engine_specs = ["engine-spec"]
            self.engine_addrs = {"regular-0": "addr"}

        def initialize(self, *, engine_addrs=None) -> None:
            return

        def shutdown(self) -> None:
            return

    class _FakeRolloutRouter:
        def __init__(self, url: str) -> None:
            self.url = url

        def initialize(self, **kwargs) -> None:
            return

        def shutdown(self) -> None:
            return

    manager = RolloutManager(cfg=SimpleNamespace(), placement_groups={})

    with (
        patch.object(rollout_manager_module, "EngineGroup", _FakeEngineGroup),
        patch.object(rollout_manager_module, "RolloutRouter", _FakeRolloutRouter),
        patch.dict("os.environ", {}, clear=True),
    ):
        manager.initialize(router_ip="127.0.0.1", router_port=31000)

        for key, value in NCCL_ENV_DEFAULTS.items():
            assert os.environ[key] == value
        assert (
            manager.router_spec.request_timeout_secs
            == rollout_manager_module.RolloutManager.REQUEST_TIMEOUT_SECONDS
        )


def test_start_sglang_engines_propagates_nccl_defaults() -> None:
    class _FakeWorkerHandle:
        def __init__(self, engine_name: str) -> None:
            self.initialize = SimpleNamespace(remote=lambda spec: spec)
            self.launch = SimpleNamespace(remote=lambda addr: (engine_name, addr))

    class _FakeEngineWorkerFactory:
        option_calls: list[dict[str, object]] = []

        @classmethod
        def options(cls, **kwargs):
            cls.option_calls.append(kwargs)
            engine_rank = len(cls.option_calls) - 1
            return SimpleNamespace(
                remote=lambda: _FakeWorkerHandle(f"regular-{engine_rank}")
            )

    cfg = SimpleNamespace(
        rollout=SimpleNamespace(
            engine_groups=[
                SimpleNamespace(
                    name="regular",
                    worker_type="regular",
                    replicas=1,
                    num_gpus_per_replica=1,
                    num_cpus_per_replica=1,
                    parallelism=SimpleNamespace(world_size=1),
                    enable_memory_saver=False,
                    sglang_server_overrides={},
                )
            ]
        )
    )
    placement_groups = {"rollout": ("pg", [0], [3])}
    _FakeEngineWorkerFactory.option_calls = []
    engine_addrs = {
        "regular-0": EngineAddr(
            host="127.0.0.1",
            port=31000,
            nccl_port=31001,
            dist_init_addr="127.0.0.1:31002",
        )
    }

    with (
        patch.object(engine_group_module.ray, "remote", lambda cls: _FakeEngineWorkerFactory),
        patch.object(engine_group_module.ray, "get", lambda payload: payload),
        patch.dict("os.environ", {}, clear=True),
    ):
        start_sglang_engines(cfg, placement_groups, engine_addrs=engine_addrs)

        assert _FakeEngineWorkerFactory.option_calls
        env_vars = _FakeEngineWorkerFactory.option_calls[0]["runtime_env"]["env_vars"]
        for key, value in NCCL_ENV_DEFAULTS.items():
            assert env_vars[key] == value
