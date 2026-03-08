#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone tests for RolloutWorkerGroup session management."""

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import mock

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import RolloutEndpoint
from openforge.rollout.group import RolloutWorkerGroup
from script_test_utils import assert_raises, run_named_tests


def make_config(*, rollout_node_pool: str) -> OpenForgeConfig:
    cluster = [
        {
            "node_pool": "train",
            "num_nodes": 1,
            "num_gpus_per_node": 4,
            "num_cpus_per_node": 8,
        }
    ]
    if rollout_node_pool != "train":
        cluster.append(
            {
                "node_pool": rollout_node_pool,
                "num_nodes": 1,
                "num_gpus_per_node": 4,
                "num_cpus_per_node": 8,
            }
        )

    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "test"},
            "gateway": {
                "host": "127.0.0.1",
                "port": 8000,
            },
            "model": {
                "model_name_or_path": "test-model",
                "tokenizer_name_or_path": "test-tokenizer",
            },
            "cluster": cluster,
            "train": {
                "backend": "fsdp2",
                "backend_config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": False,
                    "mixed_precision": {
                        "param_dtype": "float32",
                        "reduce_dtype": "float32",
                    },
                    "offload": {"mode": "none", "pin_memory": False},
                    "amp": {
                        "enabled": False,
                        "precision": "float32",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 1e-3,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 0,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                        "min_lr_rate": None,
                    },
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints_dir": "/tmp/openforge_rollout_group_test",
                "cpus_per_worker": 1,
                "parallelism": {
                    "data_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
                "placement": {
                    "node_pool": "train",
                    "strategy": "PACK",
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 16,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                    "spaces_between_words": True,
                },
                "engine_topology": "regular",
                "engines": [
                    {
                        "name": "regular",
                        "role": "regular",
                        "replicas": 1,
                        "gpus_per_engine": 1,
                        "cpus_per_engine": 0,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "placement": {
                            "node_pool": rollout_node_pool,
                            "strategy": "PACK",
                        },
                    }
                ],
            },
        }
    )


def make_pd_config() -> OpenForgeConfig:
    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "test"},
            "gateway": {
                "host": "127.0.0.1",
                "port": 8000,
            },
            "model": {
                "model_name_or_path": "test-model",
                "tokenizer_name_or_path": "test-tokenizer",
            },
            "cluster": [
                {
                    "node_pool": "train",
                    "num_nodes": 1,
                    "num_gpus_per_node": 4,
                    "num_cpus_per_node": 8,
                }
            ],
            "train": {
                "backend": "fsdp2",
                "backend_config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": False,
                    "mixed_precision": {
                        "param_dtype": "float32",
                        "reduce_dtype": "float32",
                    },
                    "offload": {"mode": "none", "pin_memory": False},
                    "amp": {
                        "enabled": False,
                        "precision": "float32",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 1e-3,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": "constant",
                        "warmup_steps": 0,
                        "min_lr": 0.0,
                        "num_cycles": 0.5,
                        "min_lr_rate": None,
                    },
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints_dir": "/tmp/openforge_rollout_group_test",
                "cpus_per_worker": 1,
                "parallelism": {
                    "data_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
                "placement": {
                    "node_pool": "train",
                    "strategy": "PACK",
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 16,
                    "stop": [],
                    "stop_token_ids": [],
                    "skip_special_tokens": True,
                    "no_stop_trim": False,
                    "spaces_between_words": True,
                },
                "engine_topology": "pd",
                "engines": [
                    {
                        "name": "prefill",
                        "role": "prefill",
                        "replicas": 1,
                        "gpus_per_engine": 1,
                        "cpus_per_engine": 0,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "placement": {
                            "node_pool": "train",
                            "strategy": "PACK",
                        },
                    },
                    {
                        "name": "decode",
                        "role": "decode",
                        "replicas": 1,
                        "gpus_per_engine": 1,
                        "cpus_per_engine": 0,
                        "parallelism": {
                            "data_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_parallel_size": 1,
                        },
                        "placement": {
                            "node_pool": "train",
                            "strategy": "PACK",
                        },
                    },
                ],
            },
        }
    )


class FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeWorker:
    def __init__(self, endpoint: RolloutEndpoint) -> None:
        self.endpoint_value = endpoint
        self.tensor_begin_calls: list[Any] = []
        self.tensor_finish_calls: list[Any] = []
        self.dist_begin_calls: list[Any] = []
        self.dist_finish_calls: list[Any] = []
        self.abort_calls: list[Any] = []
        self.load_calls: list[Any] = []

        self.begin_tensor_update = FakeRemoteMethod(self._begin_tensor_update)
        self.finish_tensor_update = FakeRemoteMethod(self._finish_tensor_update)
        self.begin_distributed_update = FakeRemoteMethod(self._begin_distributed_update)
        self.finish_distributed_update = FakeRemoteMethod(self._finish_distributed_update)
        self.abort_update = FakeRemoteMethod(self._abort_update)
        self.load_policy_artifact = FakeRemoteMethod(self._load_policy_artifact)
        self.endpoint = FakeRemoteMethod(lambda: self.endpoint_value)
        self.allocate_port = FakeRemoteMethod(lambda *, start: start)

    def _begin_tensor_update(self, session) -> None:
        self.tensor_begin_calls.append(session)

    def _finish_tensor_update(self, session) -> RolloutEndpoint:
        self.tensor_finish_calls.append(session)
        self.endpoint_value = replace(self.endpoint_value, policy_version=session.policy_version)
        return self.endpoint_value

    def _begin_distributed_update(self, session) -> None:
        self.dist_begin_calls.append(session)

    def _finish_distributed_update(self, session) -> RolloutEndpoint:
        self.dist_finish_calls.append(session)
        self.endpoint_value = replace(self.endpoint_value, policy_version=session.policy_version)
        return self.endpoint_value

    def _abort_update(self, *, session_id: str) -> None:
        self.abort_calls.append(session_id)

    def _load_policy_artifact(self, artifact) -> RolloutEndpoint:
        self.load_calls.append(artifact)
        self.endpoint_value = replace(
            self.endpoint_value,
            policy_version=artifact.policy_version,
            model_path=artifact.path,
        )
        return self.endpoint_value

    pause_generation = FakeRemoteMethod(lambda self=None, *, mode="abort": mode)
    continue_generation = FakeRemoteMethod(lambda self=None: None)
    shutdown = FakeRemoteMethod(lambda self=None: None)


class FakeRouterWorker:
    def __init__(self) -> None:
        self.initialize_calls: list[dict[str, Any]] = []
        self.restart_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0
        self.endpoint_value = RolloutEndpoint(
            name="pd-router",
            role="router",
            host="127.0.0.1",
            port=39000,
            bootstrap_port=None,
            url="http://127.0.0.1:39000",
            healthy=True,
            policy_version=None,
            model_path=None,
        )
        self.initialize = FakeRemoteMethod(self._initialize)
        self.restart = FakeRemoteMethod(self._restart)
        self.endpoint = FakeRemoteMethod(lambda: self.endpoint_value)
        self.shutdown = FakeRemoteMethod(self._shutdown)

    def _initialize(self, **kwargs):
        self.initialize_calls.append(kwargs)
        return self.endpoint_value

    def _restart(self, **kwargs):
        self.restart_calls.append(kwargs)
        return self.endpoint_value

    def _shutdown(self):
        self.shutdown_calls += 1


def _build_group(cfg: OpenForgeConfig, worker: FakeWorker) -> RolloutWorkerGroup:
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = cfg.rollout.resolve(cfg.cluster)
    group._placement_groups = []
    group._workers = [worker]
    group._router_worker = None
    group._engine_endpoints = []
    group._router_endpoint = None
    return group


def test_rollout_group_opens_tensor_sessions_for_same_pool() -> None:
    cfg = make_config(rollout_node_pool="train")
    endpoint = RolloutEndpoint(
        name="regular-0",
        role="regular",
        host="127.0.0.1",
        port=30000,
        bootstrap_port=None,
        url="http://127.0.0.1:30000",
        healthy=True,
        policy_version=None,
        model_path="test-model",
    )
    worker = FakeWorker(endpoint)
    group = _build_group(cfg, worker)

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        session = group.open_tensor_update(policy_version=7)
        routable = group.commit_update(session)

    assert session.policy_version == 7
    assert session.engine_ids == [0]
    assert worker.tensor_begin_calls[0].session_id == session.session_id
    assert worker.tensor_finish_calls[0].session_id == session.session_id
    assert routable[0].policy_version == 7


def test_rollout_group_opens_distributed_sessions_for_remote_pool() -> None:
    cfg = make_config(rollout_node_pool="rollout")
    endpoint = RolloutEndpoint(
        name="regular-0",
        role="regular",
        host="127.0.0.1",
        port=30001,
        bootstrap_port=None,
        url="http://127.0.0.1:30001",
        healthy=True,
        policy_version=None,
        model_path="test-model",
    )
    worker = FakeWorker(endpoint)
    group = _build_group(cfg, worker)

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        session = group.open_distributed_update(policy_version=8)
        group.abort_update(session)
        routable = group.commit_update(session)

    assert session.policy_version == 8
    assert session.engine_ids == [0]
    assert session.rank_offsets == {0: 1}
    assert worker.dist_begin_calls[0].session_id == session.session_id
    assert worker.abort_calls == [session.session_id]
    assert worker.dist_finish_calls[0].session_id == session.session_id
    assert routable[0].policy_version == 8


def test_rollout_group_filters_routable_endpoints_and_validates_targets() -> None:
    cfg = make_config(rollout_node_pool="train")
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = SimpleNamespace(
        engines=[
            SimpleNamespace(
                engine_id=0,
                placement=SimpleNamespace(node_pool="train"),
                gpus_per_engine=1,
                role="regular",
            ),
            SimpleNamespace(
                engine_id=1,
                placement=SimpleNamespace(node_pool="rollout"),
                gpus_per_engine=1,
                role="regular",
            ),
        ]
    )
    group._placement_groups = []
    group._workers = [
        FakeWorker(
            RolloutEndpoint(
                name="regular-0",
                role="regular",
                host="127.0.0.1",
                port=30000,
                bootstrap_port=None,
                url="http://127.0.0.1:30000",
                healthy=True,
                policy_version=None,
                model_path="test-model",
            )
        ),
        FakeWorker(
            RolloutEndpoint(
                name="regular-1",
                role="regular",
                host="127.0.0.1",
                port=30001,
                bootstrap_port=None,
                url="http://127.0.0.1:30001",
                healthy=False,
                policy_version=None,
                model_path="test-model",
            )
        ),
    ]
    group._router_worker = None
    group._engine_endpoints = []
    group._router_endpoint = None

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        assert [endpoint.name for endpoint in group.routable_endpoints()] == ["regular-0"]
        assignments = group._target_assignments(engine_ids=[1], default_mode="tensor")
        assert [engine.engine_id for _, engine in assignments] == [1]
        assert [worker.endpoint_value.name for worker in group._target_workers([0])] == ["regular-0"]
        group.pause_generation(engine_ids=[0], mode="drain")
        group.continue_generation(engine_ids=[0])

    assert_raises(
        ValueError,
        group._target_assignments,
        engine_ids=[99],
        default_mode="tensor",
        match="no rollout engines selected",
    )


def test_rollout_group_initializes_and_refreshes_pd_router() -> None:
    cfg = make_pd_config()
    resolved = cfg.rollout.resolve(cfg.cluster)
    endpoints = [
        RolloutEndpoint(
            name="prefill-0",
            role="prefill",
            host="127.0.0.1",
            port=30010,
            bootstrap_port=30011,
            url="http://127.0.0.1:30010",
            healthy=True,
            policy_version=None,
            model_path="test-model",
        ),
        RolloutEndpoint(
            name="decode-0",
            role="decode",
            host="127.0.0.1",
            port=30020,
            bootstrap_port=None,
            url="http://127.0.0.1:30020",
            healthy=True,
            policy_version=None,
            model_path="test-model",
        ),
    ]
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = resolved
    group._placement_groups = []
    group._workers = [FakeWorker(endpoints[0]), FakeWorker(endpoints[1])]
    group._router_endpoint = None
    fake_router = FakeRouterWorker()

    class FakeRemoteRouter:
        def options(self, **kwargs):
            _ = kwargs
            return self

        def remote(self):
            return fake_router

    with (
        mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value),
        mock.patch("openforge.rollout.group.ray.remote", return_value=FakeRemoteRouter()),
    ):
        router_endpoint = group._initialize_router()
        assert router_endpoint.role == "router"
        assert fake_router.initialize_calls[0]["prefill_endpoints"][0].role == "prefill"

        group._router_worker = fake_router
        refreshed = group._restart_router()
        assert refreshed.role == "router"
        assert fake_router.restart_calls[0]["decode_endpoints"][0].role == "decode"

        group._router_endpoint = router_endpoint
        assert group.routable_endpoints()[0].role == "router"


def test_rollout_group_build_worker_spec_and_create_workers() -> None:
    cfg = make_pd_config()
    resolved = cfg.rollout.resolve(cfg.cluster)
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = resolved
    group._placement_groups = []
    group._workers = []
    group._router_worker = None
    group._engine_endpoints = []
    group._router_endpoint = None

    fake_runtime_workers = [
        SimpleNamespace(
            resolve_node_ip_address=FakeRemoteMethod(lambda: "10.0.0.1"),
            allocate_port=FakeRemoteMethod(lambda *, start: start + 5),
        ),
        SimpleNamespace(
            resolve_node_ip_address=FakeRemoteMethod(lambda: "10.0.0.2"),
            allocate_port=FakeRemoteMethod(lambda *, start: start + 7),
        ),
    ]
    group._workers = fake_runtime_workers

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        spec = group._build_worker_spec(resolved.engines[0])

    assert spec.host == "10.0.0.1"
    assert spec.port == 30005
    assert spec.bootstrap_port == 30011

    group._workers = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    class FakeRemoteWorker:
        def __init__(self) -> None:
            self.options_calls: list[dict[str, Any]] = []

        def options(self, **kwargs):
            self.options_calls.append(kwargs)
            return self

        def remote(self):
            return {"worker_index": len(self.options_calls) - 1}

    fake_remote_worker = FakeRemoteWorker()

    with (
        mock.patch(
            "openforge.rollout.group.placement_group",
            side_effect=lambda bundles, strategy: FakePlacementGroup(),
        ),
        mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value),
        mock.patch("openforge.rollout.group.ray.remote", return_value=fake_remote_worker),
        mock.patch(
            "openforge.rollout.group.PlacementGroupSchedulingStrategy",
            side_effect=lambda **kwargs: kwargs,
        ),
    ):
        group._create_workers(strategy="PACK")

    assert len(group._placement_groups) == 2
    assert len(group._workers) == 2


def test_rollout_group_loads_artifacts_and_shuts_down() -> None:
    cfg = make_config(rollout_node_pool="train")
    endpoint = RolloutEndpoint(
        name="regular-0",
        role="regular",
        host="127.0.0.1",
        port=30000,
        bootstrap_port=None,
        url="http://127.0.0.1:30000",
        healthy=True,
        policy_version=None,
        model_path="test-model",
    )
    worker = FakeWorker(endpoint)
    group = _build_group(cfg, worker)
    group._placement_groups = ["pg-0"]

    with (
        mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value),
        mock.patch("openforge.rollout.group.ray.util.remove_placement_group") as remove_pg,
    ):
        routable = group.load_policy_artifact(
            SimpleNamespace(path="/tmp/policy.pt", policy_version=4),
            engine_ids=[0],
        )
        group.shutdown()

    assert routable[0].policy_version == 4
    assert group._workers == []
    remove_pg.assert_called_once_with("pg-0")


def run_suite(artifacts_dir: Path) -> int:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")
    return run_named_tests(
        [
            (
                "rollout_group_opens_tensor_sessions_for_same_pool",
                test_rollout_group_opens_tensor_sessions_for_same_pool,
            ),
            (
                "rollout_group_opens_distributed_sessions_for_remote_pool",
                test_rollout_group_opens_distributed_sessions_for_remote_pool,
            ),
            (
                "rollout_group_filters_routable_endpoints_and_validates_targets",
                test_rollout_group_filters_routable_endpoints_and_validates_targets,
            ),
            (
                "rollout_group_initializes_and_refreshes_pd_router",
                test_rollout_group_initializes_and_refreshes_pd_router,
            ),
            (
                "rollout_group_build_worker_spec_and_create_workers",
                test_rollout_group_build_worker_spec_and_create_workers,
            ),
            (
                "rollout_group_loads_artifacts_and_shuts_down",
                test_rollout_group_loads_artifacts_and_shuts_down,
            ),
        ],
        summary_label="RolloutWorkerGroup",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone tests for RolloutWorkerGroup session management."
    )
    parser.parse_args()

    with TemporaryDirectory(prefix="openforge_rollout_group_") as temp_dir:
        return run_suite(Path(temp_dir))


if __name__ == "__main__":
    raise SystemExit(main())
