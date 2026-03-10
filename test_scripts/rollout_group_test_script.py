#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone tests for RolloutWorkerGroup."""

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
class FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeWorker:
    def __init__(self, endpoint: RolloutEndpoint) -> None:
        self.endpoint_value = endpoint
        self.load_calls: list[Any] = []

        self.load_policy_artifact = FakeRemoteMethod(self._load_policy_artifact)
        self.endpoint = FakeRemoteMethod(lambda: self.endpoint_value)
        self.allocate_port = FakeRemoteMethod(lambda *, start: start)

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


def _build_group(cfg: OpenForgeConfig, worker: FakeWorker) -> RolloutWorkerGroup:
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = cfg.rollout.resolve(cfg.cluster)
    group._placement_groups = []
    group._workers = [worker]
    group._engine_endpoints = []
    return group


def test_rollout_group_loads_artifacts() -> None:
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
        routable = group.load_policy_artifact(
            SimpleNamespace(path="/tmp/policy.pt", policy_version=7),
            engine_ids=[0],
        )

    assert worker.load_calls[0].policy_version == 7
    assert routable[0].policy_version == 7


def test_rollout_group_filters_routable_endpoints_and_targets_workers() -> None:
    cfg = make_config(rollout_node_pool="train")
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = SimpleNamespace(
        engines=[
            SimpleNamespace(
                engine_id=0,
                role="regular",
            ),
            SimpleNamespace(
                engine_id=1,
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
    group._engine_endpoints = []

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        assert [endpoint.name for endpoint in group.routable_endpoints()] == ["regular-0"]
        assert [worker.endpoint_value.name for worker in group._target_workers([0])] == ["regular-0"]
        group.pause_generation(engine_ids=[0], mode="drain")
        group.continue_generation(engine_ids=[0])


def test_rollout_group_build_worker_spec_and_create_workers() -> None:
    cfg = make_config(rollout_node_pool="train")
    resolved = cfg.rollout.resolve(cfg.cluster)
    group = object.__new__(RolloutWorkerGroup)
    group.cfg = cfg
    group._resolved = resolved
    group._placement_groups = []
    group._workers = []
    group._engine_endpoints = []

    fake_runtime_workers = [
        SimpleNamespace(
            resolve_node_ip_address=FakeRemoteMethod(lambda: "10.0.0.1"),
            allocate_port=FakeRemoteMethod(lambda *, start: start + 5),
        )
    ]
    group._workers = fake_runtime_workers

    with mock.patch("openforge.rollout.group.ray.get", side_effect=lambda value: value):
        spec = group._build_worker_spec(resolved.engines[0])

    assert spec.host == "10.0.0.1"
    assert spec.port == 30005
    assert spec.bootstrap_port is None

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

    assert len(group._placement_groups) == 1
    assert len(group._workers) == 1


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
                "rollout_group_loads_artifacts",
                test_rollout_group_loads_artifacts,
            ),
            (
                "rollout_group_filters_routable_endpoints_and_targets_workers",
                test_rollout_group_filters_routable_endpoints_and_targets_workers,
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
        description="Standalone tests for RolloutWorkerGroup."
    )
    parser.parse_args()

    with TemporaryDirectory(prefix="openforge_rollout_group_") as temp_dir:
        return run_suite(Path(temp_dir))


if __name__ == "__main__":
    raise SystemExit(main())
