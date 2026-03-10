#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone tests for RolloutWorker and SGLang runtime glue."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from openforge.configs.models import OpenForgeConfig
from openforge.policy.types import PolicyArtifactRef
from openforge.rollout.sglang.client import SGLangControlClient
from openforge.rollout.sglang.runtime import SGLangRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.rollout.worker import RolloutWorker
from openforge.rollout.types import RolloutWorkerSpec
import openforge.rollout.worker as rollout_worker_module
from script_test_utils import assert_raises, run_named_tests


def make_config(
    checkpoints_dir: str,
) -> OpenForgeConfig:
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
                    "node_pool": "default",
                    "num_nodes": 1,
                    "num_gpus_per_node": 8,
                    "num_cpus_per_node": 16,
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
                    "offload": {
                        "mode": "none",
                        "pin_memory": False,
                    },
                    "amp": {
                        "enabled": False,
                        "precision": "float32",
                        "use_grad_scaler": False,
                    },
                    "optim": {
                        "lr": 5e-6,
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
                "global_batch_size": 8,
                "mini_batch_size": 4,
                "micro_batch_size": 2,
                "checkpoints_dir": checkpoints_dir,
                "cpus_per_worker": 1,
                "parallelism": {
                    "data_parallel_size": 4,
                    "pipeline_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_parallel_size": 1,
                },
                "placement": {
                    "node_pool": "default",
                    "strategy": "PACK",
                },
            },
            "rollout": {
                "backend": "sglang",
                "request": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 64,
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
                            "node_pool": "default",
                            "strategy": "PACK",
                        },
                    }
                ],
            },
        }
    )


class FakeRuntime:
    created: list["FakeRuntime"] = []

    def __init__(self, spec: SGLangEngineSpec) -> None:
        self.spec = spec
        self.started = False
        self.stopped = False
        self.load_calls: list[Any] = []
        FakeRuntime.created.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def restart(self, *, model_path: str | None = None, policy_version: int | None = None) -> None:
        if model_path is not None:
            self.spec.model_path = model_path
        if policy_version is not None:
            self.spec.policy_version = policy_version

    def endpoint(self):
        return mock.Mock(
            name=self.spec.name,
            role=self.spec.role,
            host=self.spec.host,
            port=self.spec.port,
            bootstrap_port=self.spec.bootstrap_port,
            url=self.spec.url,
            healthy=True,
            policy_version=self.spec.policy_version,
            model_path=self.spec.model_path,
        )

    def is_healthy(self) -> bool:
        return True

    def pause_generation(self, *, mode: str = "abort") -> None:
        _ = mode

    def continue_generation(self) -> None:
        return None

    def flush_cache(self) -> bool:
        return True

    def get_model_info(self) -> dict[str, Any]:
        return {"weight_version": self.spec.policy_version}

    def get_server_info(self) -> dict[str, Any]:
        return {"name": self.spec.name}

    def get_weight_version(self) -> str | None:
        return None if self.spec.policy_version is None else str(self.spec.policy_version)

    def load_policy_artifact(self, artifact: PolicyArtifactRef) -> None:
        self.load_calls.append(artifact)
        self.spec.model_path = artifact.path
        self.spec.policy_version = artifact.policy_version


def make_spec(*, port: int = 32000, model_path: str = "/tmp/model") -> SGLangEngineSpec:
    return SGLangEngineSpec(
        engine_id=0,
        name="regular-0",
        role="regular",
        host="127.0.0.1",
        port=port,
        bootstrap_port=None,
        model_path=model_path,
        tokenizer_path="/tmp/tokenizer",
        policy_version=None,
        server_args={"host": "0.0.0.0", "port": port, "model_path": model_path},
    )


def test_rollout_worker_initialize_and_load_artifact() -> None:
    FakeRuntime.created.clear()
    with TemporaryDirectory(prefix="openforge_rollout_worker_") as temp_dir:
        cfg = make_config(temp_dir)
        engine = cfg.rollout.resolve(cfg.cluster).engines[0]
        spec = RolloutWorkerSpec(
            cfg=cfg,
            engine=engine,
            host="10.0.0.8",
            port=30000,
        )
        worker = RolloutWorker()

        with mock.patch.object(rollout_worker_module, "SGLangRuntime", FakeRuntime):
            endpoint = worker.initialize(spec)
            updated = worker.load_policy_artifact(
                PolicyArtifactRef(step=11, policy_version=7, path="/tmp/policy.pt")
            )

    runtime = FakeRuntime.created[0]
    assert runtime.started is True
    assert endpoint.url == "http://10.0.0.8:30000"
    assert updated.policy_version == 7
    assert updated.model_path == "/tmp/policy.pt"

def test_rollout_worker_builds_regular_specs_and_checks_health() -> None:
    regular_cfg = make_config("/tmp")
    regular_engine = regular_cfg.rollout.resolve(regular_cfg.cluster).engines[0]
    worker = RolloutWorker()
    worker.spec = RolloutWorkerSpec(
        cfg=regular_cfg,
        engine=regular_engine,
        host="127.0.0.1",
        port=33000,
    )
    worker.engine = regular_engine
    worker.host = "127.0.0.1"
    worker.port = 33000
    worker.bootstrap_port = None
    worker.model_path = "regular-model"
    worker.policy_version = None
    regular_spec = worker._build_runtime_spec()

    assert regular_spec.role == "regular"
    assert regular_spec.server_args["dp_size"] == 1
    assert regular_spec.server_args["tp_size"] == 1

    worker.runtime = None
    assert_raises(
        RuntimeError,
        worker._runtime,
        match="has not been initialized",
    )

    worker.runtime = mock.Mock(
        endpoint=mock.Mock(
            return_value=mock.Mock(model_path=None, policy_version=8)
        )
    )
    assert_raises(
        RuntimeError,
        worker._sync_runtime_state,
        match="did not report model_path",
    )

    with (
        mock.patch(
            "openforge.rollout.worker.requests.get",
            return_value=mock.Mock(status_code=200),
        ),
        mock.patch("openforge.rollout.worker.ray.is_initialized", return_value=False),
        mock.patch("openforge.rollout.worker.socket.gethostbyname", return_value="127.0.0.2"),
        mock.patch("openforge.rollout.worker.socket.gethostname", return_value="localhost"),
        mock.patch("openforge.rollout.worker.get_free_port", return_value=34567),
    ):
        assert RolloutWorker._check_health(33000) is True
        assert RolloutWorker.resolve_node_ip_address() == "127.0.0.2"
        assert RolloutWorker.allocate_port(start=34000) == 34567

    with (
        mock.patch(
            "openforge.rollout.worker.requests.get",
            side_effect=rollout_worker_module.requests.RequestException("boom"),
        ),
        mock.patch("openforge.rollout.worker.ray.is_initialized", return_value=True),
        mock.patch(
            "openforge.rollout.worker.get_current_ray_node_ip_address",
            return_value="10.0.0.20",
        ),
    ):
        assert RolloutWorker._check_health(33000) is False
        assert RolloutWorker.resolve_node_ip_address() == "10.0.0.20"


def run_suite(artifacts_dir: Path) -> int:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")
    return run_named_tests(
        [
            (
                "rollout_worker_initialize_and_load_artifact",
                test_rollout_worker_initialize_and_load_artifact,
            ),
            (
                "rollout_worker_builds_regular_specs_and_checks_health",
                test_rollout_worker_builds_regular_specs_and_checks_health,
            ),
        ],
        summary_label="RolloutWorker",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone tests for RolloutWorker and SGLang runtime glue."
    )
    parser.parse_args()

    with TemporaryDirectory(prefix="openforge_rollout_worker_") as temp_dir:
        return run_suite(Path(temp_dir))


if __name__ == "__main__":
    raise SystemExit(main())
