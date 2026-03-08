#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone tests for RolloutWorker and SGLang runtime glue."""

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from openforge.configs.models import OpenForgeConfig
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
    WeightBucketMeta,
)
from openforge.rollout.sglang.client import SGLangControlClient
from openforge.rollout.sglang.runtime import SGLangRuntime
from openforge.rollout.sglang.spec import SGLangEngineSpec
from openforge.rollout.worker import RolloutWorker
from openforge.rollout.types import RolloutWorkerSpec
import openforge.rollout.worker as rollout_worker_module
from script_test_utils import assert_raises, run_named_tests


def make_config(
    checkpoints_dir: str,
    *,
    engine_topology: str = "regular",
) -> OpenForgeConfig:
    rollout_engines: list[dict[str, Any]]
    if engine_topology == "pd":
        rollout_engines = [
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
                    "node_pool": "default",
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
                    "node_pool": "default",
                    "strategy": "PACK",
                },
            },
        ]
    else:
        rollout_engines = [
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
        ]

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
                "engine_topology": engine_topology,
                "engines": rollout_engines,
            },
        }
    )


class FakeRuntime:
    created: list["FakeRuntime"] = []

    def __init__(self, spec: SGLangEngineSpec) -> None:
        self.spec = spec
        self.started = False
        self.stopped = False
        self.tensor_begin_calls: list[Any] = []
        self.tensor_apply_calls: list[Any] = []
        self.tensor_finish_calls: list[Any] = []
        self.dist_begin_calls: list[Any] = []
        self.dist_apply_calls: list[Any] = []
        self.dist_finish_calls: list[Any] = []
        self.abort_calls: list[str] = []
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

    def begin_tensor_update(self, session: TensorUpdateSession) -> None:
        self.tensor_begin_calls.append(session)

    def apply_tensor_bucket(self, **kwargs: Any) -> None:
        self.tensor_apply_calls.append(kwargs)
        self.spec.policy_version = kwargs["policy_version"]

    def finish_tensor_update(self, session: TensorUpdateSession) -> None:
        self.tensor_finish_calls.append(session)
        self.spec.policy_version = session.policy_version

    def begin_distributed_update(self, session: DistributedUpdateSession) -> None:
        self.dist_begin_calls.append(session)

    def apply_distributed_bucket(self, **kwargs: Any) -> None:
        self.dist_apply_calls.append(kwargs)
        self.spec.policy_version = kwargs["policy_version"]

    def finish_distributed_update(self, session: DistributedUpdateSession) -> None:
        self.dist_finish_calls.append(session)
        self.spec.policy_version = session.policy_version

    def abort_update(self, *, session_id: str) -> None:
        self.abort_calls.append(session_id)


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
            model_path="test-model",
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


def test_rollout_worker_applies_tensor_and_distributed_sessions() -> None:
    FakeRuntime.created.clear()
    with TemporaryDirectory(prefix="openforge_rollout_worker_") as temp_dir:
        cfg = make_config(temp_dir)
        engine = cfg.rollout.resolve(cfg.cluster).engines[0]
        spec = RolloutWorkerSpec(
            cfg=cfg,
            engine=engine,
            host="10.0.0.9",
            port=30000,
            model_path="test-model",
        )
        worker = RolloutWorker()
        tensor_session = TensorUpdateSession(
            session_id="tensor-1",
            policy_version=5,
            load_format="flattened_bucket",
            engine_ids=[engine.engine_id],
        )
        dist_session = DistributedUpdateSession(
            session_id="dist-1",
            policy_version=6,
            load_format="flattened_bucket",
            engine_ids=[engine.engine_id],
            master_addr="10.0.0.2",
            master_port=41234,
            group_name="rollout-sync-6",
            world_size=2,
            backend="gloo",
            rank_offsets={engine.engine_id: 1},
        )

        with mock.patch.object(rollout_worker_module, "SGLangRuntime", FakeRuntime):
            worker.initialize(spec)
            worker.begin_tensor_update(tensor_session)
            worker.apply_tensor_bucket(
                serialized_named_tensors=["bucket-a"],
                load_format="flattened_bucket",
                policy_version=5,
            )
            tensor_endpoint = worker.finish_tensor_update(tensor_session)

            worker.begin_distributed_update(dist_session)
            worker.apply_distributed_bucket(
                bucket=WeightBucketMeta(
                    names=["w1"],
                    dtypes=["float16"],
                    shapes=[[2, 3]],
                ),
                policy_version=6,
                load_format="flattened_bucket",
                group_name="rollout-sync-6",
            )
            dist_endpoint = worker.finish_distributed_update(dist_session)
            worker.abort_update(session_id="dist-1")

    runtime = FakeRuntime.created[0]
    assert runtime.tensor_begin_calls[0].session_id == "tensor-1"
    assert runtime.tensor_apply_calls[0]["policy_version"] == 5
    assert tensor_endpoint.policy_version == 5
    assert runtime.dist_begin_calls[0].session_id == "dist-1"
    assert runtime.dist_apply_calls[0]["group_name"] == "rollout-sync-6"
    assert dist_endpoint.policy_version == 6
    assert runtime.abort_calls == ["dist-1"]


def test_rollout_worker_placeholder_paths_and_helper_methods() -> None:
    with TemporaryDirectory(prefix="openforge_rollout_worker_") as temp_dir:
        cfg = make_config(temp_dir)
        engine = replace(
            cfg.rollout.resolve(cfg.cluster).engines[0],
            role="placeholder",
        )
        spec = RolloutWorkerSpec(
            cfg=cfg,
            engine=engine,
            host="10.0.0.10",
            port=30010,
            model_path="placeholder-model",
            policy_version=4,
        )
        worker = RolloutWorker()
        endpoint = worker.initialize(spec)

    assert endpoint.role == "placeholder"
    assert endpoint.url is None
    assert endpoint.policy_version == 4
    assert worker.is_healthy() is False
    assert worker.flush_cache() is True
    assert worker.get_model_info() == {}
    assert worker.get_server_info() == {}
    assert worker.get_weight_version() is None
    worker.pause_generation()
    worker.continue_generation()
    worker.begin_tensor_update(
        TensorUpdateSession(
            session_id="tensor-placeholder",
            policy_version=9,
            load_format="flattened_bucket",
            engine_ids=[0],
        )
    )
    worker.apply_tensor_bucket(
        serialized_named_tensors=["ignored"],
        load_format="flattened_bucket",
        policy_version=9,
    )
    worker.begin_distributed_update(
        DistributedUpdateSession(
            session_id="dist-placeholder",
            policy_version=9,
            load_format="flattened_bucket",
            engine_ids=[0],
            master_addr="127.0.0.1",
            master_port=1234,
            group_name="placeholder",
            world_size=2,
            backend="gloo",
            rank_offsets={0: 1},
        )
    )
    worker.apply_distributed_bucket(
        bucket=WeightBucketMeta(names=["x"], dtypes=["float32"], shapes=[[1]]),
        policy_version=9,
        load_format="flattened_bucket",
        group_name="placeholder",
    )
    worker.abort_update(session_id="ignored")
    worker.shutdown()


def test_rollout_worker_builds_pd_specs_and_checks_health() -> None:
    with TemporaryDirectory(prefix="openforge_rollout_worker_pd_") as temp_dir:
        cfg = make_config(temp_dir, engine_topology="pd")
        resolved = cfg.rollout.resolve(cfg.cluster).engines
        prefill_engine = next(engine for engine in resolved if engine.role == "prefill")
        decode_engine = next(engine for engine in resolved if engine.role == "decode")

        prefill_worker = RolloutWorker()
        prefill_worker.spec = RolloutWorkerSpec(
            cfg=cfg,
            engine=prefill_engine,
            host="10.0.0.11",
            port=31000,
            bootstrap_port=31001,
            model_path="prefill-model",
            policy_version=7,
        )
        prefill_worker.engine = prefill_engine
        prefill_worker.host = "10.0.0.11"
        prefill_worker.port = 31000
        prefill_worker.bootstrap_port = 31001
        prefill_worker.model_path = "prefill-model"
        prefill_worker.policy_version = 7
        prefill_spec = prefill_worker._build_runtime_spec()

        decode_worker = RolloutWorker()
        decode_worker.spec = RolloutWorkerSpec(
            cfg=cfg,
            engine=decode_engine,
            host="10.0.0.12",
            port=32000,
            bootstrap_port=None,
            model_path="decode-model",
            policy_version=None,
        )
        decode_worker.engine = decode_engine
        decode_worker.host = "10.0.0.12"
        decode_worker.port = 32000
        decode_worker.bootstrap_port = None
        decode_worker.model_path = "decode-model"
        decode_worker.policy_version = None
        decode_spec = decode_worker._build_runtime_spec()

    assert prefill_spec.server_args["disaggregation_mode"] == "prefill"
    assert prefill_spec.server_args["disaggregation_bootstrap_port"] == 31001
    assert prefill_spec.server_args["disaggregation_decode_tp"] == 1
    assert prefill_spec.server_args["disaggregation_decode_dp"] == 1
    assert decode_spec.server_args["disaggregation_mode"] == "decode"
    assert decode_spec.server_args["disaggregation_prefill_pp"] == 1

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
    worker.runtime = None

    assert_raises(
        RuntimeError,
        worker._runtime,
        match="has not been initialized",
    )
    assert_raises(
        ValueError,
        worker._role_parallelism,
        "decode",
        match="Missing rollout role decode",
    )

    worker.runtime = mock.Mock(
        endpoint=mock.Mock(
            return_value=mock.Mock(model_path=None, policy_version=8)
        )
    )
    worker._sync_runtime_state()
    assert worker.model_path == "regular-model"
    assert worker.policy_version == 8

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
                "rollout_worker_applies_tensor_and_distributed_sessions",
                test_rollout_worker_applies_tensor_and_distributed_sessions,
            ),
            (
                "rollout_worker_placeholder_paths_and_helper_methods",
                test_rollout_worker_placeholder_paths_and_helper_methods,
            ),
            (
                "rollout_worker_builds_pd_specs_and_checks_health",
                test_rollout_worker_builds_pd_specs_and_checks_health,
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
