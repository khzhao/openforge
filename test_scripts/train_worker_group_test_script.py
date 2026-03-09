#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone tests for TrainWorker and TrainWorkerGroup."""

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import ray
import torch
from tensordict import TensorDict

from openforge.configs.models import OpenForgeConfig
from openforge.policy.types import (
    DistributedUpdateSession,
    PolicyArtifactRef,
    TensorUpdateSession,
)
from openforge.train.backend import TrainBackend
from openforge.train.group import TrainWorkerGroup
from openforge.train.types import (
    CheckpointInfo,
    TrainStepResult,
    TrainWorkerSpec,
    TrainWorkerState,
)
from openforge.train.worker import TrainWorker
from script_test_utils import assert_raises, run_named_tests


def make_config(*, world_size: int = 4, cpus_per_worker: int = 1) -> OpenForgeConfig:
    return OpenForgeConfig.model_validate(
        {
            "data": {"backend": "test"},
            "gateway": {
                "host": "127.0.0.1",
                "port": 8000,
            },
            "model": {
                "model_name_or_path": "local-test-model",
                "tokenizer_name_or_path": "local-test-tokenizer",
            },
            "cluster": [
                {
                    "node_pool": "default",
                    "num_nodes": 1,
                    "num_gpus_per_node": max(5, world_size + 1),
                    "num_cpus_per_node": max(8, world_size * max(1, cpus_per_worker)),
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
                        "lr": 5e-3,
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
                "global_batch_size": world_size,
                "mini_batch_size": world_size,
                "micro_batch_size": 1,
                "checkpoints_dir": "/tmp/openforge_train_worker_group_test",
                "cpus_per_worker": cpus_per_worker,
                "parallelism": {
                    "data_parallel_size": world_size,
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
                    "max_new_tokens": 32,
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


def make_batch(*, num_tokens: int = 4, token_offset: int = 0) -> TensorDict:
    tokens = torch.arange(
        1 + token_offset,
        1 + token_offset + num_tokens,
        dtype=torch.long,
    )
    return TensorDict(
        {
            "tokens": tokens,
            "cu_seqlens": torch.tensor([0, num_tokens], dtype=torch.int32),
            "loss_mask": torch.ones(num_tokens, dtype=torch.float32),
            "rewards": torch.ones(num_tokens, dtype=torch.float32),
            "position_ids": torch.arange(num_tokens, dtype=torch.long),
        },
        batch_size=[],
    )

class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.device = "cpu"
        self.rank = -1

    def initialize(self, spec: TrainWorkerSpec) -> None:
        self.rank = spec.rank
        self.calls.append(("initialize", spec.rank, spec.world_size))

    def zero_grad(self) -> None:
        self.calls.append("zero_grad")

    def forward(self, batch: TensorDict) -> TensorDict:
        token_count = int(batch.get("tokens").numel())
        self.calls.append(("forward", token_count))
        return TensorDict({"loss": torch.tensor(1.0)}, batch_size=[])

    def backward(self, forward_out: TensorDict) -> None:
        self.calls.append(("backward", float(forward_out.get("loss"))))

    def no_sync(self):
        self.calls.append("no_sync")
        return nullcontext()

    def step_optimizer(self, *, global_step: int | None = None) -> TrainStepResult:
        self.calls.append(("step_optimizer", global_step))
        return TrainStepResult(
            rank=self.rank,
            global_step=global_step,
            metrics={"lr": 1.0, "token_count": 4.0},
        )

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ) -> CheckpointInfo:
        self.calls.append(("save_checkpoint", step, policy_version, save_optimizer))
        return CheckpointInfo(
            step=step,
            policy_version=policy_version,
            path=f"/tmp/rank_{self.rank}_step_{step}_policy_{policy_version}.pt",
        )

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ) -> CheckpointInfo | None:
        self.calls.append(("load_checkpoint", latest, step, load_optimizer))
        return CheckpointInfo(step=101, policy_version=201, path="/tmp/loaded.pt")

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ) -> PolicyArtifactRef | None:
        self.calls.append(("export_policy_artifact", step, policy_version))
        if self.rank != 0:
            return None
        return PolicyArtifactRef(
            step=step,
            policy_version=policy_version,
            path=f"/tmp/export_step_{step}_policy_{policy_version}.pt",
        )

    def push_tensor_update(
        self,
        session: TensorUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        self.calls.append(("push_tensor_update", session.session_id, step, policy_version))

    def push_distributed_update(
        self,
        session: DistributedUpdateSession,
        *,
        step: int,
        policy_version: int,
    ) -> None:
        self.calls.append(
            ("push_distributed_update", session.session_id, step, policy_version)
        )

    def sleep(self) -> None:
        self.calls.append("sleep")

    def wakeup(self) -> None:
        self.calls.append("wakeup")

    def clear_memory(self) -> None:
        self.calls.append("clear_memory")

    def shutdown(self) -> None:
        self.calls.append("shutdown")


class InvalidReturnBackend(FakeBackend):
    def step_optimizer(self, *, global_step: int | None = None):
        self.calls.append(("invalid_step_optimizer", global_step))
        return {"legacy_metric": 2.0}

    def save_checkpoint(
        self,
        *,
        step: int,
        policy_version: int,
        save_optimizer: bool = True,
    ):
        self.calls.append(("invalid_save_checkpoint", step, policy_version, save_optimizer))
        return "/tmp/legacy-checkpoint.pt"

    def load_checkpoint(
        self,
        *,
        latest: bool = True,
        step: int | None = None,
        load_optimizer: bool = True,
    ):
        self.calls.append(("invalid_load_checkpoint", latest, step, load_optimizer))
        return (12, 34)

    def export_policy_artifact(
        self,
        *,
        step: int,
        policy_version: int,
    ):
        self.calls.append(("invalid_export_policy_artifact", step, policy_version))
        return "/tmp/legacy-artifact.pt"


class FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeGroupWorker:
    def __init__(
        self,
        *,
        rank: int,
        checkpoint: CheckpointInfo | None = None,
        artifact: PolicyArtifactRef | None = None,
    ) -> None:
        self.rank = rank
        self.status = FakeRemoteMethod(
            lambda: TrainWorkerState(
                rank=rank,
                world_size=2,
                backend="fsdp2",
                device="cpu",
                initialized=True,
                sleeping=False,
            )
        )
        self.step = FakeRemoteMethod(
            lambda microbatches, *, global_step=None: TrainStepResult(
                rank=rank,
                global_step=global_step,
                metrics={"num_microbatches": float(len(microbatches))},
            )
        )
        self.save_checkpoint = FakeRemoteMethod(
            lambda *, step, policy_version, save_optimizer=True: CheckpointInfo(
                step=step,
                policy_version=policy_version,
                path=f"/tmp/rank_{rank}_step_{step}_policy_{policy_version}.pt",
            )
        )
        self.load_checkpoint = FakeRemoteMethod(
            lambda *, latest=True, step=None, load_optimizer=True: checkpoint
        )
        self.export_policy_artifact = FakeRemoteMethod(
            lambda *, step, policy_version: artifact
        )
        self.push_tensor_update = FakeRemoteMethod(
            lambda session, *, step, policy_version: (rank, session, step, policy_version)
        )
        self.push_distributed_update = FakeRemoteMethod(
            lambda session, *, step, policy_version: (rank, session, step, policy_version)
        )
        self.sleep = FakeRemoteMethod(lambda: rank)
        self.wakeup = FakeRemoteMethod(lambda: rank)
        self.clear_memory = FakeRemoteMethod(lambda: rank)
        self.shutdown = FakeRemoteMethod(lambda: rank)


def test_train_worker_local() -> None:
    cfg = make_config(world_size=2, cpus_per_worker=0)
    spec = TrainWorkerSpec(
        cfg=cfg,
        rank=1,
        world_size=2,
        master_addr="127.0.0.1",
        master_port=29599,
    )
    tensor_session = TensorUpdateSession(
        session_id="tensor-1",
        policy_version=5,
        load_format="flattened_bucket",
        engine_ids=[0],
    )
    distributed_session = DistributedUpdateSession(
        session_id="dist-1",
        policy_version=6,
        load_format="flattened_bucket",
        engine_ids=[0],
        master_addr="127.0.0.1",
        master_port=29600,
        group_name="dist-group",
        world_size=2,
        backend="gloo",
        rank_offsets={0: 1},
    )

    with mock.patch("openforge.train.worker.FSDP2Backend", FakeBackend):
        worker = TrainWorker()
        state = worker.initialize(spec)
        step = worker.step([make_batch()], global_step=7)
        checkpoint = worker.save_checkpoint(step=3, policy_version=11)
        restored = worker.load_checkpoint()
        artifact = worker.export_policy_artifact(step=9, policy_version=15)
        worker.push_tensor_update(tensor_session, step=9, policy_version=15)
        worker.push_distributed_update(distributed_session, step=9, policy_version=15)
        worker.sleep()
        worker.wakeup()
        worker.clear_memory()
        worker.shutdown()

    assert state.rank == 1
    assert step.rank == 1
    assert step.global_step == 7
    assert checkpoint.path.endswith("policy_11.pt")
    assert restored is not None and restored.step == 101
    assert artifact is None


def test_train_worker_rejects_invalid_inputs_and_unsupported_backend() -> None:
    cfg = make_config(world_size=1, cpus_per_worker=0)
    spec = TrainWorkerSpec(
        cfg=cfg,
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=29599,
    )

    with mock.patch("openforge.train.worker.FSDP2Backend", FakeBackend):
        worker = TrainWorker()
        worker.initialize(spec)
        assert_raises(
            ValueError,
            worker.step,
            [],
            global_step=1,
            match="step requires at least one microbatch",
        )

    bad_cfg = cfg.model_copy(deep=True)
    bad_cfg.train.backend = "unknown"
    bad_spec = TrainWorkerSpec(
        cfg=bad_cfg,
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=29599,
    )
    assert_raises(
        ValueError,
        TrainWorker().initialize,
        bad_spec,
        match="Unsupported backend",
    )


def test_train_backend_is_abstract() -> None:
    assert_raises(TypeError, TrainBackend)


def test_train_worker_rejects_legacy_backend_return_shapes() -> None:
    cfg = make_config(world_size=1, cpus_per_worker=0)
    spec = TrainWorkerSpec(
        cfg=cfg,
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=29599,
    )

    with mock.patch("openforge.train.worker.FSDP2Backend", InvalidReturnBackend):
        worker = TrainWorker()
        worker.initialize(spec)
        assert_raises(
            TypeError,
            worker.step,
            [make_batch()],
            global_step=5,
            match="step_optimizer must return TrainStepResult",
        )
        assert_raises(
            TypeError,
            worker.save_checkpoint,
            step=7,
            policy_version=8,
            match="save_checkpoint must return CheckpointInfo",
        )
        assert_raises(
            TypeError,
            worker.load_checkpoint,
            match="load_checkpoint must return CheckpointInfo \\| None",
        )
        assert_raises(
            TypeError,
            worker.export_policy_artifact,
            step=9,
            policy_version=10,
            match="export_policy_artifact must return PolicyArtifactRef \\| None",
        )


def fake_initialize(self: TrainWorker, spec: TrainWorkerSpec):
    self._state = TrainWorkerState(
        rank=spec.rank,
        world_size=spec.world_size,
        backend=spec.cfg.train.backend,
        device="cpu",
        initialized=True,
        sleeping=False,
    )
    self.spec = spec
    return self._state


def fake_status(self: TrainWorker):
    return self._state


def fake_step(self: TrainWorker, microbatches, *, global_step=None):
    return TrainStepResult(
        rank=self._state.rank,
        global_step=global_step,
        metrics={"num_microbatches": float(len(microbatches))},
    )


def fake_save_checkpoint(self: TrainWorker, *, step: int, policy_version: int, save_optimizer: bool = True):
    _ = save_optimizer
    return CheckpointInfo(
        step=step,
        policy_version=policy_version,
        path=f"/tmp/rank_{self._state.rank}_step_{step}_policy_{policy_version}.pt",
    )


def fake_load_checkpoint(self: TrainWorker, *, latest: bool = True, step: int | None = None, load_optimizer: bool = True):
    _ = latest, step, load_optimizer
    return CheckpointInfo(step=8, policy_version=21, path=f"/tmp/rank_{self._state.rank}.pt")


def fake_export_policy_artifact(self: TrainWorker, *, step: int, policy_version: int):
    if self._state.rank != 0:
        return None
    return PolicyArtifactRef(
        step=step,
        policy_version=policy_version,
        path=f"/tmp/export_step_{step}_policy_{policy_version}.pt",
    )


def fake_push_tensor_update(self: TrainWorker, session, *, step: int, policy_version: int):
    _ = session, step, policy_version


def fake_push_distributed_update(self: TrainWorker, session, *, step: int, policy_version: int):
    _ = session, step, policy_version


def fake_sleep(self: TrainWorker) -> None:
    self._state = replace(self._state, sleeping=True)


def fake_wakeup(self: TrainWorker) -> None:
    self._state = replace(self._state, sleeping=False)


def fake_clear_memory(self: TrainWorker) -> None:
    return None


def fake_shutdown(self: TrainWorker) -> None:
    self._state = replace(self._state, initialized=False)


def test_train_worker_group_validates_inputs_and_handles_missing_artifacts() -> None:
    group = object.__new__(TrainWorkerGroup)
    group._world_size = 2
    group._workers = [
        FakeGroupWorker(rank=0, checkpoint=None, artifact=None),
        FakeGroupWorker(rank=1, checkpoint=None, artifact=None),
    ]
    group._pg = "pg"

    assert_raises(
        ValueError,
        group.step,
        [[make_batch()]],
        global_step=3,
        match="Expected one microbatch sequence",
    )

    with mock.patch("openforge.train.group.ray.get", side_effect=lambda value: value):
        assert group.load_checkpoint() is None
        assert [state.rank for state in group.states()] == [0, 1]
        assert_raises(
            RuntimeError,
            group.export_policy_artifact,
            step=4,
            policy_version=9,
            match="did not return a policy artifact",
        )


def test_train_worker_group_creates_workers_and_placement_groups() -> None:
    group = object.__new__(TrainWorkerGroup)
    group._world_size = 2
    group._num_gpus_per_worker = 1
    group._num_cpus_per_worker = 3
    group._workers = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    class FakeRemoteWorker:
        def __init__(self) -> None:
            self.options_calls: list[dict[str, object]] = []

        def options(self, **kwargs):
            self.options_calls.append(kwargs)
            return self

        def remote(self):
            return {"worker_index": len(self.options_calls) - 1}

    fake_pg = FakePlacementGroup()
    fake_remote_worker = FakeRemoteWorker()

    with (
        mock.patch("openforge.train.group.placement_group", return_value=fake_pg) as placement_group,
        mock.patch("openforge.train.group.ray.get", side_effect=lambda value: value),
        mock.patch("openforge.train.group.ray.remote", return_value=fake_remote_worker),
        mock.patch(
            "openforge.train.group.PlacementGroupSchedulingStrategy",
            side_effect=lambda **kwargs: kwargs,
        ),
    ):
        created_pg = group._create_placement_group("PACK")
        group._pg = created_pg
        group._create_workers()

    placement_group.assert_called_once()
    assert created_pg is fake_pg
    assert group._workers == [{"worker_index": 0}, {"worker_index": 1}]
    assert fake_remote_worker.options_calls[0]["scheduling_strategy"][
        "placement_group_bundle_index"
    ] == 0
    assert fake_remote_worker.options_calls[1]["scheduling_strategy"][
        "placement_group_bundle_index"
    ] == 1


def test_train_worker_group_ray(artifacts_dir: Path) -> None:
    cfg = make_config(world_size=4, cpus_per_worker=1)
    batches = [[make_batch(num_tokens=4 + rank)] for rank in range(4)]
    tensor_session = TensorUpdateSession(
        session_id="tensor-2",
        policy_version=10,
        load_format="flattened_bucket",
        engine_ids=[0],
        transport_metadata={"rollout_workers": [], "rollout_engines": []},
    )
    distributed_session = DistributedUpdateSession(
        session_id="dist-2",
        policy_version=11,
        load_format="flattened_bucket",
        engine_ids=[0],
        master_addr="127.0.0.1",
        master_port=29601,
        group_name="dist-group-2",
        world_size=2,
        backend="gloo",
        rank_offsets={0: 1},
        transport_metadata={"rollout_workers": [], "rollout_engines": []},
    )

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(TrainWorker, "initialize", fake_initialize))
        stack.enter_context(mock.patch.object(TrainWorker, "status", fake_status))
        stack.enter_context(mock.patch.object(TrainWorker, "step", fake_step))
        stack.enter_context(mock.patch.object(TrainWorker, "save_checkpoint", fake_save_checkpoint))
        stack.enter_context(mock.patch.object(TrainWorker, "load_checkpoint", fake_load_checkpoint))
        stack.enter_context(mock.patch.object(TrainWorker, "export_policy_artifact", fake_export_policy_artifact))
        stack.enter_context(mock.patch.object(TrainWorker, "push_tensor_update", fake_push_tensor_update))
        stack.enter_context(mock.patch.object(TrainWorker, "push_distributed_update", fake_push_distributed_update))
        stack.enter_context(mock.patch.object(TrainWorker, "sleep", fake_sleep))
        stack.enter_context(mock.patch.object(TrainWorker, "wakeup", fake_wakeup))
        stack.enter_context(mock.patch.object(TrainWorker, "clear_memory", fake_clear_memory))
        stack.enter_context(mock.patch.object(TrainWorker, "shutdown", fake_shutdown))
        with TemporaryDirectory(prefix="ofray_") as ray_temp_dir:
            ray.init(
                num_cpus=8,
                num_gpus=4,
                include_dashboard=False,
                ignore_reinit_error=False,
                local_mode=True,
                log_to_driver=False,
                _temp_dir=ray_temp_dir,
            )
            try:
                group = TrainWorkerGroup(
                    cfg,
                    master_addr="127.0.0.1",
                    master_port=29601,
                    strategy="PACK",
                )
                states = group.initialize()
                assert [state.rank for state in states] == [0, 1, 2, 3]

                results = group.step(batches, global_step=5)
                assert [result.rank for result in results] == [0, 1, 2, 3]
                assert all(result.global_step == 5 for result in results)

                checkpoints = group.save_checkpoint(step=8, policy_version=21)
                assert len(checkpoints) == 4
                assert checkpoints[0].path.endswith("policy_21.pt")

                checkpoint_info = group.load_checkpoint()
                assert checkpoint_info is not None and checkpoint_info.step == 8

                artifact = group.export_policy_artifact(step=14, policy_version=35)
                assert artifact.path.endswith("policy_35.pt")

                group.push_tensor_update(tensor_session, step=14, policy_version=35)
                group.push_distributed_update(distributed_session, step=15, policy_version=36)
                group.sleep()
                group.wakeup()
                group.clear_memory()
                group.shutdown()
            finally:
                ray.shutdown()


def run_suite(artifacts_dir: Path) -> int:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")
    return run_named_tests(
        [
            ("train_worker_local", test_train_worker_local),
            (
                "train_worker_rejects_invalid_inputs_and_unsupported_backend",
                test_train_worker_rejects_invalid_inputs_and_unsupported_backend,
            ),
            ("train_backend_is_abstract", test_train_backend_is_abstract),
            (
                "train_worker_rejects_legacy_backend_return_shapes",
                test_train_worker_rejects_legacy_backend_return_shapes,
            ),
            (
                "train_worker_group_validates_inputs_and_handles_missing_artifacts",
                test_train_worker_group_validates_inputs_and_handles_missing_artifacts,
            ),
            (
                "train_worker_group_creates_workers_and_placement_groups",
                test_train_worker_group_creates_workers_and_placement_groups,
            ),
            ("train_worker_group_ray", lambda: test_train_worker_group_ray(artifacts_dir)),
        ],
        summary_label="TrainWorker",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone tests for TrainWorker and TrainWorkerGroup."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.artifacts_dir is None:
        with TemporaryDirectory(prefix="openforge_train_worker_group_") as temp_dir:
            return run_suite(Path(temp_dir))

    return run_suite(args.artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())
