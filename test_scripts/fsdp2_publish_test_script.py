#!/usr/bin/env python3
# Copyright 2026 openforge

"""Focused unit tests for train.fsdp2 publish helpers."""

import argparse
from types import SimpleNamespace
from unittest import mock

import torch

import openforge.train.fsdp2.publish as publish_module
from openforge.configs.models import OpenForgeConfig
from openforge.policy.types import DistributedUpdateSession, TensorUpdateSession
from openforge.train.fsdp2.publish import DistributedWeightUpdater, TensorWeightUpdater, WeightUpdater
from script_test_utils import assert_raises, run_named_tests


def make_config(*, update_weight_buffer_size: int = 32) -> OpenForgeConfig:
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
                    "update_weight_buffer_size": update_weight_buffer_size,
                },
                "global_batch_size": 1,
                "mini_batch_size": 1,
                "micro_batch_size": 1,
                "checkpoints_dir": "/tmp/openforge_publish_test",
                "cpus_per_worker": 1,
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


class FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeRolloutWorker:
    def __init__(self) -> None:
        self.begin_tensor_calls = []
        self.finish_tensor_calls = []
        self.apply_tensor_calls = []
        self.begin_distributed_calls = []
        self.finish_distributed_calls = []
        self.apply_distributed_calls = []

        self.begin_tensor_update = FakeRemoteMethod(
            lambda session: self.begin_tensor_calls.append(session)
        )
        self.finish_tensor_update = FakeRemoteMethod(
            lambda session: self.finish_tensor_calls.append(session)
        )
        self.apply_tensor_bucket = FakeRemoteMethod(
            lambda **kwargs: self.apply_tensor_calls.append(kwargs)
        )
        self.begin_distributed_update = FakeRemoteMethod(
            lambda session: self.begin_distributed_calls.append(session)
        )
        self.finish_distributed_update = FakeRemoteMethod(
            lambda session: self.finish_distributed_calls.append(session)
        )
        self.apply_distributed_bucket = FakeRemoteMethod(
            lambda **kwargs: self.apply_distributed_calls.append(kwargs)
        )


class RecordingUpdater(WeightUpdater):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[object] = []
        self.fail_after_first_bucket = False

    def connect_rollout_engines(self, *, rollout_workers, rollout_engines) -> None:
        self.events.append(("connect", len(rollout_workers), len(rollout_engines)))

    def disconnect_rollout_engines(self) -> None:
        self.events.append("disconnect")

    def before_update(self) -> None:
        self.events.append("before")

    def after_update(self) -> None:
        self.events.append("after")

    def update_bucket_weights(self, named_tensors):
        self.events.append([name for name, _ in named_tensors])
        if self.fail_after_first_bucket and len(self.events) >= 3:
            raise RuntimeError("bucket failure")


class FakeDTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.device_mesh = SimpleNamespace(ndim=1)

    def to(self, device=None):
        _ = device
        return self

    def redistribute(self, *, placements, async_op):
        _ = placements, async_op
        return SimpleNamespace(to_local=lambda: self.tensor)


class FakeFlattenedTensorBucket:
    def __init__(self, *, named_tensors):
        self.named_tensors = named_tensors

    def get_flattened_tensor(self):
        return torch.tensor([1.0], dtype=torch.float32)


def make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 2, bias=False),
    )


def test_weight_updater_buckets_and_cleans_up_on_error() -> None:
    updater = RecordingUpdater(
        make_config(update_weight_buffer_size=8),
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    updater.update_weights(policy_version=3)
    assert updater.weight_version == 3
    assert updater.events[0] == "before"
    assert updater.events[-2:] == ["after", "disconnect"]

    failing = RecordingUpdater(
        make_config(update_weight_buffer_size=8),
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    failing.fail_after_first_bucket = True
    assert_raises(
        RuntimeError,
        failing.update_weights,
        policy_version=4,
        match="bucket failure",
    )
    assert failing.events[-2:] == ["after", "disconnect"]


def test_weight_updater_prepare_tensor_handles_dtensor_and_waitable() -> None:
    updater = RecordingUpdater(
        make_config(),
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    local_tensor = torch.ones(2)

    with (
        mock.patch.object(publish_module, "DTensor", FakeDTensor),
        mock.patch.object(publish_module, "Replicate", return_value="replicate"),
    ):
        assert torch.equal(updater._prepare_tensor(FakeDTensor(local_tensor)), local_tensor)

    waited_tensor = torch.zeros(1)
    updater.wait_and_update_bucket(
        [
            ("ready", SimpleNamespace(wait=lambda: waited_tensor)),
            ("plain", torch.ones(1)),
        ]
    )
    assert updater.events[-1] == ["ready", "plain"]


def test_tensor_weight_updater_connects_groups_and_runs_rpc_flow() -> None:
    cfg = make_config()
    worker_a = FakeRolloutWorker()
    worker_b = FakeRolloutWorker()
    updater = TensorWeightUpdater(
        cfg,
        make_model(),
        rank=1,
        world_size=4,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )

    with mock.patch.object(publish_module.dist, "new_group", side_effect=["g0", "g1"]):
        updater.connect_rollout_engines(
            rollout_workers=[worker_a, worker_b],
            rollout_engines=[
                SimpleNamespace(gpus_per_engine=1),
                SimpleNamespace(gpus_per_engine=2),
            ],
        )

    assert updater._rollout_worker is worker_b
    assert updater._gather_group == "g1"
    assert updater._gather_src_rank == 1

    updater.weight_version = 9
    with mock.patch.object(publish_module.ray, "get", side_effect=lambda value: value):
        updater.before_update()
        updater.after_update()

    assert worker_b.begin_tensor_calls[0].policy_version == 9
    assert worker_b.finish_tensor_calls[0].policy_version == 9

    src_updater = TensorWeightUpdater(
        cfg,
        make_model(),
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    src_updater._rollout_worker = worker_a
    src_updater._gather_group = "gather"
    src_updater._gather_group_size = 2
    src_updater._gather_src_rank = 0
    src_updater.weight_version = 11

    def fake_gather_object(*, obj, object_gather_list, dst, group):
        _ = obj, dst, group
        object_gather_list[0] = ["rank0-a", "rank0-b"]
        object_gather_list[1] = ["rank1-a", "rank1-b"]

    with (
        mock.patch(
            "openforge.rollout.sglang.serialization.serialize_named_tensors_for_sglang",
            return_value=["ignored-a", "ignored-b"],
        ),
        mock.patch.object(publish_module, "monkey_patch_torch_reductions"),
        mock.patch.object(publish_module.dist, "gather_object", side_effect=fake_gather_object),
        mock.patch.object(publish_module.ray, "get", side_effect=lambda value: value),
    ):
        src_updater.update_bucket_weights([("w", torch.ones(1))])

    assert worker_a.apply_tensor_calls[0]["serialized_named_tensors"] == [
        "rank0-a",
        "rank1-a",
    ]
    assert worker_a.apply_tensor_calls[1]["serialized_named_tensors"] == [
        "rank0-b",
        "rank1-b",
    ]

    too_large = TensorWeightUpdater(
        cfg,
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    assert_raises(
        ValueError,
        too_large.connect_rollout_engines,
        rollout_workers=[worker_a, worker_b],
        rollout_engines=[
            SimpleNamespace(gpus_per_engine=1),
            SimpleNamespace(gpus_per_engine=1),
        ],
        match="requires rollout GPUs",
    )


def test_tensor_weight_updater_disconnects_groups() -> None:
    updater = TensorWeightUpdater(
        make_config(),
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    updater._created_groups = ["g0", "g1"]
    with mock.patch.object(publish_module.dist, "destroy_process_group") as destroy:
        updater.disconnect_rollout_engines()
    assert destroy.call_count == 2
    assert updater._created_groups == []


def test_distributed_weight_updater_session_and_rpc_flow() -> None:
    cfg = make_config()
    worker = FakeRolloutWorker()
    session = DistributedUpdateSession(
        session_id="dist-session",
        policy_version=13,
        load_format="flattened_bucket",
        engine_ids=[0],
        master_addr="127.0.0.1",
        master_port=45000,
        group_name="dist-sync",
        world_size=3,
        backend="gloo",
        rank_offsets={0: 1},
    )
    updater = DistributedWeightUpdater(
        cfg,
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
        session=session,
    )
    updater.connect_rollout_engines(
        rollout_workers=[worker],
        rollout_engines=[SimpleNamespace(gpus_per_engine=2)],
    )

    with (
        mock.patch(
            "sglang.srt.utils.common.init_custom_process_group",
            return_value="process-group",
        ),
        mock.patch.object(publish_module.ray, "get", side_effect=lambda value: value),
        mock.patch.object(publish_module.dist, "destroy_process_group") as destroy_group,
    ):
        updater.before_update()
        updater.after_update()

    assert updater._group_name is None
    assert worker.begin_distributed_calls == []
    destroy_group.assert_called_once_with("process-group")

    implicit = DistributedWeightUpdater(
        cfg,
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    implicit.connect_rollout_engines(
        rollout_workers=[worker],
        rollout_engines=[SimpleNamespace(gpus_per_engine=2)],
    )
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value.getsockname.return_value = ("", 46000)
    fake_socket.__exit__.return_value = False

    with (
        mock.patch.object(publish_module.socket, "socket", return_value=fake_socket),
        mock.patch(
            "sglang.srt.utils.common.init_custom_process_group",
            return_value="process-group-2",
        ),
        mock.patch.object(publish_module.ray, "get", side_effect=lambda value: value),
        mock.patch.object(publish_module.dist, "destroy_process_group"),
    ):
        implicit.weight_version = 17
        implicit.before_update()
        implicit.after_update()

    assert worker.begin_distributed_calls[0].group_name == "openforge-rollout-sync-17"
    assert worker.finish_distributed_calls[0].policy_version == 17


def test_distributed_weight_updater_broadcasts_bucket_metadata() -> None:
    cfg = make_config()
    worker = FakeRolloutWorker()
    updater = DistributedWeightUpdater(
        cfg,
        make_model(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    updater._rollout_workers = [worker]
    updater._group_name = "sync-group"
    updater._process_group = "process-group"
    updater.weight_version = 19

    with (
        mock.patch.object(
            publish_module,
            "FlattenedTensorBucket",
            FakeFlattenedTensorBucket,
        ),
        mock.patch.object(publish_module.dist, "broadcast") as broadcast,
        mock.patch.object(publish_module.ray, "get", side_effect=lambda value: value),
    ):
        updater.update_bucket_weights([("w", torch.ones(2, 2, dtype=torch.float16))])

    assert worker.apply_distributed_calls[0]["bucket"].names == ["w"]
    assert worker.apply_distributed_calls[0]["bucket"].dtypes == ["float16"]
    broadcast.assert_called_once()

    idle = DistributedWeightUpdater(
        cfg,
        make_model(),
        rank=1,
        world_size=1,
        device=torch.device("cpu"),
        master_addr="127.0.0.1",
    )
    idle.update_bucket_weights([("w", torch.ones(1))])


TESTS = [
    ("weight_updater_buckets_and_cleans_up_on_error", test_weight_updater_buckets_and_cleans_up_on_error),
    (
        "weight_updater_prepare_tensor_handles_dtensor_and_waitable",
        test_weight_updater_prepare_tensor_handles_dtensor_and_waitable,
    ),
    (
        "tensor_weight_updater_connects_groups_and_runs_rpc_flow",
        test_tensor_weight_updater_connects_groups_and_runs_rpc_flow,
    ),
    (
        "tensor_weight_updater_disconnects_groups",
        test_tensor_weight_updater_disconnects_groups,
    ),
    (
        "distributed_weight_updater_session_and_rpc_flow",
        test_distributed_weight_updater_session_and_rpc_flow,
    ),
    (
        "distributed_weight_updater_broadcasts_bucket_metadata",
        test_distributed_weight_updater_broadcasts_bucket_metadata,
    ),
]


def run_suite() -> int:
    return run_named_tests(TESTS, summary_label="FSDP2 publish")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone tests for train.fsdp2 publish helpers."
    )
    parser.parse_args()
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
