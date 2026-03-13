# Copyright 2026 openforge

from __future__ import annotations

import argparse
import os
from typing import Literal

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import ray
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer

from _sglang_weight_update_common import (
    DEFAULT_MODEL,
    assert_success,
    build_cfg,
    find_free_port,
    require_visible_gpus,
    resolve_local_model_path,
)
from openforge.rollout.spec import EngineSpec
from openforge.rollout.worker import RolloutWorker
from openforge.train.fsdp2.weight_updater import WeightUpdater
from openforge.train.group import TrainWorkerGroup

SyncMode = Literal["disk", "tensor", "distributed"]


def parse_args(*, default_policy_version: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--policy-version", type=int, default=default_policy_version)
    return parser.parse_args()


def build_microbatch(model_path: str) -> TensorDict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode(
        "OpenForge end to end FSDP2 to SGLang weight sync test.",
        add_special_tokens=True,
    )
    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or token_ids[-1]
    while len(token_ids) < 16:
        token_ids.append(pad_token_id)
    token_ids = token_ids[:32]

    tokens = torch.tensor(token_ids, dtype=torch.long)
    return TensorDict(
        {
            "tokens": tokens,
            "position_ids": torch.arange(tokens.numel(), dtype=torch.long),
            "rewards": torch.linspace(0.25, 1.0, tokens.numel(), dtype=torch.float32),
            "loss_mask": torch.ones(tokens.numel() - 1, dtype=torch.float32),
        },
        batch_size=[],
    )


def create_rollout_worker(cfg):
    remote_worker = ray.remote(RolloutWorker)
    group_cfg = cfg.rollout.engine_groups[0]
    worker = remote_worker.options(
        num_cpus=group_cfg.num_cpus_per_replica,
        num_gpus=group_cfg.num_gpus_per_replica,
    ).remote()
    host = "127.0.0.1"
    port = ray.get(worker.allocate_port.remote(30000))
    spec = EngineSpec(
        cfg=cfg,
        name=f"{group_cfg.name}-0",
        role=group_cfg.role,
        worker_type=group_cfg.worker_type,
        num_gpus=group_cfg.num_gpus_per_replica,
        num_cpus=group_cfg.num_cpus_per_replica,
        parallelism=group_cfg.parallelism,
        engine_replica_index=0,
        host=host,
        port=port,
    )
    endpoint = ray.get(worker.initialize.remote(spec))
    if not endpoint.healthy:
        raise RuntimeError("rollout worker did not become healthy")
    return worker


def assert_trainer_weights_changed(
    before: list[list[tuple[str, torch.Tensor]]],
    after: list[list[tuple[str, torch.Tensor]]],
) -> None:
    after_tensors = {
        name: tensor
        for bucket in after
        for name, tensor in bucket
        if tensor.is_floating_point()
    }
    for bucket in before:
        for name, tensor in bucket:
            if not tensor.is_floating_point():
                continue
            updated = after_tensors.get(name)
            if updated is not None and not torch.equal(tensor, updated):
                return
    raise RuntimeError("trainer weights did not change after the training step")


def assert_weights_changed(worker) -> None:
    response = ray.get(worker.check_weights.remote(action="compare"))
    if response.get("success") is True:
        raise RuntimeError(
            "SGLang weights still match the initial snapshot after sync"
        )
    if response.get("success") is None:
        raise RuntimeError(f"check_weights(compare) returned invalid response: {response}")


def run_sync_test(mode: SyncMode, *, default_policy_version: int) -> int:
    args = parse_args(default_policy_version=default_policy_version)
    require_visible_gpus(2)
    model_path = resolve_local_model_path(args.model_path)
    cfg = build_cfg(model_path, train_total_gpus=1)
    cfg.train.config.optim.lr = 1e-2

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    train_group = TrainWorkerGroup()
    rollout_worker = None
    try:
        train_group.initialize(
            cfg,
            master_addr="127.0.0.1",
            master_port=find_free_port(),
        )
        rollout_worker = create_rollout_worker(cfg)

        assert_success(
            ray.get(rollout_worker.check_weights.remote(action="snapshot")),
            context="check_weights(snapshot)",
        )

        pre_step_weights = train_group.build_tensor_buckets(bucket_bytes=64 << 20)
        batch = build_microbatch(model_path)
        step_results = []
        for global_step in range(1, 4):
            step_results = train_group.step([[batch]], global_step=global_step)
        post_step_weights = train_group.build_tensor_buckets(bucket_bytes=64 << 20)
        assert_trainer_weights_changed(pre_step_weights, post_step_weights)

        updater = WeightUpdater(
            train_group,
            bucket_bytes=(4 << 20) if mode == "distributed" else (256 << 20),
        )
        updater.sync(
            [rollout_worker],
            policy_version=args.policy_version,
            mode=mode,
        )

        actual_version = ray.get(rollout_worker.get_weight_version.remote())
        if actual_version != str(args.policy_version):
            raise RuntimeError(
                f"Expected weight_version={args.policy_version}, got {actual_version!r}"
            )
        assert_weights_changed(rollout_worker)
        assert_success(
            ray.get(rollout_worker.check_weights.remote(action="snapshot")),
            context="check_weights(snapshot_after_sync)",
        )
        assert_success(
            ray.get(rollout_worker.check_weights.remote(action="compare")),
            context="check_weights(compare_after_resnapshot)",
        )

        print(f"[{mode}] SUCCESS train_metrics={step_results[0].metrics}")
        return 0
    finally:
        if rollout_worker is not None:
            try:
                ray.get(rollout_worker.shutdown.remote())
            except Exception:
                pass
        try:
            train_group.shutdown()
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()
