#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone 4-GPU FSDP2 backend validation script.

This script launches one worker process per GPU with isolated
`CUDA_VISIBLE_DEVICES`, then validates:
- distributed gradient accumulation with micro batching
- distributed checkpoint save/load
- model-only checkpoint restore behavior

It avoids external model downloads by patching the backend to use a tiny local
GPT-2 model plus a dummy tokenizer.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest import mock

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers import GPT2Config, GPT2LMHeadModel

from openforge.configs.models import OpenForgeConfig
import openforge.train.fsdp2.backend as fsdp_backend_module
from openforge.train.fsdp2.backend import FSDP2Backend
from openforge.train.types import TrainWorkerSpec
from openforge.utils.networking import get_free_port


VOCAB_SIZE = 64
MODEL_SEED = 1729
CHECKPOINT_SEED = 2718
EXPORT_SEED = 3141
REFERENCE_MODEL_SEED = 4242
REFERENCE_MODEL_NAME = "local-test-ref-model"
TINY_GPT2_KWARGS = {
    "vocab_size": VOCAB_SIZE,
    "n_positions": 32,
    "n_embd": 16,
    "n_layer": 2,
    "n_head": 2,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "attn_pdrop": 0.0,
}
ACCUMULATION_MICROBATCHES = [
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
    [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
    [[31, 32, 33, 34, 35], [36, 37, 38, 39, 40]],
]
CHECKPOINT_BATCH_ONE = [
    [[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]],
    [[13, 14, 15, 16, 17], [18, 19, 20, 21, 22]],
    [[23, 24, 25, 26, 27], [28, 29, 30, 31, 32]],
    [[33, 34, 35, 36, 37], [38, 39, 40, 41, 42]],
]
CHECKPOINT_BATCH_TWO = [
    [[4, 6, 8, 10, 12], [14, 16, 18, 20, 22]],
    [[5, 7, 9, 11, 13], [15, 17, 19, 21, 23]],
    [[24, 26, 28, 30, 32], [34, 36, 38, 40, 42]],
    [[25, 27, 29, 31, 33], [35, 37, 39, 41, 43]],
]
EXPORT_BATCHES = [
    [[2, 5, 8, 11, 14], [17, 20, 23, 26, 29]],
    [[3, 6, 9, 12, 15], [18, 21, 24, 27, 30]],
    [[4, 7, 10, 13, 16], [19, 22, 25, 28, 31]],
    [[32, 35, 38, 41, 44], [45, 48, 51, 54, 57]],
]
PACKED_MASKED_FORWARD_BATCHES = [
    [[1, 2, 3, 4], [5, 6, 7]],
    [[8, 9, 10], [11, 12, 13, 14]],
    [[15, 16, 17, 18], [19, 20, 21]],
    [[22, 23, 24], [25, 26, 27, 28]],
]
PACKED_MASKED_FORWARD_LOSS_MASKS = [
    [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
]
PACKED_MASKED_FORWARD_REWARDS = [
    [0.5, 1.25, 0.75, 2.0, 0.5, 1.0, 1.5],
    [1.1, 0.8, 1.7, 0.3, 2.2, 1.4, 0.6],
    [0.9, 1.6, 0.4, 1.8, 1.2, 0.7, 2.1],
    [1.5, 0.2, 1.1, 0.9, 1.7, 1.3, 0.8],
]
TEST_NAMES = (
    "gradient_accumulation",
    "checkpoint_round_trip",
    "model_only_checkpoint",
    "internal_ref_model",
    "packed_masked_forward",
    "sleep_wakeup",
)

def parse_gpu_ids(raw: str) -> list[int]:
    gpu_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    return gpu_ids


def make_config(
    run_dir: Path,
    *,
    world_size: int,
    mini_batch_size: int,
    micro_batch_size: int,
    kl_coef: float = 0.0,
    reference_model_name_or_path: str | None = None,
    offload_mode: str = "none",
) -> OpenForgeConfig:
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
                "reference_model_name_or_path": reference_model_name_or_path,
            },
            "algo": {
                "kl_coef": kl_coef,
            },
            "cluster": [
                {
                    "node_pool": "default",
                    "num_nodes": 1,
                    "num_gpus_per_node": max(2, world_size + 1),
                    "num_cpus_per_node": 32,
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
                        "mode": offload_mode,
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
                "global_batch_size": mini_batch_size * world_size,
                "mini_batch_size": mini_batch_size,
                "micro_batch_size": micro_batch_size,
                "checkpoints_dir": str(run_dir / "checkpoints"),
                "cpus_per_worker": 1,
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


def make_model(seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(**TINY_GPT2_KWARGS))


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=5e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def materialize_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "full_tensor"):
        try:
            value = value.full_tensor()
        except Exception:
            pass
    if hasattr(value, "to_local"):
        try:
            value = value.to_local()
        except Exception:
            pass
    return value.detach().cpu().clone()


def clone_structure(value: Any) -> Any:
    if torch.is_tensor(value):
        return materialize_tensor(value)
    if isinstance(value, dict):
        return {key: clone_structure(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [clone_structure(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(clone_structure(inner) for inner in value)
    return value


def assert_structure_close(expected: Any, actual: Any, *, path: str = "root") -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), f"{path}: expected tensor, got {type(actual)}"
        assert expected.shape == actual.shape, (
            f"{path}: shape mismatch {expected.shape} != {actual.shape}"
        )
        assert expected.dtype == actual.dtype, (
            f"{path}: dtype mismatch {expected.dtype} != {actual.dtype}"
        )
        left = materialize_tensor(expected)
        right = materialize_tensor(actual)
        if left.is_floating_point():
            assert torch.allclose(left, right, atol=1e-6, rtol=1e-5), (
                f"{path}: floating tensor mismatch"
            )
        else:
            assert torch.equal(left, right), f"{path}: tensor mismatch"
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert expected.keys() == actual.keys(), f"{path}: key mismatch"
        for key in expected:
            assert_structure_close(expected[key], actual[key], path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual)}"
        assert len(expected) == len(actual), f"{path}: length mismatch"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_structure_close(left, right, path=f"{path}[{index}]")
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), f"{path}: expected tuple, got {type(actual)}"
        assert len(expected) == len(actual), f"{path}: tuple length mismatch"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_structure_close(left, right, path=f"{path}[{index}]")
        return

    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def make_batch(
    sequences: list[list[int]],
    *,
    loss_mask: list[float] | None = None,
    rewards: list[float] | None = None,
) -> TensorDict:
    flat_tokens: list[int] = []
    position_ids: list[int] = []
    cu_seqlens = [0]

    for sequence in sequences:
        flat_tokens.extend(sequence)
        position_ids.extend(range(len(sequence)))
        cu_seqlens.append(cu_seqlens[-1] + len(sequence))

    token_count = len(flat_tokens)
    return TensorDict(
        {
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "loss_mask": torch.tensor(
                loss_mask if loss_mask is not None else [1.0] * token_count,
                dtype=torch.float32,
            ),
            "rewards": torch.tensor(
                rewards if rewards is not None else [1.0] * token_count,
                dtype=torch.float32,
            ),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        },
        batch_size=[],
    )


def compute_expected_forward_outputs(
    model: torch.nn.Module,
    batch: TensorDict,
    *,
    device: torch.device,
    kl_coef: float = 0.0,
    ref_model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    tokens = batch.get("tokens").to(device=device)
    loss_mask = batch.get("loss_mask").to(device=device).float()
    rewards = batch.get("rewards").to(device=device).float()
    position_ids = batch.get("position_ids").to(device=device)

    logits = model(
        input_ids=tokens.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
    ).logits[0, :-1, :]
    curr_log_probs = FSDP2Backend._chosen_log_probs(logits=logits, tokens=tokens)
    transition_mask = FSDP2Backend._transition_mask(
        loss_mask=loss_mask,
        position_ids=position_ids,
    )
    loss = FSDP2Backend._masked_mean(
        values=-(rewards[1:] * curr_log_probs),
        mask=transition_mask,
    )

    resolved_ref_log_probs = None
    if ref_model is not None:
        ref_logits = ref_model(
            input_ids=tokens.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
        ).logits[0, :-1, :]
        resolved_ref_log_probs = FSDP2Backend._chosen_log_probs(
            logits=ref_logits,
            tokens=tokens,
        )

    outputs = {
        "curr_log_probs": curr_log_probs.detach().cpu(),
    }
    if resolved_ref_log_probs is None:
        outputs["loss"] = loss.detach().cpu()
        return outputs

    loss = loss + (
        kl_coef
        * FSDP2Backend._masked_mean(
            values=curr_log_probs - resolved_ref_log_probs,
            mask=transition_mask,
        )
    )
    outputs["ref_log_probs"] = resolved_ref_log_probs.detach().cpu()
    outputs["loss"] = loss.detach().cpu()
    return outputs


def save_full_model_state_snapshot(model: torch.nn.Module, path: Path, *, rank: int) -> None:
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    if rank == 0:
        torch.save(clone_structure(state), path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    return clone_structure(torch.load(path, map_location="cpu"))


def mutate_backend_state(backend: FSDP2Backend) -> None:
    with torch.no_grad():
        for parameter in backend.model.parameters():
            parameter.add_(0.125)

    for state in backend.optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.add(0.25)

    backend.lr_scheduler.step()


def distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def parameter_devices(model: torch.nn.Module) -> set[str]:
    return {str(parameter.device) for parameter in model.parameters()}


def optimizer_state_devices(optimizer: torch.optim.Optimizer) -> set[str]:
    return {
        str(value.device)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    }


@contextmanager
def initialized_backend(
    run_dir: Path,
    *,
    seed: int,
    mini_batch_size: int,
    micro_batch_size: int,
    kl_coef: float = 0.0,
    reference_model_name_or_path: str | None = None,
    offload_mode: str = "none",
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> Iterator[tuple[FSDP2Backend, OpenForgeConfig]]:
    if dist.is_initialized():
        raise RuntimeError("process group already initialized before backend setup")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(
        run_dir,
        world_size=world_size,
        mini_batch_size=mini_batch_size,
        micro_batch_size=micro_batch_size,
        kl_coef=kl_coef,
        reference_model_name_or_path=reference_model_name_or_path,
        offload_mode=offload_mode,
    )
    def model_factory(model_name_or_path: str, *_args: Any, **_kwargs: Any) -> GPT2LMHeadModel:
        if world_size > 1:
            assert rank == 0, "from_pretrained() should only run on rank 0"
        if model_name_or_path == REFERENCE_MODEL_NAME:
            return make_model(REFERENCE_MODEL_SEED)
        return make_model(seed)

    def config_factory(*_args: Any, **_kwargs: Any) -> GPT2Config:
        return GPT2Config(**TINY_GPT2_KWARGS)

    def from_config_factory(
        config: GPT2Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> GPT2LMHeadModel:
        return GPT2LMHeadModel(config)

    backend = FSDP2Backend()
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                fsdp_backend_module.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=model_factory,
            )
        )
        stack.enter_context(
            mock.patch.object(
                fsdp_backend_module.AutoConfig,
                "from_pretrained",
                side_effect=config_factory,
            )
        )
        stack.enter_context(
            mock.patch.object(
                fsdp_backend_module.AutoModelForCausalLM,
                "from_config",
                side_effect=from_config_factory,
            )
        )
        try:
            backend.initialize(
                TrainWorkerSpec(
                    cfg=cfg,
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                )
            )
            yield backend, cfg
        finally:
            try:
                backend.shutdown()
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()


def run_gradient_accumulation_worker(
    run_dir: Path,
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    micro_sequences = ACCUMULATION_MICROBATCHES[rank]
    with initialized_backend(
        run_dir,
        seed=MODEL_SEED,
        mini_batch_size=4,
        micro_batch_size=2,
        kl_coef=0.0,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    ) as (backend, _cfg):
        backend.zero_grad()
        metrics = None
        for index, sequence in enumerate(micro_sequences):
            context = backend.no_sync() if index < len(micro_sequences) - 1 else nullcontext()
            with context:
                forward_out = backend.forward(make_batch([sequence]))
                backend.backward(forward_out)
            if index == len(micro_sequences) - 1:
                metrics = backend.step_optimizer(global_step=7)

        assert metrics is not None
        assert metrics.metrics["gradient_accumulation_steps"] == 2.0
        assert metrics.global_step == 7

        save_full_model_state_snapshot(
            backend.model,
            run_dir / "distributed_full_state.pt",
            rank=rank,
        )
        distributed_barrier()
        if rank == 0:
            save_json(
                run_dir / "rank0_metrics.json",
                {key: float(value) for key, value in metrics.items()},
            )


def run_checkpoint_round_trip_worker(
    run_dir: Path,
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    save_optimizer: bool,
) -> None:
    with initialized_backend(
        run_dir,
        seed=CHECKPOINT_SEED,
        mini_batch_size=4,
        micro_batch_size=4,
        kl_coef=0.0,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    ) as (backend, _cfg):
        first_batch = make_batch(CHECKPOINT_BATCH_ONE[rank])
        backend.zero_grad()
        forward_out = backend.forward(first_batch)
        backend.backward(forward_out)
        backend.step_optimizer(global_step=1)

        first_path = backend.save_checkpoint(
            step=3,
            policy_version=11,
            save_optimizer=save_optimizer,
        )
        first_payload = load_checkpoint_payload(first_path)

        if save_optimizer:
            second_batch = make_batch(CHECKPOINT_BATCH_TWO[rank])
            backend.zero_grad()
            forward_out = backend.forward(second_batch)
            backend.backward(forward_out)
            backend.step_optimizer(global_step=2)

            second_path = backend.save_checkpoint(
                step=5,
                policy_version=13,
                save_optimizer=True,
            )
            second_payload = load_checkpoint_payload(second_path)

            mutate_backend_state(backend)
            restored = backend.load_checkpoint(latest=True)
            assert restored is not None and restored.step == 5 and restored.policy_version == 13
            assert_structure_close(
                second_payload["model_state"],
                clone_structure(backend.model.state_dict()),
                path=f"rank{rank}.latest.model_state",
            )
            assert_structure_close(
                second_payload["optimizer_state"],
                clone_structure(backend.optimizer.state_dict()),
                path=f"rank{rank}.latest.optimizer_state",
            )
            assert_structure_close(
                second_payload["scheduler_state"],
                clone_structure(backend.lr_scheduler.state_dict()),
                path=f"rank{rank}.latest.scheduler_state",
            )

            mutate_backend_state(backend)
            restored = backend.load_checkpoint(latest=False, step=3)
            assert restored is not None and restored.step == 3 and restored.policy_version == 11
            assert_structure_close(
                first_payload["model_state"],
                clone_structure(backend.model.state_dict()),
                path=f"rank{rank}.explicit.model_state",
            )
            assert_structure_close(
                first_payload["optimizer_state"],
                clone_structure(backend.optimizer.state_dict()),
                path=f"rank{rank}.explicit.optimizer_state",
            )
            assert_structure_close(
                first_payload["scheduler_state"],
                clone_structure(backend.lr_scheduler.state_dict()),
                path=f"rank{rank}.explicit.scheduler_state",
            )
        else:
            assert "optimizer_state" not in first_payload
            assert "scheduler_state" not in first_payload
            mutate_backend_state(backend)
            optimizer_before = clone_structure(backend.optimizer.state_dict())
            scheduler_before = clone_structure(backend.lr_scheduler.state_dict())

            restored = backend.load_checkpoint(step=3)
            assert restored is not None and restored.step == 3 and restored.policy_version == 11
            assert_structure_close(
                first_payload["model_state"],
                clone_structure(backend.model.state_dict()),
                path=f"rank{rank}.model_only.model_state",
            )
            assert_structure_close(
                optimizer_before,
                clone_structure(backend.optimizer.state_dict()),
                path=f"rank{rank}.model_only.optimizer_state",
            )
            assert_structure_close(
                scheduler_before,
                clone_structure(backend.lr_scheduler.state_dict()),
                path=f"rank{rank}.model_only.scheduler_state",
            )

        distributed_barrier()


def run_internal_ref_model_worker(
    run_dir: Path,
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    kl_coef = 0.75
    with initialized_backend(
        run_dir,
        seed=MODEL_SEED,
        mini_batch_size=4,
        micro_batch_size=4,
        kl_coef=kl_coef,
        reference_model_name_or_path=REFERENCE_MODEL_NAME,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    ) as (backend, _cfg):
        batch = make_batch(CHECKPOINT_BATCH_ONE[rank])
        actor_model = make_model(MODEL_SEED)
        actor_model.eval()
        ref_model = make_model(REFERENCE_MODEL_SEED)
        ref_model.eval()

        forward_out = backend.forward(batch)
        actual = {
            "loss": materialize_tensor(forward_out.get("loss")),
            "curr_log_probs": materialize_tensor(forward_out.get("curr_log_probs")),
            "ref_log_probs": materialize_tensor(forward_out.get("ref_log_probs")),
        }
        expected = compute_expected_forward_outputs(
            actor_model,
            batch,
            device=torch.device("cpu"),
            kl_coef=kl_coef,
            ref_model=ref_model,
        )
        assert_structure_close(expected, actual, path=f"rank{rank}.internal_ref")

        backend.zero_grad()
        backend.backward(forward_out)
        metrics = backend.step_optimizer(global_step=4)
        assert metrics.global_step == 4

        distributed_barrier()


def run_packed_masked_forward_worker(
    run_dir: Path,
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    with initialized_backend(
        run_dir,
        seed=MODEL_SEED,
        mini_batch_size=4,
        micro_batch_size=4,
        kl_coef=0.0,
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    ) as (backend, _cfg):
        batch = make_batch(
            PACKED_MASKED_FORWARD_BATCHES[rank],
            loss_mask=PACKED_MASKED_FORWARD_LOSS_MASKS[rank],
            rewards=PACKED_MASKED_FORWARD_REWARDS[rank],
        )
        actor_model = make_model(MODEL_SEED)
        actor_model.eval()

        forward_out = backend.forward(batch)
        actual = {
            "loss": materialize_tensor(forward_out.get("loss")),
            "curr_log_probs": materialize_tensor(forward_out.get("curr_log_probs")),
        }
        expected = compute_expected_forward_outputs(
            actor_model,
            batch,
            device=torch.device("cpu"),
            kl_coef=0.0,
        )
        assert_structure_close(
            expected,
            actual,
            path=f"rank{rank}.packed_masked_forward",
        )

        backend.zero_grad()
        backend.backward(forward_out)
        metrics = backend.step_optimizer(global_step=6)
        assert metrics.global_step == 6

        distributed_barrier()


def run_sleep_wakeup_worker(
    run_dir: Path,
    *,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    with initialized_backend(
        run_dir,
        seed=CHECKPOINT_SEED,
        mini_batch_size=4,
        micro_batch_size=4,
        kl_coef=0.0,
        offload_mode="none",
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    ) as (backend, _cfg):
        batch = make_batch(CHECKPOINT_BATCH_TWO[rank])
        backend.zero_grad()
        forward_out = backend.forward(batch)
        backend.backward(forward_out)
        backend.step_optimizer(global_step=1)

        devices_before_sleep = parameter_devices(backend.model)
        optimizer_devices_before_sleep = optimizer_state_devices(backend.optimizer)
        assert devices_before_sleep == {"cuda:0"}, devices_before_sleep
        assert optimizer_devices_before_sleep, "expected AdamW state after first step"

        backend.sleep()
        assert backend._sleeping is True
        assert parameter_devices(backend.model) == {"cpu"}
        assert optimizer_state_devices(backend.optimizer) == {"cpu"}

        backend.wakeup()
        assert backend._sleeping is False
        assert parameter_devices(backend.model) == {"cuda:0"}
        assert optimizer_state_devices(backend.optimizer) == {"cuda:0"}

        backend.zero_grad()
        forward_out = backend.forward(batch)
        backend.backward(forward_out)
        metrics = backend.step_optimizer(global_step=2)
        assert metrics.global_step == 2

        distributed_barrier()


def worker_main(args: argparse.Namespace) -> int:
    test_dir = args.artifacts_dir / args.test_name
    try:
        if args.test_name == "gradient_accumulation":
            run_gradient_accumulation_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        elif args.test_name == "checkpoint_round_trip":
            run_checkpoint_round_trip_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
                save_optimizer=True,
            )
        elif args.test_name == "model_only_checkpoint":
            run_checkpoint_round_trip_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
                save_optimizer=False,
            )
        elif args.test_name == "internal_ref_model":
            run_internal_ref_model_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        elif args.test_name == "packed_masked_forward":
            run_packed_masked_forward_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        elif args.test_name == "sleep_wakeup":
            run_sleep_wakeup_worker(
                test_dir,
                rank=args.rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        else:
            raise ValueError(f"unknown test {args.test_name}")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def run_reference_accumulation_state(device: torch.device) -> dict[str, Any]:
    model = make_model(MODEL_SEED).to(device)
    model.train()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    all_sequences = [
        sequence
        for rank_sequences in ACCUMULATION_MICROBATCHES
        for sequence in rank_sequences
    ]
    optimizer.zero_grad(set_to_none=True)
    scale = float(len(all_sequences))
    for sequence in all_sequences:
        batch = make_batch([sequence])
        tokens = batch.get("tokens").to(device=device)
        loss_mask = batch.get("loss_mask").to(device=device).float()
        rewards = batch.get("rewards").to(device=device).float()
        position_ids = batch.get("position_ids").to(device=device)

        logits = model(
            input_ids=tokens.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
        ).logits[0, :-1, :]
        curr_log_probs = FSDP2Backend._chosen_log_probs(logits=logits, tokens=tokens)
        transition_mask = FSDP2Backend._transition_mask(
            loss_mask=loss_mask,
            position_ids=position_ids,
        )
        loss = (
            FSDP2Backend._masked_mean(
                values=-(rewards[1:] * curr_log_probs),
                mask=transition_mask,
            )
            / scale
        )
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return clone_structure(model.state_dict())


def validate_gradient_accumulation_outputs(
    test_dir: Path,
    *,
    reference_device: torch.device,
) -> None:
    distributed_path = test_dir / "gradient_accumulation" / "distributed_full_state.pt"
    assert distributed_path.exists(), f"missing distributed state snapshot: {distributed_path}"
    distributed_state = clone_structure(torch.load(distributed_path, map_location="cpu"))
    reference_state = run_reference_accumulation_state(reference_device)
    assert_structure_close(
        reference_state,
        distributed_state,
        path="distributed_vs_reference",
    )

    metrics = json.loads(
        (test_dir / "gradient_accumulation" / "rank0_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["gradient_accumulation_steps"] == 2.0
    assert metrics["global_step"] == 7.0


def launch_workers_for_test(
    script_path: Path,
    test_name: str,
    artifacts_dir: Path,
    gpu_ids: list[int],
) -> None:
    master_addr = "127.0.0.1"
    master_port = get_free_port(start=29500)
    processes: list[subprocess.Popen[Any]] = []
    log_paths: list[Path] = []
    log_handles = []

    for rank, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_path = artifacts_dir / test_name / f"rank_{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--test-name",
            test_name,
            "--rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
            "--master-addr",
            master_addr,
            "--master-port",
            str(master_port),
            "--artifacts-dir",
            str(artifacts_dir),
        ]
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        log_paths.append(log_path)
        log_handles.append(log_handle)

    failure: tuple[int, int] | None = None
    try:
        while processes:
            for index, process in enumerate(list(processes)):
                returncode = process.poll()
                if returncode is None:
                    continue
                if returncode != 0 and failure is None:
                    failure = (index, returncode)
                    for other in processes:
                        if other.poll() is None:
                            other.terminate()
                    break
                processes.remove(process)
            if failure is not None:
                break
            time.sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for log_handle in log_handles:
            if not log_handle.closed:
                log_handle.close()

    if failure is not None:
        failed_rank, returncode = failure
        for process in processes:
            if process.poll() is None:
                process.kill()
        print(f"[FAIL] {test_name} (rank {failed_rank}, exit {returncode})")
        for rank, log_path in enumerate(log_paths):
            print()
            print(f"--- rank {rank} log: {log_path} ---")
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8").rstrip())
        raise RuntimeError(f"{test_name} failed")

    for rank, log_path in enumerate(log_paths):
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8").rstrip()
        if text:
            print(f"--- rank {rank} log ---")
            print(text)


def driver_main(args: argparse.Namespace) -> int:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) != 4:
        raise ValueError(f"expected exactly 4 GPUs, got {gpu_ids}")

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False in the driver process")

    visible = torch.cuda.device_count()
    if visible <= max(gpu_ids):
        raise RuntimeError(
            f"requested GPUs {gpu_ids}, but only {visible} CUDA devices are visible"
        )

    print(f"Using GPUs: {gpu_ids}")
    print(f"Artifacts directory: {args.artifacts_dir}")

    script_path = Path(__file__).resolve()
    tests = list(TEST_NAMES)
    failures: list[str] = []

    for test_name in tests:
        print(f"[RUN ] {test_name}")
        try:
            launch_workers_for_test(
                script_path,
                test_name,
                args.artifacts_dir,
                gpu_ids,
            )
            if test_name == "gradient_accumulation":
                validate_gradient_accumulation_outputs(
                    args.artifacts_dir,
                    reference_device=torch.device("cuda", gpu_ids[0]),
                )
        except Exception:
            failures.append(test_name)
            print(traceback.format_exc().rstrip())
        else:
            print(f"[PASS] {test_name}")

    if failures:
        print()
        print("Failed tests:")
        for name in failures:
            print(f"- {name}")
        return 1

    print()
    print(f"All {len(tests)} GPU tests passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone 4-GPU FSDP2 backend validation script."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory to store logs and artifacts. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated physical GPU ids for the 4 worker processes.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-name", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--master-addr", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--master-port", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.artifacts_dir is None:
        with TemporaryDirectory(prefix="openforge_fsdp2_gpu_") as temp_dir:
            args.artifacts_dir = Path(temp_dir)
            if args.worker:
                return worker_main(args)
            return driver_main(args)

    args.artifacts_dir = Path(args.artifacts_dir)
    if args.worker:
        return worker_main(args)
    return driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
