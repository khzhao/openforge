#!/usr/bin/env python3
# Copyright 2026 openforge

"""Standalone CPU tests for the FSDP2 backend.

This script complements the multi-GPU validation script with fast local tests
for backend logic that does not require real distributed GPU execution:
- forward loss / KL math and backward scaling
- optimizer stepping and scheduler behavior
- checkpoint load flags and checkpoint path selection
- rollout weight export gating on rank 0
- sleep / wakeup behavior for different offload modes
- model construction helper branches
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from tensordict import TensorDict

from openforge.configs.models import OpenForgeConfig
import openforge.train.fsdp2.backend as fsdp_backend_module
import openforge.train.fsdp2.runtime as fsdp_runtime_module
from openforge.train.fsdp2.backend import FSDP2Backend
from openforge.train.fsdp2.memory import offload_params, onload_params
from openforge.train.types import CheckpointInfo, TrainStepResult

VOCAB_SIZE = 32

class ToyCausalLM(torch.nn.Module):
    """Tiny deterministic language model with a HuggingFace-like interface."""

    def __init__(
        self,
        *,
        seed: int,
        linear_attention: bool = False,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 12,
        max_positions: int = 32,
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.token_emb = torch.nn.Embedding(vocab_size, hidden_size)
            self.position_emb = torch.nn.Embedding(max_positions, hidden_size)
            self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                layer_types=(
                    ["linear_attention"] if linear_attention else ["full_attention"]
                )
            )
        )
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> SimpleNamespace:
        hidden = self.token_emb(input_ids) + self.position_emb(position_ids)
        return SimpleNamespace(logits=self.proj(hidden))

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        state_dict: dict[str, torch.Tensor],
        safe_serialization: bool,
    ) -> tuple[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(
            json.dumps({"safe_serialization": safe_serialization}, indent=2),
            encoding="utf-8",
        )
        torch.save(clone_structure(state_dict), save_path / "model_state.pt")
        return (str(save_path / "model_state.pt"),)


def make_config(
    run_dir: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    mini_batch_size: int = 4,
    micro_batch_size: int = 2,
    lr: float = 0.2,
    kl_coef: float = 0.0,
    reference_model_name_or_path: str | None = None,
    offload_mode: str = "none",
    scheduler_type: str = "constant",
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    min_lr_rate: float | None = None,
    num_cycles: float = 0.5,
    shard_modules: list[str] | None = None,
) -> OpenForgeConfig:
    _ = rank
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
            "algo": {"kl_coef": kl_coef},
            "cluster": [
                {
                    "node_pool": "default",
                    "num_nodes": 1,
                    "num_gpus_per_node": max(2, world_size + 1),
                    "num_cpus_per_node": 8,
                }
            ],
            "train": {
                "backend": "fsdp2",
                "backend_config": {
                    "gradient_checkpointing": False,
                    "reshard_after_forward": False,
                    "shard_modules": shard_modules,
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
                        "lr": lr,
                        "adam_beta1": 0.9,
                        "adam_beta2": 0.95,
                        "adam_eps": 1e-8,
                        "weight_decay": 0.0,
                        "max_grad_norm": 1.0,
                    },
                    "scheduler": {
                        "type": scheduler_type,
                        "warmup_steps": warmup_steps,
                        "min_lr": min_lr,
                        "num_cycles": num_cycles,
                        "min_lr_rate": min_lr_rate,
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


def materialize_tensor(value: torch.Tensor) -> torch.Tensor:
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
        torch.testing.assert_close(
            materialize_tensor(actual),
            materialize_tensor(expected),
            msg=path,
        )
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
        assert len(expected) == len(actual), f"{path}: length mismatch"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_structure_close(left, right, path=f"{path}[{index}]")
        return

    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def manual_forward_outputs(
    model: torch.nn.Module,
    batch: TensorDict,
    *,
    ref_model: torch.nn.Module | None = None,
    kl_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    tokens = batch.get("tokens")
    loss_mask = batch.get("loss_mask").float()
    rewards = batch.get("rewards").float()
    position_ids = batch.get("position_ids")

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

    outputs = {"loss": loss, "curr_log_probs": curr_log_probs}
    if ref_model is None or kl_coef <= 0.0:
        return outputs

    ref_logits = ref_model(
        input_ids=tokens.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
    ).logits[0, :-1, :]
    ref_log_probs = FSDP2Backend._chosen_log_probs(logits=ref_logits, tokens=tokens)
    loss = loss + (
        kl_coef
        * FSDP2Backend._masked_mean(
            values=curr_log_probs - ref_log_probs,
            mask=transition_mask,
        )
    )
    outputs["loss"] = loss
    outputs["ref_log_probs"] = ref_log_probs
    return outputs


def build_backend(
    run_dir: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    model_seed: int = 123,
    ref_model_seed: int = 456,
    mini_batch_size: int = 4,
    micro_batch_size: int = 2,
    lr: float = 0.2,
    kl_coef: float = 0.0,
    reference_model_name_or_path: str | None = None,
    offload_mode: str = "none",
    scheduler_type: str = "constant",
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    min_lr_rate: float | None = None,
    num_cycles: float = 0.5,
    shard_modules: list[str] | None = None,
) -> FSDP2Backend:
    cfg = make_config(
        run_dir,
        rank=rank,
        world_size=world_size,
        mini_batch_size=mini_batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        kl_coef=kl_coef,
        reference_model_name_or_path=reference_model_name_or_path,
        offload_mode=offload_mode,
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        min_lr_rate=min_lr_rate,
        num_cycles=num_cycles,
        shard_modules=shard_modules,
    )

    backend = FSDP2Backend()
    backend.cfg = cfg
    backend.rank = rank
    backend.world_size = world_size
    backend.master_addr = "127.0.0.1"
    backend.master_port = 29601
    backend.device = torch.device("cpu")
    backend.model = ToyCausalLM(seed=model_seed)
    backend.ref_model = (
        ToyCausalLM(seed=ref_model_seed)
        if reference_model_name_or_path is not None
        else None
    )
    backend.optimizer = torch.optim.AdamW(
        backend.model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    backend.lr_scheduler = backend._build_lr_scheduler()
    backend.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    backend._sleeping = False
    return backend


def clone_model(model: ToyCausalLM) -> ToyCausalLM:
    clone = ToyCausalLM(seed=0)
    clone.load_state_dict(clone_structure(model.state_dict()))
    return clone


def seed_optimizer_state(backend: FSDP2Backend) -> None:
    backend.zero_grad()
    for parameter in backend.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    backend.optimizer.step()


def mutate_optimizer_state(optimizer: torch.optim.Optimizer, *, delta: float) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.add(delta)


def test_forward_without_reference_model(artifacts_dir: Path) -> None:
    backend = build_backend(artifacts_dir / "forward_no_ref")
    batch = make_batch(
        [[1, 2, 3, 4], [5, 6, 7]],
        loss_mask=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        rewards=[0.5, 1.25, 0.75, 2.0, 0.5, 1.0, 1.5],
    )

    outputs = backend.forward(batch)
    expected = manual_forward_outputs(backend.model, batch)

    torch.testing.assert_close(outputs.get("loss"), expected["loss"])
    torch.testing.assert_close(
        outputs.get("curr_log_probs"),
        expected["curr_log_probs"],
    )
    assert "ref_log_probs" not in outputs.keys()


def test_forward_with_reference_model_and_backward_scaling(
    artifacts_dir: Path,
) -> None:
    backend = build_backend(
        artifacts_dir / "forward_with_ref",
        kl_coef=0.35,
        reference_model_name_or_path="local-ref-model",
        mini_batch_size=4,
        micro_batch_size=2,
    )
    assert backend.ref_model is not None

    batch = make_batch(
        [[3, 4, 5, 6], [7, 8, 9]],
        loss_mask=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        rewards=[0.2, 0.9, 1.4, 0.5, 2.0, 1.1, 0.7],
    )

    outputs = backend.forward(batch)
    expected = manual_forward_outputs(
        backend.model,
        batch,
        ref_model=backend.ref_model,
        kl_coef=backend.cfg.algo.kl_coef,
    )

    torch.testing.assert_close(outputs.get("loss"), expected["loss"])
    torch.testing.assert_close(
        outputs.get("curr_log_probs"),
        expected["curr_log_probs"],
    )
    torch.testing.assert_close(outputs.get("ref_log_probs"), expected["ref_log_probs"])

    backend.zero_grad()
    backend.backward(outputs)

    manual_model = clone_model(backend.model)
    manual_ref_model = clone_model(backend.ref_model)
    manual_outputs = manual_forward_outputs(
        manual_model,
        batch,
        ref_model=manual_ref_model,
        kl_coef=backend.cfg.algo.kl_coef,
    )
    (
        manual_outputs["loss"] / backend.cfg.train.gradient_accumulation_steps
    ).backward()

    for backend_param, manual_param in zip(
        backend.model.parameters(),
        manual_model.parameters(),
        strict=True,
    ):
        assert backend_param.grad is not None
        assert manual_param.grad is not None
        torch.testing.assert_close(backend_param.grad, manual_param.grad)

    assert all(param.grad is None for param in backend.ref_model.parameters())


def test_step_optimizer_tracks_lr_schedule(artifacts_dir: Path) -> None:
    backend = build_backend(
        artifacts_dir / "optimizer_step",
        scheduler_type="linear",
        warmup_steps=2,
        min_lr_rate=0.25,
        lr=0.2,
        mini_batch_size=4,
        micro_batch_size=2,
    )

    first_param_before = materialize_tensor(next(backend.model.parameters()))
    assert abs(backend.optimizer.param_groups[0]["lr"] - 0.0) < 1e-7
    observed_lrs: list[float] = []
    for step in range(4):
        backend.zero_grad()
        for parameter in backend.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        metrics = backend.step_optimizer(global_step=step + 1)
        assert isinstance(metrics, TrainStepResult)
        observed_lrs.append(metrics.metrics["lr"])
        assert metrics.global_step == step + 1
        assert metrics.metrics["gradient_accumulation_steps"] == 2.0

    expected_lrs = [0.1, 0.2, 0.125, 0.05]
    for observed, expected in zip(observed_lrs, expected_lrs, strict=True):
        assert abs(observed - expected) < 1e-7

    first_param_after = materialize_tensor(next(backend.model.parameters()))
    assert not torch.equal(first_param_before, first_param_after)


def test_checkpoint_load_skips_optimizer_when_requested(artifacts_dir: Path) -> None:
    backend = build_backend(artifacts_dir / "checkpoint_load")
    seed_optimizer_state(backend)

    checkpoint_info = backend.save_checkpoint(
        step=3,
        policy_version=11,
        save_optimizer=True,
    )
    assert isinstance(checkpoint_info, CheckpointInfo)
    assert Path(checkpoint_info.path).exists()

    expected_model = clone_structure(backend.model.state_dict())

    with torch.no_grad():
        for parameter in backend.model.parameters():
            parameter.add_(0.5)
    mutate_optimizer_state(backend.optimizer, delta=0.75)
    backend.lr_scheduler.step()

    mutated_optimizer = clone_structure(backend.optimizer.state_dict())
    mutated_scheduler = clone_structure(backend.lr_scheduler.state_dict())

    restored = backend.load_checkpoint(latest=False, step=3, load_optimizer=False)
    assert restored is not None and restored.step == 3 and restored.policy_version == 11
    assert_structure_close(expected_model, backend.model.state_dict())
    assert_structure_close(mutated_optimizer, backend.optimizer.state_dict())
    assert_structure_close(mutated_scheduler, backend.lr_scheduler.state_dict())
    assert backend.load_checkpoint(latest=False, step=999) is None


def test_resolve_checkpoint_path_prefers_latest_rank_local_file(
    artifacts_dir: Path,
) -> None:
    checkpoints_dir = artifacts_dir / "resolve_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "step_00000002.rank_00000.pt",
        "step_00000005.rank_00001.pt",
        "step_00000009.rank_00001.pt",
    ):
        (checkpoints_dir / name).write_text("test", encoding="utf-8")

    backend = FSDP2Backend()
    backend.rank = 1

    latest = backend._resolve_checkpoint_path(checkpoints_dir, latest=True, step=None)
    assert latest == checkpoints_dir / "step_00000009.rank_00001.pt"

    explicit = backend._resolve_checkpoint_path(checkpoints_dir, latest=True, step=5)
    assert explicit == checkpoints_dir / "step_00000005.rank_00001.pt"

    assert (
        backend._resolve_checkpoint_path(checkpoints_dir, latest=False, step=None)
        is None
    )
    assert (
        backend._resolve_checkpoint_path(checkpoints_dir, latest=True, step=3) is None
    )


def test_sleep_and_wakeup_respect_offload_mode(artifacts_dir: Path) -> None:
    backend = build_backend(artifacts_dir / "sleep_none", offload_mode="none")
    with (
        mock.patch.object(
            fsdp_backend_module,
            "offload_params",
        ) as patched_offload_params,
        mock.patch.object(
            fsdp_backend_module,
            "offload_optimizer",
        ) as patched_offload_optimizer,
        mock.patch.object(
            fsdp_backend_module,
            "onload_params",
        ) as patched_onload_params,
        mock.patch.object(
            fsdp_backend_module,
            "onload_optimizer",
        ) as patched_onload_optimizer,
        mock.patch.object(backend, "clear_memory") as clear_memory,
    ):
        backend.sleep()
        backend.sleep()
        backend.wakeup()
        backend.wakeup()

    assert backend._sleeping is False
    patched_offload_params.assert_called_once_with(
        backend.model,
        offload_grad=True,
    )
    patched_offload_optimizer.assert_called_once_with(backend.optimizer)
    patched_onload_params.assert_called_once_with(
        backend.model,
        torch.device("cpu"),
        onload_grad=True,
    )
    patched_onload_optimizer.assert_called_once_with(
        backend.optimizer,
        torch.device("cpu"),
    )
    assert clear_memory.call_count == 2

    cpu_offload_backend = build_backend(
        artifacts_dir / "sleep_cpu",
        offload_mode="cpu",
    )
    with (
        mock.patch.object(
            fsdp_backend_module,
            "offload_params",
        ) as patched_offload_params,
        mock.patch.object(
            fsdp_backend_module,
            "offload_optimizer",
        ) as patched_offload_optimizer,
        mock.patch.object(
            fsdp_backend_module,
            "onload_params",
        ) as patched_onload_params,
        mock.patch.object(
            fsdp_backend_module,
            "onload_optimizer",
        ) as patched_onload_optimizer,
        mock.patch.object(cpu_offload_backend, "clear_memory") as clear_memory,
    ):
        cpu_offload_backend.sleep()
        assert cpu_offload_backend._sleeping is True
        cpu_offload_backend.wakeup()
        assert cpu_offload_backend._sleeping is False

    patched_offload_params.assert_not_called()
    patched_offload_optimizer.assert_not_called()
    patched_onload_params.assert_not_called()
    patched_onload_optimizer.assert_not_called()
    assert clear_memory.call_count == 2


def test_offload_and_onload_params_move_gradients(_artifacts_dir: Path) -> None:
    class FakeGrad:
        def __init__(self, device: torch.device) -> None:
            self.device = device
            self.detach_called = False
            self.last_to: tuple[torch.device, bool] | None = None

        def detach(self) -> "FakeGrad":
            self.detach_called = True
            return self

        def to(
            self,
            device: torch.device | str,
            non_blocking: bool = False,
        ) -> "FakeGrad":
            target_device = torch.device(device)
            self.device = target_device
            self.last_to = (target_device, non_blocking)
            return self

    class FakeParameter:
        def __init__(self, grad: "FakeGrad | None") -> None:
            self.grad = grad

    class FakeModel:
        def __init__(self, parameters: list["FakeParameter"]) -> None:
            self._parameters = parameters
            self.to_calls: list[torch.device] = []

        def to(self, *, device: torch.device | str) -> "FakeModel":
            self.to_calls.append(torch.device(device))
            return self

        def parameters(self):
            return iter(self._parameters)

    cpu_grad = FakeGrad(torch.device("cpu"))
    cuda_grad = FakeGrad(torch.device("cuda", 0))
    parameters = [
        FakeParameter(cpu_grad),
        FakeParameter(cuda_grad),
        FakeParameter(None),
    ]
    model = FakeModel(parameters)

    offload_params(model, offload_grad=True)

    assert model.to_calls == [torch.device("cpu")]
    assert cpu_grad.detach_called is False
    assert cpu_grad.last_to is None
    assert cuda_grad.detach_called is True
    assert cuda_grad.last_to == (torch.device("cpu"), True)
    assert parameters[1].grad is cuda_grad

    cuda_grad.detach_called = False
    cuda_grad.last_to = None

    onload_params(model, torch.device("cuda", 0), onload_grad=True)

    assert model.to_calls == [torch.device("cpu"), torch.device("cuda", 0)]
    assert cuda_grad.detach_called is True
    assert cuda_grad.last_to == (torch.device("cuda", 0), True)
    assert parameters[1].grad is cuda_grad


def test_fsdp2_base_infers_common_transformer_block_paths(_artifacts_dir: Path) -> None:
    class QwenLikeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )

    class Gpt2LikeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )

    qwen_like = QwenLikeModel()
    gpt2_like = Gpt2LikeModel()

    qwen_blocks = fsdp_runtime_module.resolve_shard_modules(qwen_like, None)
    gpt2_blocks = fsdp_runtime_module.resolve_shard_modules(gpt2_like, None)

    assert qwen_blocks == list(qwen_like.model.layers)
    assert gpt2_blocks == list(gpt2_like.transformer.h)


def test_fsdp2_base_resolve_shard_modules_validates_and_deduplicates(
    _artifacts_dir: Path,
) -> None:
    class ExplicitShardModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )
            self.layers_alias = self.transformer.h
            self.not_a_module = 7

    model = ExplicitShardModel()
    resolved = fsdp_runtime_module.resolve_shard_modules(
        model,
        ["transformer.h", "layers_alias"],
    )
    assert resolved == list(model.transformer.h)

    try:
        fsdp_runtime_module.resolve_shard_modules(model, ["missing.layers"])
    except ValueError as exc:
        assert "attribute not found" in str(exc)
    else:
        raise AssertionError("expected invalid shard path to raise ValueError")

    try:
        fsdp_runtime_module.resolve_shard_modules(model, ["not_a_module"])
    except TypeError as exc:
        assert "must resolve to nn.Module" in str(exc)
    else:
        raise AssertionError("expected non-module shard path to raise TypeError")


def test_apply_fsdp2_shards_blocks_then_root(_artifacts_dir: Path) -> None:
    class QwenLikeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )
            self.model.embed_tokens = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    model = QwenLikeModel()
    mp_policy = fsdp_runtime_module.MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    offload_policy = fsdp_runtime_module.OffloadPolicy()

    with mock.patch.object(
        fsdp_runtime_module,
        "fully_shard",
        side_effect=lambda module, **_kwargs: module,
    ) as fully_shard:
        wrapped = fsdp_runtime_module.apply_fsdp2(
            model,
            device_mesh="mesh",
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=True,
        )

    assert wrapped is model
    assert fully_shard.call_count == 3
    assert fully_shard.call_args_list[0].args[0] is model.model.layers[0]
    assert fully_shard.call_args_list[1].args[0] is model.model.layers[1]
    assert fully_shard.call_args_list[2].args[0] is model
    assert fully_shard.call_args_list[0].kwargs["reshard_after_forward"] is True
    assert fully_shard.call_args_list[1].kwargs["reshard_after_forward"] is True
    assert fully_shard.call_args_list[2].kwargs["reshard_after_forward"] is False


def test_create_model_runtime_builds_expected_policies(artifacts_dir: Path) -> None:
    backend = build_backend(artifacts_dir / "model_runtime", offload_mode="cpu")

    with mock.patch.object(
        fsdp_backend_module,
        "create_device_mesh",
        return_value="test-mesh",
    ) as create_device_mesh:
        mesh, mp_policy, offload_policy = backend._create_model_runtime(
            device=torch.device("cpu"),
            cpu_offload=True,
        )

    create_device_mesh.assert_called_once_with(
        dp_size=1,
        world_size=1,
        device_type="cpu",
    )
    assert mesh == "test-mesh"
    assert mp_policy.param_dtype == torch.float32
    assert mp_policy.reduce_dtype == torch.float32
    assert isinstance(offload_policy, fsdp_backend_module.CPUOffloadPolicy)


def test_build_model_and_state_for_fsdp_uses_rank_specific_paths(
    artifacts_dir: Path,
) -> None:
    rank0_backend = build_backend(
        artifacts_dir / "build_model_rank0",
        rank=0,
        world_size=2,
    )
    rank0_model = ToyCausalLM(seed=21)
    with (
        mock.patch.object(
            fsdp_backend_module.AutoModelForCausalLM,
            "from_pretrained",
            return_value=rank0_model,
        ) as from_pretrained,
        mock.patch.object(
            fsdp_backend_module.AutoModelForCausalLM,
            "from_config",
        ) as from_config,
    ):
        model, full_state = rank0_backend._build_model_and_state_for_fsdp("toy-model")

    assert model is rank0_model
    assert set(full_state) == set(rank0_model.state_dict())
    from_pretrained.assert_called_once()
    from_config.assert_not_called()

    rank1_backend = build_backend(
        artifacts_dir / "build_model_rank1_meta",
        rank=1,
        world_size=2,
    )
    with (
        mock.patch.object(
            fsdp_backend_module.AutoConfig,
            "from_pretrained",
            return_value=SimpleNamespace(tie_word_embeddings=False),
        ),
        mock.patch.object(
            fsdp_backend_module.AutoModelForCausalLM,
            "from_config",
            side_effect=lambda *_args, **_kwargs: ToyCausalLM(seed=22),
        ),
    ):
        model, full_state = rank1_backend._build_model_and_state_for_fsdp("toy-model")

    assert full_state == {}
    assert next(model.parameters()).device.type == "meta"

    tied_backend = build_backend(
        artifacts_dir / "build_model_rank1_cpu",
        rank=1,
        world_size=2,
    )
    with (
        mock.patch.object(
            fsdp_backend_module.AutoConfig,
            "from_pretrained",
            return_value=SimpleNamespace(tie_word_embeddings=True),
        ),
        mock.patch.object(
            fsdp_backend_module.AutoModelForCausalLM,
            "from_config",
            side_effect=lambda *_args, **_kwargs: ToyCausalLM(seed=23),
        ),
    ):
        model, full_state = tied_backend._build_model_and_state_for_fsdp("toy-model")

    assert full_state == {}
    assert next(model.parameters()).device.type == "cpu"


def test_create_fsdp_model_rejects_linear_attention_and_freezes_eval(
    artifacts_dir: Path,
) -> None:
    backend = build_backend(artifacts_dir / "create_fsdp_model")

    with mock.patch.object(
        backend,
        "_build_model_and_state_for_fsdp",
        return_value=(ToyCausalLM(seed=31, linear_attention=True), {}),
    ):
        try:
            backend._create_fsdp_model(
                model_name_or_path="toy-model",
                device=torch.device("cpu"),
                device_mesh="mesh",
                mp_policy=fsdp_backend_module.MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                ),
                offload_policy=fsdp_backend_module.OffloadPolicy(),
                reshard_after_forward=False,
                enable_gradient_checkpointing=False,
                train_mode=True,
            )
        except NotImplementedError as exc:
            assert "full-attention-only" in str(exc)
        else:
            raise AssertionError("expected linear attention rejection")

    candidate = ToyCausalLM(seed=32)
    with (
        mock.patch.object(
            backend,
            "_build_model_and_state_for_fsdp",
            return_value=(candidate, {"weight": torch.ones(1)}),
        ),
        mock.patch.object(
            fsdp_backend_module,
            "apply_fsdp2",
            side_effect=lambda **kwargs: kwargs["model"],
        ) as apply_fsdp2,
        mock.patch.object(
            backend,
            "_load_full_state_dict_from_rank0",
            side_effect=lambda model, *_args, **_kwargs: model,
        ) as load_state,
    ):
        model = backend._create_fsdp_model(
            model_name_or_path="toy-model",
            device=torch.device("cpu"),
            device_mesh="mesh",
            mp_policy=fsdp_backend_module.MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            offload_policy=fsdp_backend_module.OffloadPolicy(),
            reshard_after_forward=True,
            enable_gradient_checkpointing=True,
            train_mode=False,
        )

    assert model is candidate
    assert candidate.gradient_checkpointing_enabled is True
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    apply_fsdp2.assert_called_once()
    load_state.assert_called_once()


def test_create_fsdp_model_passes_configured_shard_modules(
    artifacts_dir: Path,
) -> None:
    backend = build_backend(
        artifacts_dir / "create_fsdp_model_shard_modules",
        shard_modules=["transformer.h"],
    )
    candidate = ToyCausalLM(seed=33)
    with (
        mock.patch.object(
            backend,
            "_build_model_and_state_for_fsdp",
            return_value=(candidate, {}),
        ),
        mock.patch.object(
            fsdp_backend_module,
            "apply_fsdp2",
            side_effect=lambda **kwargs: kwargs["model"],
        ) as apply_fsdp2,
        mock.patch.object(
            backend,
            "_load_full_state_dict_from_rank0",
            side_effect=lambda model, *_args, **_kwargs: model,
        ),
    ):
        backend._create_fsdp_model(
            model_name_or_path="toy-model",
            device=torch.device("cpu"),
            device_mesh="mesh",
            mp_policy=fsdp_backend_module.MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            offload_policy=fsdp_backend_module.OffloadPolicy(),
            reshard_after_forward=False,
            enable_gradient_checkpointing=False,
            train_mode=True,
        )

    assert apply_fsdp2.call_args.kwargs["shard_modules"] == ["transformer.h"]


TESTS: list[tuple[str, Any]] = [
    ("forward_without_reference_model", test_forward_without_reference_model),
    (
        "forward_with_reference_model_and_backward_scaling",
        test_forward_with_reference_model_and_backward_scaling,
    ),
    ("step_optimizer_tracks_lr_schedule", test_step_optimizer_tracks_lr_schedule),
    (
        "checkpoint_load_skips_optimizer_when_requested",
        test_checkpoint_load_skips_optimizer_when_requested,
    ),
    (
        "resolve_checkpoint_path_prefers_latest_rank_local_file",
        test_resolve_checkpoint_path_prefers_latest_rank_local_file,
    ),
    ("sleep_and_wakeup_respect_offload_mode", test_sleep_and_wakeup_respect_offload_mode),
    (
        "offload_and_onload_params_move_gradients",
        test_offload_and_onload_params_move_gradients,
    ),
    (
        "fsdp2_base_infers_common_transformer_block_paths",
        test_fsdp2_base_infers_common_transformer_block_paths,
    ),
    (
        "fsdp2_base_resolve_shard_modules_validates_and_deduplicates",
        test_fsdp2_base_resolve_shard_modules_validates_and_deduplicates,
    ),
    ("apply_fsdp2_shards_blocks_then_root", test_apply_fsdp2_shards_blocks_then_root),
    (
        "create_model_runtime_builds_expected_policies",
        test_create_model_runtime_builds_expected_policies,
    ),
    (
        "build_model_and_state_for_fsdp_uses_rank_specific_paths",
        test_build_model_and_state_for_fsdp_uses_rank_specific_paths,
    ),
    (
        "create_fsdp_model_rejects_linear_attention_and_freezes_eval",
        test_create_fsdp_model_rejects_linear_attention_and_freezes_eval,
    ),
    (
        "create_fsdp_model_passes_configured_shard_modules",
        test_create_fsdp_model_passes_configured_shard_modules,
    ),
]


def run_suite(artifacts_dir: Path) -> int:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")

    for name, test_fn in TESTS:
        print(f"[RUN ] {name}")
        test_fn(artifacts_dir)
        print(f"[PASS] {name}")

    print()
    print(f"All {len(TESTS)} FSDP2 backend tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone CPU tests for the FSDP2 backend."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for temporary artifacts. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    if args.artifacts_dir is None:
        with TemporaryDirectory(prefix="openforge_fsdp2_backend_") as temp_dir:
            return run_suite(Path(temp_dir))

    return run_suite(args.artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())
