# Copyright 2026 openforge

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaConfig, LlamaForCausalLM

from openforge.train.fsdp2.base import FSDP2Engine, _selected_token_log_probs
from openforge.train.worker import TrainWorker
from openforge.utils.packed import pack_micro_batch


class _FakeEngine:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.forward_batches: list[dict[str, torch.Tensor]] = []
        self.backward_calls = 0
        self.no_sync_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_batches.append(batch)
        return {"loss": torch.tensor(1.0)}

    def backward(self, outputs: dict[str, torch.Tensor]) -> None:
        assert "loss" in outputs
        self.backward_calls += 1

    def step_optimizer(self) -> dict[str, float]:
        return {"lr": 0.1}

    def compute_old_log_probs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros(batch["loss_mask"].shape[0], dtype=torch.float32)


def _sample(
    tokens: list[int],
    advantage: float,
    *,
    prompt_length: int = 2,
) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "position_ids": torch.arange(len(tokens), dtype=torch.long),
        "advantages": torch.full((len(tokens),), advantage, dtype=torch.float32),
        "loss_mask": torch.tensor(
            [False] * max(prompt_length - 1, 0)
            + [True] * (len(tokens) - prompt_length),
            dtype=torch.float32,
        ),
        "rollout_log_probs": torch.zeros(max(len(tokens) - 1, 0), dtype=torch.float32),
        "lengths": torch.tensor(len(tokens), dtype=torch.long),
    }


def _mini_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "tokens": pad_sequence(
            [sample["tokens"] for sample in samples],
            batch_first=True,
        ),
        "position_ids": pad_sequence(
            [sample["position_ids"] for sample in samples],
            batch_first=True,
        ),
        "advantages": pad_sequence(
            [sample["advantages"] for sample in samples],
            batch_first=True,
        ),
        "loss_mask": pad_sequence(
            [sample["loss_mask"] for sample in samples],
            batch_first=True,
        ),
        "rollout_log_probs": pad_sequence(
            [sample["rollout_log_probs"] for sample in samples],
            batch_first=True,
        ),
        "lengths": torch.stack([sample["lengths"] for sample in samples]),
    }


def test_train_worker_step_chunks_and_packs_local_minibatch() -> None:
    worker = TrainWorker()
    worker.spec = SimpleNamespace(
        rank=0,
        cfg=SimpleNamespace(
            train=SimpleNamespace(
                micro_batch_size=2,
            )
        ),
    )
    worker.engine = _FakeEngine()

    result = worker.step(
        _mini_batch(
            [
                _sample([1, 2, 3], 1.0),
                _sample([4, 5, 6, 7], 2.0),
                _sample([8, 9, 10, 11, 12], 3.0),
                _sample([13, 14, 15], 4.0),
            ]
        ),
        global_step=7,
    )

    assert result.rank == 0
    assert result.global_step == 7
    assert result.metrics == {"lr": 0.1, "global_step": 7.0}
    assert worker.engine.zero_grad_calls == 1
    assert worker.engine.no_sync_calls == 1
    assert worker.engine.backward_calls == 2
    assert len(worker.engine.forward_batches) == 2

    first = worker.engine.forward_batches[0]
    second = worker.engine.forward_batches[1]
    assert first["tokens"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert first["position_ids"].tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert first["loss_mask"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert first["rollout_log_probs"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert first["advantages"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert first["cu_seqlens"].tolist() == [0, 3, 7]
    assert second["tokens"].tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
    assert second["position_ids"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert second["loss_mask"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert second["rollout_log_probs"].tolist() == [0.0] * 7
    assert second["cu_seqlens"].tolist() == [0, 5, 8]


def test_train_worker_only_aggregates_scalar_forward_metrics() -> None:
    worker = TrainWorker()
    worker.spec = SimpleNamespace(
        rank=0,
        cfg=SimpleNamespace(
            train=SimpleNamespace(
                micro_batch_size=2,
            )
        ),
    )
    worker.engine = _FakeEngine()
    worker.engine.forward = lambda batch: {
        "loss": torch.tensor(1.0),
        "pg_loss": torch.tensor(2.0),
        "curr_log_probs": torch.tensor([0.1, 0.2]),
    }

    result = worker.step(
        _mini_batch(
            [
                _sample([1, 2, 3], 1.0),
                _sample([4, 5, 6], 2.0),
            ]
        )
    )

    assert result.metrics == {"pg_loss": 2.0, "lr": 0.1, "global_step": -1.0}


def test_fsdp2_full_grad_norm_uses_full_tensor_for_metric_and_clipping() -> None:
    engine = object.__new__(FSDP2Engine)
    engine.device = torch.device("cpu")
    engine.use_grad_scaler = False
    engine.cfg = SimpleNamespace(
        train=SimpleNamespace(
            gradient_accumulation_steps=1,
            config=SimpleNamespace(
                optim=SimpleNamespace(max_grad_norm=1.0),
            ),
        )
    )
    optimizer_step_calls = 0
    scheduler_step_calls = 0

    def optimizer_step() -> None:
        nonlocal optimizer_step_calls
        optimizer_step_calls += 1

    def scheduler_step() -> None:
        nonlocal scheduler_step_calls
        scheduler_step_calls += 1

    engine.optimizer = SimpleNamespace(
        step=optimizer_step,
        param_groups=[{"lr": 0.25}],
    )
    engine.scheduler = SimpleNamespace(step=scheduler_step)
    engine.main_model = SimpleNamespace(
        parameters=lambda: [
            torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        ]
    )

    class _FakeGradNorm:
        def full_tensor(self) -> torch.Tensor:
            return torch.tensor([5.0], dtype=torch.float64)

    with patch(
        "torch.nn.utils.clip_grad_norm_",
        lambda params, max_grad_norm: _FakeGradNorm(),
    ):
        metrics = engine.step_optimizer()

    assert metrics["grad_norm"] == 5.0
    assert metrics["lr"] == 0.25
    assert optimizer_step_calls == 1
    assert scheduler_step_calls == 1


def test_pack_micro_batch_tracks_cu_seqlens_and_advantages() -> None:
    packed = pack_micro_batch(
        _mini_batch(
            [
                _sample([1, 2, 3, 4], 1.5, prompt_length=3),
                _sample([5, 6, 7], -2.0, prompt_length=1),
                _sample([8, 9], 0.25, prompt_length=2),
            ]
        )
    )

    assert packed["tokens"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert packed["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1]
    assert packed["advantages"].tolist() == [
        1.5,
        1.5,
        1.5,
        1.5,
        -2.0,
        -2.0,
        -2.0,
        0.25,
        0.25,
    ]
    assert packed["loss_mask"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert packed["rollout_log_probs"].tolist() == [0.0] * 8
    assert packed["cu_seqlens"].tolist() == [0, 4, 7, 9]


def _loss(model: LlamaForCausalLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        input_ids=batch["tokens"].unsqueeze(0),
        position_ids=batch["position_ids"].unsqueeze(0),
        use_cache=False,
    ).logits.squeeze(0)[:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = batch["tokens"][1:]
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    numerator = (
        -(batch["advantages"][1:] * token_log_probs) * batch["loss_mask"]
    ).sum()
    denominator = batch["loss_mask"].sum().clamp_min(1.0)
    return numerator / denominator


def test_selected_token_log_probs_matches_log_softmax_gather() -> None:
    torch.manual_seed(0)
    logits = torch.randn(11, 19)
    targets = torch.randint(0, logits.shape[-1], (logits.shape[0],))

    expected = (
        F.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )
    actual = _selected_token_log_probs(logits, targets)

    torch.testing.assert_close(actual, expected)


def _unpacked_loss(
    model: LlamaForCausalLM,
    samples: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    numerator = torch.tensor(0.0)
    denominator = torch.tensor(0.0)
    for sample in samples:
        length = sample["tokens"].numel()
        batch = {
            "tokens": sample["tokens"],
            "position_ids": torch.arange(length, dtype=torch.long),
            "advantages": sample["advantages"],
            "loss_mask": sample["loss_mask"],
        }
        numerator = numerator + _loss(model, batch) * batch["loss_mask"].sum()
        denominator = denominator + batch["loss_mask"].sum()
    return numerator / denominator


def test_hf_style_packing_matches_unpacked_gradients() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    packed_model = LlamaForCausalLM(config).eval()
    unpacked_model = copy.deepcopy(packed_model).eval()

    samples = [
        _sample([1, 2, 3], 1.0),
        _sample([4, 5, 6, 7], 2.0),
    ]
    packed_batch = pack_micro_batch(_mini_batch(samples))

    packed_loss = _loss(packed_model, packed_batch)
    unpacked_loss = _unpacked_loss(unpacked_model, samples)

    packed_loss.backward()
    unpacked_loss.backward()

    assert torch.allclose(packed_loss, unpacked_loss, atol=1e-5, rtol=1e-5)
    for packed_param, unpacked_param in zip(
        packed_model.parameters(),
        unpacked_model.parameters(),
        strict=True,
    ):
        if packed_param.grad is None:
            assert unpacked_param.grad is None
            continue
        assert unpacked_param.grad is not None
        assert torch.allclose(
            packed_param.grad,
            unpacked_param.grad,
            atol=1e-5,
            rtol=1e-5,
        )


def test_hf_style_packing_matches_unpacked_gradients_with_signed_advantages() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    packed_model = LlamaForCausalLM(config).eval()
    unpacked_model = copy.deepcopy(packed_model).eval()

    samples = [
        _sample([1, 2, 3, 4], 1.5, prompt_length=3),
        _sample([5, 6, 7], -0.75, prompt_length=1),
        _sample([8, 9, 10, 11, 12], 0.25, prompt_length=2),
    ]
    packed_batch = pack_micro_batch(_mini_batch(samples))

    packed_loss = _loss(packed_model, packed_batch)
    unpacked_loss = _unpacked_loss(unpacked_model, samples)

    packed_loss.backward()
    unpacked_loss.backward()

    assert torch.allclose(packed_loss, unpacked_loss, atol=1e-5, rtol=1e-5)
    for packed_param, unpacked_param in zip(
        packed_model.parameters(),
        unpacked_model.parameters(),
        strict=True,
    ):
        if packed_param.grad is None:
            assert unpacked_param.grad is None
            continue
        assert unpacked_param.grad is not None
        assert torch.allclose(
            packed_param.grad,
            unpacked_param.grad,
            atol=1e-5,
            rtol=1e-5,
        )


def test_zero_advantages_produce_zero_loss() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    batch = pack_micro_batch(
        _mini_batch(
            [
                _sample([1, 2, 3, 4], 0.0, prompt_length=2),
                _sample([5, 6, 7], 0.0, prompt_length=1),
            ]
        )
    )

    loss = _loss(model, batch)
    loss.backward()

    assert loss.item() == 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        assert torch.count_nonzero(param.grad) == 0
