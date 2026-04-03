# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import torch

from openforge.algo import GRPOAlgorithm
from openforge.configs.algo import GRPOConfig


def test_grpo_group_advantages_are_mean_centered_and_std_normalized() -> None:
    algo = GRPOAlgorithm(GRPOConfig())

    advantages = algo.compute_group_advantages(
        torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([-0.8006408, -0.3202563, 1.1208972]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_grpo_loss_has_policy_gradient_at_unit_ratio() -> None:
    algo = GRPOAlgorithm(GRPOConfig())
    curr_log_probs = torch.tensor([0.1, -0.3, 0.2], requires_grad=True)
    old_log_probs = curr_log_probs.detach().clone()
    advantages = torch.tensor([1.5, -2.0, 0.25])
    loss_mask = torch.tensor([1.0, 1.0, 0.0])

    outputs = algo.compute_loss(
        curr_log_probs=curr_log_probs,
        old_log_probs=old_log_probs,
        rollout_log_probs=None,
        advantages=advantages,
        loss_mask=loss_mask,
    )
    outputs["loss"].backward()

    torch.testing.assert_close(
        curr_log_probs.grad,
        torch.tensor([-0.75, 1.0, -0.0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_grpo_low_var_kl_penalty_is_added() -> None:
    algo = GRPOAlgorithm(GRPOConfig(kl_coef=0.5))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0, 0.2]),
        old_log_probs=torch.tensor([0.0, 0.2]),
        rollout_log_probs=None,
        advantages=torch.tensor([1.0, 1.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
        ref_log_probs=torch.tensor([0.0, -0.1]),
    )

    assert "kl_loss" in outputs
    assert float(outputs["kl_loss"]) > 0.0
    assert float(outputs["loss"]) > float(outputs["pg_loss"])


def test_grpo_entropy_bonus_reduces_loss() -> None:
    algo = GRPOAlgorithm(GRPOConfig(entropy_coef=0.3))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0, 0.0]),
        old_log_probs=torch.tensor([0.0, 0.0]),
        rollout_log_probs=None,
        advantages=torch.tensor([1.0, 1.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
        entropy=torch.tensor([0.5, 1.0]),
    )

    assert "entropy_loss" in outputs
    torch.testing.assert_close(outputs["entropy_loss"], torch.tensor(0.75))
    torch.testing.assert_close(outputs["loss"], outputs["pg_loss"] - torch.tensor(0.225))


def test_grpo_uses_asymmetric_upper_clip_when_configured() -> None:
    algo = GRPOAlgorithm(GRPOConfig(clip_range=0.2, clip_range_high=0.28))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.5]),
        old_log_probs=torch.tensor([0.0]),
        rollout_log_probs=None,
        advantages=torch.tensor([1.0]),
        loss_mask=torch.tensor([1.0]),
    )

    torch.testing.assert_close(outputs["loss"], torch.tensor(-1.28))


def test_grpo_loss_stays_finite_for_large_logprob_deltas() -> None:
    algo = GRPOAlgorithm(GRPOConfig(kl_coef=0.1))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([-1000.0, -900.0]),
        old_log_probs=torch.tensor([0.0, 0.0]),
        rollout_log_probs=None,
        advantages=torch.tensor([1.0, -1.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
        ref_log_probs=torch.tensor([1000.0, 900.0]),
    )

    for value in outputs.values():
        assert torch.isfinite(value).all()
