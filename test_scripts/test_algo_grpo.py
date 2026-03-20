# Copyright 2026 openforge

from __future__ import annotations

import torch

from openforge.algo import GRPOAlgorithm
from openforge.configs.algo import AlgorithmConfig


def test_grpo_group_advantages_are_mean_centered_and_std_normalized() -> None:
    algo = GRPOAlgorithm(AlgorithmConfig())

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
    algo = GRPOAlgorithm(AlgorithmConfig())
    curr_log_probs = torch.tensor([0.1, -0.3, 0.2], requires_grad=True)
    advantages = torch.tensor([1.5, -2.0, 0.25])
    loss_mask = torch.tensor([1.0, 1.0, 0.0])

    outputs = algo.compute_loss(
        curr_log_probs=curr_log_probs,
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
    algo = GRPOAlgorithm(AlgorithmConfig(kl_coef=0.5))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0, 0.2]),
        advantages=torch.tensor([1.0, 1.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
        ref_log_probs=torch.tensor([0.0, -0.1]),
    )

    assert "kl_loss" in outputs
    assert float(outputs["kl_loss"]) > 0.0
    assert float(outputs["loss"]) > float(outputs["pg_loss"])


def test_grpo_entropy_bonus_reduces_loss() -> None:
    algo = GRPOAlgorithm(AlgorithmConfig(entropy_coef=0.3))

    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0, 0.0]),
        advantages=torch.tensor([1.0, 1.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
        entropy=torch.tensor([0.5, 1.0]),
    )

    assert "entropy_loss" in outputs
    torch.testing.assert_close(outputs["entropy_loss"], torch.tensor(0.75))
    torch.testing.assert_close(outputs["loss"], outputs["pg_loss"] - torch.tensor(0.225))
