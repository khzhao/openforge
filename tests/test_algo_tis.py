# Copyright 2026 openforge

from __future__ import annotations

import torch

from _script_test_utils import run_tests

from openforge.algo import GRPOAlgorithm
from openforge.configs.algo import GRPOConfig, GRPOTISConfig


def test_grpo_tis_truncates_behavior_correction() -> None:
    algo = GRPOAlgorithm(GRPOTISConfig(tis_cap=2.0))
    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0]),
        old_log_probs=torch.tensor([0.0]),
        rollout_log_probs=torch.tensor([-10.0]),
        advantages=torch.tensor([1.0]),
        loss_mask=torch.tensor([1.0]),
    )
    torch.testing.assert_close(outputs["loss"], torch.tensor(-2.0))


def test_grpo_ignores_rollout_log_probs() -> None:
    algo = GRPOAlgorithm(GRPOConfig())
    outputs = algo.compute_loss(
        curr_log_probs=torch.tensor([0.0]),
        old_log_probs=torch.tensor([0.0]),
        rollout_log_probs=None,
        advantages=torch.tensor([1.0]),
        loss_mask=torch.tensor([1.0]),
    )
    torch.testing.assert_close(outputs["loss"], torch.tensor(-1.0))


def main() -> int:
    return run_tests(
        [
            test_grpo_tis_truncates_behavior_correction,
            test_grpo_ignores_rollout_log_probs,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
