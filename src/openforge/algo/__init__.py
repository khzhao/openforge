# Copyright 2026 openforge

from __future__ import annotations

from typing import Protocol

import torch

from openforge.configs.algo import GRPOConfig, GRPOTISConfig

__all__ = [
    "Algorithm",
    "GRPOAlgorithm",
]


class Algorithm(Protocol):
    """Minimal algorithm boundary for advantage shaping and loss math."""

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute one scalar advantage per trajectory in a completed group."""

    def compute_loss(
        self,
        *,
        curr_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor | None,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        entropy: torch.Tensor | None = None,
        ref_log_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the scalar training loss for one packed sample."""


class GRPOAlgorithm:
    """Simple GRPO implementation for one-pass on-policy updates."""

    def __init__(self, cfg: GRPOConfig | GRPOTISConfig) -> None:
        self.cfg = cfg

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        advantages = rewards.float() - rewards.float().mean()
        if self.cfg.normalize_group_std and rewards.numel() > 1:
            advantages = advantages / (advantages.std() + 1e-6)
        return advantages

    def compute_loss(
        self,
        *,
        curr_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor | None,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        entropy: torch.Tensor | None = None,
        ref_log_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ratio = (curr_log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(
            min=1.0 - self.cfg.clip_range,
            max=1.0 + self.cfg.clip_range,
        )
        if self.cfg.name == "grpo_tis":
            assert rollout_log_probs is not None
            tis = (old_log_probs - rollout_log_probs).exp().clamp(max=self.cfg.tis_cap)
        else:
            tis = 1.0
        policy_loss = -torch.minimum(
            tis * ratio * advantages,
            tis * clipped_ratio * advantages,
        )
        loss = (policy_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        outputs = {
            "loss": loss,
            "pg_loss": loss.detach(),
        }

        entropy_loss = None
        if entropy is not None:
            entropy_loss = (entropy * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            loss = loss - self.cfg.entropy_coef * entropy_loss
            outputs["loss"] = loss
            outputs["entropy_loss"] = entropy_loss.detach()

        if ref_log_probs is None:
            return outputs

        kl = (
            (ref_log_probs - curr_log_probs).exp()
            - 1.0
            - (ref_log_probs - curr_log_probs)
        )
        kl_loss = (kl * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        loss = loss + self.cfg.kl_coef * kl_loss
        outputs["loss"] = loss
        outputs["kl_loss"] = kl_loss.detach()
        return outputs
