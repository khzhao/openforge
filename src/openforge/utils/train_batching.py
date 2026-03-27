# Copyright 2026 openforge

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from openforge.utils.packed import pack_micro_batch
from openforge.utils.seqlen_balancing import (
    get_minimum_num_micro_batches,
    get_seqlen_balanced_partitions,
)

__all__ = [
    "build_rank_microbatch_groups",
    "collate_samples",
    "pack_microbatch_group",
    "pack_minibatch",
    "trajectory_total_tokens",
]


def collate_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad and stack per-sample tensors into one dense minibatch."""
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


def trajectory_total_tokens(
    trajectory_samples: list[dict[str, torch.Tensor]],
) -> int:
    """Return the total token count across all samples in a trajectory."""
    return sum(int(sample["lengths"]) for sample in trajectory_samples)


def build_rank_microbatch_groups(
    trajectory_samples: list[list[dict[str, torch.Tensor]]],
    *,
    world_size: int,
    mini_batch_size: int,
    micro_batch_size: int,
    max_tokens_per_micro_batch: int | None,
) -> list[list[list[dict[str, torch.Tensor]]]]:
    """Split trajectories into per-rank minibatches and microbatch groups."""
    rank_minibatches_per_update: list[list[list[dict[str, torch.Tensor]]]] = []
    offset = 0
    trajectories_per_minibatch = world_size * mini_batch_size
    while offset < len(trajectory_samples):
        rank_trajectory_batches: list[list[list[dict[str, torch.Tensor]]]] = []
        for _rank in range(world_size):
            rank_trajectory_batches.append(
                trajectory_samples[offset : offset + mini_batch_size]
            )
            offset += mini_batch_size
        rank_minibatches_per_update.append(
            _build_rank_microbatches(
                rank_trajectory_batches,
                micro_batch_size=micro_batch_size,
                max_tokens_per_micro_batch=max_tokens_per_micro_batch,
            )
        )

    assert offset == len(trajectory_samples)
    assert len(trajectory_samples) % trajectories_per_minibatch == 0
    return rank_minibatches_per_update


def _build_rank_microbatches(
    rank_trajectory_batches: list[list[list[dict[str, torch.Tensor]]]],
    *,
    micro_batch_size: int,
    max_tokens_per_micro_batch: int | None,
) -> list[list[dict[str, torch.Tensor]]]:
    if max_tokens_per_micro_batch is None:
        rank_microbatches = []
        for trajectory_batch in rank_trajectory_batches:
            microbatches = []
            for offset in range(0, len(trajectory_batch), micro_batch_size):
                microbatches.append(
                    collate_samples(
                        [
                            sample
                            for trajectory_samples in trajectory_batch[
                                offset : offset + micro_batch_size
                            ]
                            for sample in trajectory_samples
                        ]
                    )
                )
            rank_microbatches.append(microbatches)
        return rank_microbatches

    micro_batch_count = max(
        max(1, len(rank_trajectory_batches[0]) // max(micro_batch_size, 1)),
        max(
            get_minimum_num_micro_batches(
                [
                    trajectory_total_tokens(trajectory)
                    for trajectory in trajectory_batch
                ],
                max_tokens_per_micro_batch,
            )
            for trajectory_batch in rank_trajectory_batches
        ),
    )
    rank_microbatches = []
    for trajectory_batch in rank_trajectory_batches:
        trajectory_lengths = [
            trajectory_total_tokens(trajectory) for trajectory in trajectory_batch
        ]
        partitions = get_seqlen_balanced_partitions(
            trajectory_lengths,
            micro_batch_count,
            equal_size=False,
        )
        microbatches = []
        for partition in partitions:
            microbatches.append(
                collate_samples(
                    [
                        sample
                        for trajectory_index in partition
                        for sample in trajectory_batch[trajectory_index]
                    ]
                )
            )
        rank_microbatches.append(microbatches)
    return rank_microbatches


def pack_minibatch(
    mini_batch: dict[str, torch.Tensor],
    *,
    micro_batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Slice a dense minibatch into packed microbatches."""
    batch_size = int(mini_batch["lengths"].shape[0])
    assert batch_size > 0, "step requires at least one sample"
    microbatches = [
        {
            key: value[index : index + micro_batch_size]
            for key, value in mini_batch.items()
        }
        for index in range(0, batch_size, micro_batch_size)
    ]
    return [pack_micro_batch(micro_batch) for micro_batch in microbatches]


def pack_microbatch_group(
    microbatch_group: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Pack each already-sliced microbatch in a group."""
    return [pack_micro_batch(microbatch) for microbatch in microbatch_group]
