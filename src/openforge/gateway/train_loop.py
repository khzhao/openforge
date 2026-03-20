# Copyright 2026 openforge

from __future__ import annotations

import asyncio

import torch
from torch.nn.utils.rnn import pad_sequence

from openforge.data import OpenForgeStore, Trajectory, Turn
from openforge.runtime import create_algorithm

__all__ = ["TrainLoop"]


class TrainLoop:
    """Background trainer for one session.

    One completed trajectory contributes one sample per stored turn.
    """

    POLL_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        *,
        session_id: str,
        store: OpenForgeStore,
        train_manager: object,
    ) -> None:
        self.session_id = session_id
        self.store = store
        self.train_manager = train_manager
        self.train = train_manager.cfg.train
        self.algorithm = create_algorithm(train_manager.cfg)
        self.world_size = train_manager.world_size
        assert (
            self.train.global_batch_size == self.world_size * self.train.mini_batch_size
        )
        assert self.train.mini_batch_size % self.train.micro_batch_size == 0
        self.global_step = 0
        self.policy_version = 0
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    def start(self) -> None:
        assert self._task is None
        self._stop = False
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        task = self._task
        assert task is not None
        self._stop = True
        await task
        self._task = None

    async def run(self) -> None:
        while not self._stop:
            trained = await self.train_once()
            if trained:
                continue
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    async def train_once(self) -> bool:
        ready_groups = await self._list_ready_groups()
        group_sample_counts: list[int] = []
        turns_by_trajectory_id: dict[str, list[Turn]] = {}
        for group in ready_groups:
            sample_count = 0
            for trajectory in group:
                turns = await self.store.list_turns(trajectory.trajectory_id)
                assert turns
                turns_by_trajectory_id[trajectory.trajectory_id] = turns
                sample_count += len(turns)
            group_sample_counts.append(sample_count)

        selected_group_indexes = self._select_group_indexes(group_sample_counts)
        if selected_group_indexes is None:
            return False

        trajectories: list[Trajectory] = []
        samples: list[dict[str, torch.Tensor]] = []
        for index in selected_group_indexes:
            group = ready_groups[index]
            rewards = torch.tensor(
                [float(trajectory.final_reward) for trajectory in group],
                dtype=torch.float32,
            )
            group_advantages = self.algorithm.compute_group_advantages(rewards).tolist()
            for trajectory, advantage in zip(
                group,
                group_advantages,
                strict=True,
            ):
                trajectories.append(trajectory)
                samples.extend(
                    self._build_samples(
                        trajectory,
                        turns_by_trajectory_id[trajectory.trajectory_id],
                        float(advantage),
                    )
                )

        assert len(samples) == self.train.global_batch_size
        rank_minibatches: list[dict[str, torch.Tensor]] = []
        offset = 0
        for _rank in range(self.world_size):
            rank_minibatches.append(
                self._collate(samples[offset : offset + self.train.mini_batch_size])
            )
            offset += self.train.mini_batch_size

        next_global_step = self.global_step + 1
        next_policy_version = self.policy_version + 1
        self._step_and_sync(
            rank_minibatches,
            global_step=next_global_step,
            policy_version=next_policy_version,
        )

        for trajectory in trajectories:
            await self.store.update_trajectory(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    parent_trajectory_id=trajectory.parent_trajectory_id,
                    status="trained",
                    final_reward=trajectory.final_reward,
                )
            )

        self.global_step = next_global_step
        self.policy_version = next_policy_version
        return True

    async def _list_ready_groups(self) -> list[list[Trajectory]]:
        trajectories = await self.store.list_trajectories(self.session_id)
        parent_ids = {
            trajectory.parent_trajectory_id
            for trajectory in trajectories
            if trajectory.parent_trajectory_id is not None
        }
        groups: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            if trajectory.parent_trajectory_id is not None:
                groups.setdefault(trajectory.parent_trajectory_id, []).append(
                    trajectory
                )
                continue
            if trajectory.trajectory_id in parent_ids:
                continue
            groups[trajectory.trajectory_id] = [trajectory]

        ready_groups = [
            sorted(group, key=lambda trajectory: trajectory.trajectory_id)
            for group in groups.values()
            if all(trajectory.status == "completed" for trajectory in group)
        ]
        ready_groups.sort(key=lambda group: group[0].trajectory_id)
        return ready_groups

    def _select_group_indexes(
        self,
        group_sample_counts: list[int],
    ) -> list[int] | None:
        target = self.train.global_batch_size
        totals: dict[int, list[int]] = {0: []}
        for index, sample_count in enumerate(group_sample_counts):
            next_totals = dict(totals)
            for total, chosen in totals.items():
                next_total = total + sample_count
                if next_total > target or next_total in next_totals:
                    continue
                next_totals[next_total] = [*chosen, index]
            totals = next_totals
        return totals.get(target)

    def _step_and_sync(
        self,
        rank_minibatches: list[dict[str, torch.Tensor]],
        *,
        global_step: int,
        policy_version: int,
    ) -> None:
        self.train_manager.step(
            rank_minibatches,
            global_step=global_step,
        )
        self.train_manager.sync_rollout_weights(
            policy_version=policy_version,
            mode="distributed",
        )

    def _build_samples(
        self,
        trajectory: Trajectory,
        turns: list[Turn],
        advantage: float,
    ) -> list[dict[str, torch.Tensor]]:
        assert trajectory.final_reward is not None
        samples: list[dict[str, torch.Tensor]] = []
        for turn in turns:
            length = len(turn.input_ids)
            samples.append(
                {
                    "tokens": torch.tensor(
                        turn.input_ids,
                        dtype=torch.long,
                    ),
                    "position_ids": torch.tensor(
                        turn.position_ids,
                        dtype=torch.long,
                    ),
                    "advantages": torch.full(
                        (length,),
                        advantage,
                        dtype=torch.float32,
                    ),
                    "loss_mask": torch.tensor(
                        turn.loss_mask,
                        dtype=torch.float32,
                    ),
                    "lengths": torch.tensor(length, dtype=torch.long),
                }
            )
        return samples

    @staticmethod
    def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
            "lengths": torch.stack([sample["lengths"] for sample in samples]),
        }
