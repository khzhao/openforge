# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import logging

import torch
from torch.nn.utils.rnn import pad_sequence

from openforge.data import OpenForgeStore, Trajectory, Turn
from openforge.runtime import create_algorithm

__all__ = ["TrainLoop"]

logger = logging.getLogger(__name__)


class TrainLoop:
    """Background trainer for one session.

    Batch readiness is measured in completed trajectories.
    One completed trajectory still contributes one sample per stored turn.
    """

    POLL_INTERVAL_SECONDS = 0.1
    ROLLOUT_WEIGHT_SYNC_MODE = "distributed"

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
            self.train.global_batch_size
            % (self.world_size * self.train.mini_batch_size)
            == 0
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
        self._task.add_done_callback(self._on_done)

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

    def _on_done(self, task: asyncio.Task[None]) -> None:
        if self._stop or task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        logger.error(
            "train loop failed",
            exc_info=(type(error), error, error.__traceback__),
        )

    async def train_once(self) -> bool:
        ready_groups = await self._list_ready_groups()
        eligible_groups: list[list[Trajectory]] = []
        group_trajectory_counts: list[int] = []
        turns_by_trajectory_id: dict[str, list[Turn]] = {}
        for group in ready_groups:
            group_rollout_version = self.policy_version
            for trajectory in group:
                turns = await self.store.list_turns(trajectory.trajectory_id)
                assert turns
                turns_by_trajectory_id[trajectory.trajectory_id] = turns
                group_rollout_version = min(
                    group_rollout_version,
                    min(turn.rollout_model_version for turn in turns),
                )
            if (
                self.policy_version - group_rollout_version
                > self.train.max_rollout_policy_lag
            ):
                continue
            eligible_groups.append(group)
            group_trajectory_counts.append(len(group))

        selected_group_indexes = self._select_group_indexes(group_trajectory_counts)
        if selected_group_indexes is None:
            return False

        trajectories: list[Trajectory] = []
        trajectory_samples: list[list[dict[str, torch.Tensor]]] = []
        for index in selected_group_indexes:
            group = eligible_groups[index]
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
                trajectory_samples.append(
                    self._build_samples(
                        trajectory,
                        turns_by_trajectory_id[trajectory.trajectory_id],
                        float(advantage),
                    )
                )

        assert len(trajectory_samples) == self.train.global_batch_size
        rank_minibatches_per_update: list[list[dict[str, torch.Tensor]]] = []
        offset = 0
        trajectories_per_minibatch = self.world_size * self.train.mini_batch_size
        while offset < len(trajectory_samples):
            rank_minibatches: list[dict[str, torch.Tensor]] = []
            for _rank in range(self.world_size):
                rank_trajectories = trajectory_samples[
                    offset : offset + self.train.mini_batch_size
                ]
                rank_minibatches.append(
                    self._collate(
                        [
                            sample
                            for trajectory in rank_trajectories
                            for sample in trajectory
                        ]
                    )
                )
                offset += self.train.mini_batch_size
            rank_minibatches_per_update.append(rank_minibatches)
        assert (
            len(rank_minibatches_per_update) * trajectories_per_minibatch
            == self.train.global_batch_size
        )

        next_global_step = self.global_step + 1
        next_policy_version = self.policy_version + 1
        await asyncio.to_thread(
            self._step_and_sync,
            rank_minibatches_per_update,
            global_step=next_global_step,
            policy_version=next_policy_version,
        )

        for trajectory in trajectories:
            await self.store.update_trajectory(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    group_id=trajectory.group_id,
                    status="trained",
                    expected_group_size=trajectory.expected_group_size,
                    final_reward=trajectory.final_reward,
                )
            )

        self.global_step = next_global_step
        self.policy_version = next_policy_version
        return True

    async def _list_ready_groups(self) -> list[list[Trajectory]]:
        trajectories = await self.store.list_trajectories(self.session_id)
        groups: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            group_key = (
                str(trajectory.group_id)
                if trajectory.group_id is not None
                else trajectory.trajectory_id
            )
            groups.setdefault(group_key, []).append(trajectory)

        ready_groups = [
            sorted(group, key=lambda trajectory: trajectory.trajectory_id)
            for group in groups.values()
            if self._is_ready_group(group)
        ]
        ready_groups.sort(key=lambda group: group[0].trajectory_id)
        return ready_groups

    def _is_ready_group(self, group: list[Trajectory]) -> bool:
        if not all(trajectory.status == "completed" for trajectory in group):
            return False
        expected_sizes = {trajectory.expected_group_size for trajectory in group}
        if len(expected_sizes) != 1:
            logger.warning(
                "skipping group with inconsistent expected_group_size: %s",
                [trajectory.trajectory_id for trajectory in group],
            )
            return False
        expected_group_size = next(iter(expected_sizes))
        if len(group) != expected_group_size:
            if len(group) > expected_group_size:
                logger.warning(
                    "skipping oversized group: expected %s trajectories, found %s for %s",
                    expected_group_size,
                    len(group),
                    [trajectory.trajectory_id for trajectory in group],
                )
            return False
        return True

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
        rank_minibatches_per_update: list[list[dict[str, torch.Tensor]]],
        *,
        global_step: int,
        policy_version: int,
    ) -> None:
        self.train_manager.step_update(
            rank_minibatches_per_update,
            global_step=global_step,
        )
        self.train_manager.sync_rollout_weights(
            policy_version=policy_version,
            mode=self.ROLLOUT_WEIGHT_SYNC_MODE,
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
            length = len(turn.token_ids)
            samples.append(
                {
                    "tokens": torch.tensor(
                        turn.token_ids,
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
                    "rollout_log_probs": torch.tensor(
                        turn.rollout_log_probs,
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
            "rollout_log_probs": pad_sequence(
                [sample["rollout_log_probs"] for sample in samples],
                batch_first=True,
            ),
            "lengths": torch.stack([sample["lengths"] for sample in samples]),
        }
