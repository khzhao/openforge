# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from _script_test_utils import install_test_stubs, run_tests

install_test_stubs()

runtime_module = ModuleType("openforge.runtime")


class _Algorithm:
    def compute_group_advantages(self, rewards):
        return rewards - rewards.mean()


def _create_algorithm(_cfg):
    return _Algorithm()


runtime_module.create_algorithm = _create_algorithm
sys.modules["openforge.runtime"] = runtime_module

from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn
from openforge.gateway.train_loop import TrainLoop


class _FakeTrainConfig:
    def __init__(
        self,
        *,
        global_batch_size: int,
        mini_batch_size: int,
        micro_batch_size: int,
        max_rollout_policy_lag: int,
    ) -> None:
        self.global_batch_size = global_batch_size
        self.mini_batch_size = mini_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_rollout_policy_lag = max_rollout_policy_lag


class _FakeTrainManager:
    def __init__(
        self,
        *,
        global_batch_size: int,
        mini_batch_size: int,
        micro_batch_size: int,
        max_rollout_policy_lag: int,
    ) -> None:
        self.world_size = 1
        self.cfg = SimpleNamespace(
            algo=SimpleNamespace(name="grpo"),
            train=_FakeTrainConfig(
                global_batch_size=global_batch_size,
                mini_batch_size=mini_batch_size,
                micro_batch_size=micro_batch_size,
                max_rollout_policy_lag=max_rollout_policy_lag,
            ),
        )
        self.step_calls: list[int] = []
        self.sync_calls: list[int] = []

    def step_update(
        self,
        rank_minibatches_per_update,
        *,
        global_step: int,
    ) -> None:
        self.step_calls.append(global_step)

    def sync_rollout_weights(
        self,
        *,
        policy_version: int,
        mode: str | None = None,
        bucket_bytes: int | None = None,
    ) -> None:
        self.sync_calls.append(policy_version)


def _turn(trajectory_id: str) -> Turn:
    return Turn(
        trajectory_id=trajectory_id,
        turn_index=0,
        rollout_model_version=0,
        prompt_length=2,
        token_ids=[1, 2, 3],
        position_ids=[0, 1, 2],
        loss_mask=[False, True],
        rollout_log_probs=[0.0, -0.1],
    )


async def _create_completed_trajectories(
    store: SQLiteOpenForgeStore,
    trajectories: list[Trajectory],
) -> None:
    await store.create_trajectories(trajectories)
    await store.append_turns([_turn(trajectory.trajectory_id) for trajectory in trajectories])


def test_train_loop_waits_for_expected_group_size() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _create_completed_trajectories(
            store,
            [
                Trajectory(
                    trajectory_id="t0",
                    session_id="s0",
                    group_id="g0",
                    status="completed",
                    expected_group_size=3,
                    final_reward=1.0,
                ),
                Trajectory(
                    trajectory_id="t1",
                    session_id="s0",
                    group_id="g0",
                    status="completed",
                    expected_group_size=3,
                    final_reward=2.0,
                ),
            ],
        )

        loop = TrainLoop(
            session_id="s0",
            store=store,
            train_manager=_FakeTrainManager(
                global_batch_size=2,
                mini_batch_size=2,
                micro_batch_size=1,
                max_rollout_policy_lag=0,
            ),
        )

        assert await loop.train_once() is False
        assert loop.global_step == 0
        await store.close()

    asyncio.run(run())


def test_train_loop_trains_when_group_reaches_expected_size() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _create_completed_trajectories(
            store,
            [
                Trajectory(
                    trajectory_id="t0",
                    session_id="s0",
                    group_id="g0",
                    status="completed",
                    expected_group_size=2,
                    final_reward=1.0,
                ),
                Trajectory(
                    trajectory_id="t1",
                    session_id="s0",
                    group_id="g0",
                    status="completed",
                    expected_group_size=2,
                    final_reward=2.0,
                ),
            ],
        )

        train_manager = _FakeTrainManager(
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        assert await loop.train_once() is True
        assert loop.global_step == 1
        trained = await store.list_trajectories("s0", status="trained")
        assert [trajectory.trajectory_id for trajectory in trained] == ["t0", "t1"]
        assert all(trajectory.expected_group_size == 2 for trajectory in trained)
        await store.close()

    asyncio.run(run())


def main() -> int:
    return run_tests(
        [
            test_train_loop_waits_for_expected_group_size,
            test_train_loop_trains_when_group_reaches_expected_size,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
