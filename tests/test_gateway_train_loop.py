# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch
from _script_test_utils import install_test_stubs, run_tests

install_test_stubs()

from openforge.configs.algo import GRPOConfig
from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn
from openforge.gateway.train_loop import TrainLoop
from openforge.train.types import TrainStepResult


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

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.mini_batch_size // self.micro_batch_size


class _FakeTrainManager:
    def __init__(
        self,
        *,
        world_size: int,
        global_batch_size: int,
        mini_batch_size: int,
        micro_batch_size: int,
        max_rollout_policy_lag: int,
    ) -> None:
        self.world_size = world_size
        self.cfg = SimpleNamespace(
            algo=GRPOConfig(),
            train=_FakeTrainConfig(
                global_batch_size=global_batch_size,
                mini_batch_size=mini_batch_size,
                micro_batch_size=micro_batch_size,
                max_rollout_policy_lag=max_rollout_policy_lag,
            ),
        )
        self.step_calls: list[tuple[int, list[object]]] = []
        self.sync_calls: list[tuple[int, str | None, int | None]] = []

    def step_update(
        self,
        rank_minibatches_per_update,
        *,
        global_step: int,
    ) -> list[TrainStepResult]:
        assert len(rank_minibatches_per_update) == 1
        rank_minibatches = rank_minibatches_per_update[0]
        self.step_calls.append((global_step, rank_minibatches))
        return [
            TrainStepResult(rank=rank, global_step=global_step, metrics={"lr": 0.1})
            for rank in range(self.world_size)
        ]

    def sync_rollout_weights(
        self,
        *,
        policy_version: int,
        mode: str | None = None,
        bucket_bytes: int | None = None,
    ) -> None:
        self.sync_calls.append((policy_version, mode, bucket_bytes))


def _turn(
    trajectory_id: str,
    *,
    turn_index: int,
    prompt_length: int,
    token_ids: list[int],
    rollout_model_version: int = 0,
) -> Turn:
    return Turn(
        trajectory_id=trajectory_id,
        turn_index=turn_index,
        rollout_model_version=rollout_model_version,
        prompt_length=prompt_length,
        token_ids=token_ids,
        position_ids=list(range(len(token_ids))),
        loss_mask=[False] * max(prompt_length - 1, 0)
        + [True] * (len(token_ids) - prompt_length),
        rollout_log_probs=[0.0] * max(prompt_length - 1, 0)
        + [-0.1] * (len(token_ids) - prompt_length),
    )


async def _create_trajectory(
    store: SQLiteOpenForgeStore,
    trajectory: Trajectory,
) -> None:
    await store.create_trajectories([trajectory])


async def _append_turn(
    store: SQLiteOpenForgeStore,
    turn: Turn,
) -> None:
    await store.append_turns([turn])


async def _completed_trajectory(
    store: SQLiteOpenForgeStore,
    *,
    session_id: str,
    trajectory_id: str,
    reward: float,
    token_ids: list[int],
    prompt_length: int = 2,
    group_id: str | None = None,
    expected_group_size: int = 1,
    rollout_model_version: int = 0,
) -> None:
    await _create_trajectory(
        store,
        Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            group_id=group_id,
            status="completed",
            expected_group_size=expected_group_size,
            final_reward=reward,
        )
    )
    await _append_turn(
        store,
        _turn(
            trajectory_id,
            turn_index=0,
            prompt_length=prompt_length,
            token_ids=token_ids,
            rollout_model_version=rollout_model_version,
        )
    )


async def _completed_multiturn_trajectory(
    store: SQLiteOpenForgeStore,
    *,
    session_id: str,
    trajectory_id: str,
    reward: float,
    turns: list[tuple[int, list[int]]],
    group_id: str | None = None,
    expected_group_size: int = 1,
    rollout_model_version: int = 0,
) -> None:
    await _create_trajectory(
        store,
        Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            group_id=group_id,
            status="completed",
            expected_group_size=expected_group_size,
            final_reward=reward,
        )
    )
    for turn_index, (prompt_length, token_ids) in enumerate(turns):
        await _append_turn(
            store,
            _turn(
                trajectory_id,
                turn_index=turn_index,
                prompt_length=prompt_length,
                token_ids=token_ids,
                rollout_model_version=rollout_model_version,
            )
        )


def test_train_loop_train_once_consumes_one_global_batch() -> None:
    """Consume exactly one global batch worth of completed trajectories."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        for index in range(5):
            await _completed_trajectory(
                store,
                session_id="s0",
                trajectory_id=f"t{index}",
                reward=1.0 + index,
                token_ids=[10 + index, 20 + index, 30 + index],
            )

        train_manager = _FakeTrainManager(
            world_size=2,
            global_batch_size=4,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        assert loop.global_step == 1
        assert loop.policy_version == 1
        assert train_manager.sync_calls == [(1, "distributed", None)]
        assert len(train_manager.step_calls) == 1
        global_step, rank_minibatches = train_manager.step_calls[0]
        assert global_step == 1
        assert len(rank_minibatches) == 2
        assert rank_minibatches[0]["tokens"].tolist() == [[10, 20, 30], [11, 21, 31]]
        torch.testing.assert_close(
            rank_minibatches[0]["rollout_log_probs"],
            torch.tensor(
                [
                    [0.0, -0.1],
                    [0.0, -0.1],
                ]
            ),
            atol=1e-6,
            rtol=1e-6,
        )
        assert rank_minibatches[0]["advantages"].tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert rank_minibatches[0]["lengths"].tolist() == [3, 3]
        assert rank_minibatches[1]["tokens"].tolist() == [[12, 22, 32], [13, 23, 33]]
        assert rank_minibatches[1]["advantages"].tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert rank_minibatches[1]["lengths"].tolist() == [3, 3]

        trained_ids = [
            (await store.get_trajectory(f"t{index}")).status for index in range(4)
        ]
        remaining = await store.get_trajectory("t4")
        assert trained_ids == ["trained", "trained", "trained", "trained"]
        assert remaining is not None
        assert remaining.status == "completed"
        await store.close()

    asyncio.run(run())


def test_train_loop_skips_groups_beyond_rollout_policy_lag() -> None:
    """Skip completed groups that are too stale for the current policy version."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="fresh",
            reward=1.0,
            token_ids=[1, 2, 3],
            rollout_model_version=4,
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="stale",
            reward=2.0,
            token_ids=[4, 5, 6],
            rollout_model_version=1,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=1,
            mini_batch_size=1,
            micro_batch_size=1,
            max_rollout_policy_lag=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)
        loop.policy_version = 5

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [[1, 2, 3]]
        fresh = await store.get_trajectory("fresh")
        stale = await store.get_trajectory("stale")
        assert fresh is not None
        assert stale is not None
        assert fresh.status == "trained"
        assert stale.status == "completed"
        await store.close()

    asyncio.run(run())


def test_train_loop_train_once_splits_rank_local_minibatches() -> None:
    """Split one global batch into rank-local padded minibatches."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            token_ids=[1, 2, 3],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=2.0,
            token_ids=[4, 5, 6, 7],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t2",
            reward=3.0,
            token_ids=[8, 9, 10, 11, 12],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t3",
            reward=4.0,
            token_ids=[13, 14, 15],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=4,
            mini_batch_size=4,
            micro_batch_size=2,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        global_step, rank_minibatches = train_manager.step_calls[0]
        assert global_step == 1
        assert len(rank_minibatches) == 1
        assert rank_minibatches[0]["tokens"].tolist() == [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 12],
            [13, 14, 15, 0, 0],
        ]
        assert rank_minibatches[0]["advantages"].tolist() == [[0.0] * 5] * 4
        assert rank_minibatches[0]["lengths"].tolist() == [3, 4, 5, 3]
        await store.close()

    asyncio.run(run())


def test_train_loop_builds_group_relative_advantages() -> None:
    """Compute group-relative advantages before building training samples."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-a",
            reward=1.5,
            token_ids=[1, 2, 3, 4],
            prompt_length=3,
            group_id="g0",
            expected_group_size=2,
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-b",
            reward=-2.0,
            token_ids=[5, 6, 7],
            prompt_length=1,
            group_id="g0",
            expected_group_size=2,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        torch.testing.assert_close(
            rank_minibatches[0]["advantages"],
            torch.tensor(
                [
                    [0.7071064, 0.7071064, 0.7071064, 0.7071064],
                    [-0.7071064, -0.7071064, -0.7071064, 0.0],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        assert rank_minibatches[0]["loss_mask"].tolist() == [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        await store.close()

    asyncio.run(run())


def test_train_loop_uses_all_turns_in_completed_trajectory() -> None:
    """Include every stored turn from a completed trajectory in training."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_multiturn_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.5,
            turns=[
                (2, [1, 2, 3]),
                (3, [1, 2, 3, 4]),
            ],
            group_id="g0",
            expected_group_size=2,
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=-2.0,
            token_ids=[5, 6, 7],
            prompt_length=1,
            group_id="g0",
            expected_group_size=2,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [
            [1, 2, 3, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 0],
        ]
        torch.testing.assert_close(
            rank_minibatches[0]["advantages"],
            torch.tensor(
                [
                    [0.7071064, 0.7071064, 0.7071064, 0.0],
                    [0.7071064, 0.7071064, 0.7071064, 0.7071064],
                    [-0.7071064, -0.7071064, -0.7071064, 0.0],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        assert rank_minibatches[0]["loss_mask"].tolist() == [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        t0 = await store.get_trajectory("t0")
        t1 = await store.get_trajectory("t1")
        assert t0 is not None
        assert t1 is not None
        assert t0.status == "trained"
        assert t1.status == "trained"
        await store.close()

    asyncio.run(run())


def test_train_loop_counts_batch_size_in_trajectories() -> None:
    """Measure batch readiness in completed trajectories, not stored turns."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_multiturn_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            turns=[
                (2, [1, 2, 3]),
                (2, [1, 2, 4]),
            ],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=9.0,
            token_ids=[7, 8, 9],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [
            [1, 2, 3],
            [1, 2, 4],
            [7, 8, 9],
        ]
        assert rank_minibatches[0]["advantages"].tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        t0 = await store.get_trajectory("t0")
        t1 = await store.get_trajectory("t1")
        assert t0 is not None
        assert t1 is not None
        assert t0.status == "trained"
        assert t1.status == "trained"
        await store.close()

    asyncio.run(run())


def test_train_loop_handles_variable_turn_counts_per_trajectory() -> None:
    """Train once even when trajectories contribute different numbers of turns."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_multiturn_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            turns=[
                (2, [1, 2, 3]),
                (2, [1, 2, 4]),
            ],
        )
        await _completed_multiturn_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=2.0,
            turns=[
                (2, [5, 6, 7]),
                (2, [5, 6, 8]),
            ],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t2",
            reward=3.0,
            token_ids=[9, 10, 11],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t3",
            reward=4.0,
            token_ids=[12, 13, 14],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=4,
            mini_batch_size=4,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, 7],
            [5, 6, 8],
            [9, 10, 11],
            [12, 13, 14],
        ]
        for trajectory_id in ["t0", "t1", "t2", "t3"]:
            trajectory = await store.get_trajectory(trajectory_id)
            assert trajectory is not None
            assert trajectory.status == "trained"
        await store.close()

    asyncio.run(run())


def test_train_loop_waits_for_all_siblings_in_group() -> None:
    """Wait until every sibling trajectory in a group is completed."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-a",
            reward=1.0,
            token_ids=[1, 2, 3],
            group_id="g0",
            expected_group_size=2,
        )
        await _create_trajectory(
            store,
            Trajectory(
                trajectory_id="child-b",
                session_id="s0",
                group_id="g0",
                status="active",
                expected_group_size=2,
            )
        )
        await _append_turn(
            store,
            _turn(
                "child-b",
                turn_index=0,
                prompt_length=2,
                token_ids=[4, 5, 6],
            )
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="solo",
            reward=9.0,
            token_ids=[7, 8, 9],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is False
        assert train_manager.step_calls == []
        await store.close()

    asyncio.run(run())


def test_train_loop_trains_completed_sibling_group_together() -> None:
    """Train completed trajectories from the same group together."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-a",
            reward=1.0,
            token_ids=[1, 2, 3],
            group_id="g0",
            expected_group_size=2,
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-b",
            reward=2.0,
            token_ids=[4, 5, 6, 7],
            group_id="g0",
            expected_group_size=2,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [[1, 2, 3, 0], [4, 5, 6, 7]]
        torch.testing.assert_close(
            rank_minibatches[0]["advantages"],
            torch.tensor(
                [
                    [-0.7071064, -0.7071064, -0.7071064, 0.0],
                    [0.7071064, 0.7071064, 0.7071064, 0.7071064],
                ]
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        await store.close()

    asyncio.run(run())


def test_train_loop_run_polls_until_batch_is_ready() -> None:
    """Poll until a full batch becomes available and then train it once."""

    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            token_ids=[1, 2, 3],
        )
        await _create_trajectory(
            store,
            Trajectory(
                trajectory_id="t1",
                session_id="s0",
                group_id=None,
                status="active",
            )
        )
        await _append_turn(
            store,
            _turn(
                "t1",
                turn_index=0,
                prompt_length=2,
                token_ids=[4, 5, 6],
            )
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
            max_rollout_policy_lag=0,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)
        loop.POLL_INTERVAL_SECONDS = 0.01
        loop.start()

        await asyncio.sleep(0.05)
        assert train_manager.step_calls == []

        await store.update_trajectory(
            Trajectory(
                trajectory_id="t1",
                session_id="s0",
                group_id=None,
                status="completed",
                final_reward=2.0,
            )
        )

        for _ in range(100):
            if train_manager.step_calls:
                break
            await asyncio.sleep(0.01)

        await loop.stop()

        assert len(train_manager.step_calls) == 1
        assert train_manager.sync_calls == [(1, "distributed", None)]
        t0 = await store.get_trajectory("t0")
        t1 = await store.get_trajectory("t1")
        assert t0 is not None
        assert t1 is not None
        assert t0.status == "trained"
        assert t1.status == "trained"
        await store.close()

    asyncio.run(run())


def main() -> int:
    return run_tests(
        [
            test_train_loop_train_once_consumes_one_global_batch,
            test_train_loop_train_once_splits_rank_local_minibatches,
            test_train_loop_builds_group_relative_advantages,
            test_train_loop_uses_all_turns_in_completed_trajectory,
            test_train_loop_counts_batch_size_in_trajectories,
            test_train_loop_handles_variable_turn_counts_per_trajectory,
            test_train_loop_skips_groups_beyond_rollout_policy_lag,
            test_train_loop_waits_for_all_siblings_in_group,
            test_train_loop_trains_completed_sibling_group_together,
            test_train_loop_run_polls_until_batch_is_ready,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
