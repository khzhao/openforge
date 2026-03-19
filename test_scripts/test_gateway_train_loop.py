# Copyright 2026 openforge

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
    ) -> None:
        self.global_batch_size = global_batch_size
        self.mini_batch_size = mini_batch_size
        self.micro_batch_size = micro_batch_size

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
    ) -> None:
        self.world_size = world_size
        self.cfg = SimpleNamespace(
            train=_FakeTrainConfig(
                global_batch_size=global_batch_size,
                mini_batch_size=mini_batch_size,
                micro_batch_size=micro_batch_size,
            )
        )
        self.step_calls: list[tuple[int, list[object]]] = []
        self.sync_calls: list[int] = []

    def step(
        self,
        rank_minibatches,
        *,
        global_step: int,
    ) -> list[TrainStepResult]:
        self.step_calls.append((global_step, rank_minibatches))
        return [
            TrainStepResult(rank=rank, global_step=global_step, metrics={"lr": 0.1})
            for rank in range(self.world_size)
        ]

    def sync_rollout_weights(self, *, policy_version: int) -> None:
        self.sync_calls.append(policy_version)


def _turn(
    trajectory_id: str,
    *,
    turn_index: int,
    prompt_length: int,
    input_ids: list[int],
) -> Turn:
    return Turn(
        trajectory_id=trajectory_id,
        turn_index=turn_index,
        rollout_model_version="policy-0",
        prompt_length=prompt_length,
        input_ids=input_ids,
        position_ids=list(range(len(input_ids))),
        loss_mask=[False] * max(prompt_length - 1, 0)
        + [True] * (len(input_ids) - prompt_length),
    )


async def _completed_trajectory(
    store: SQLiteOpenForgeStore,
    *,
    session_id: str,
    trajectory_id: str,
    reward: float,
    input_ids: list[int],
    prompt_length: int = 2,
) -> None:
    await store.create_trajectory(
        Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            parent_trajectory_id=None,
            status="completed",
            final_reward=reward,
        )
    )
    await store.append_turn(
        _turn(
            trajectory_id,
            turn_index=0,
            prompt_length=prompt_length,
            input_ids=input_ids,
        )
    )


async def _completed_multiturn_trajectory(
    store: SQLiteOpenForgeStore,
    *,
    session_id: str,
    trajectory_id: str,
    reward: float,
    turns: list[tuple[int, list[int]]],
) -> None:
    await store.create_trajectory(
        Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            parent_trajectory_id=None,
            status="completed",
            final_reward=reward,
        )
    )
    for turn_index, (prompt_length, input_ids) in enumerate(turns):
        await store.append_turn(
            _turn(
                trajectory_id,
                turn_index=turn_index,
                prompt_length=prompt_length,
                input_ids=input_ids,
            )
        )


def test_train_loop_train_once_consumes_one_global_batch() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        for index in range(5):
            await _completed_trajectory(
                store,
                session_id="s0",
                trajectory_id=f"t{index}",
                reward=1.0 + index,
                input_ids=[10 + index, 20 + index, 30 + index],
            )

        train_manager = _FakeTrainManager(
            world_size=2,
            global_batch_size=4,
            mini_batch_size=2,
            micro_batch_size=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        assert loop.global_step == 1
        assert loop.policy_version == 1
        assert train_manager.sync_calls == [1]
        assert len(train_manager.step_calls) == 1
        global_step, rank_minibatches = train_manager.step_calls[0]
        assert global_step == 1
        assert len(rank_minibatches) == 2
        assert rank_minibatches[0]["tokens"].tolist() == [[10, 20, 30], [11, 21, 31]]
        assert rank_minibatches[0]["advantages"].tolist() == [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
        assert rank_minibatches[0]["lengths"].tolist() == [3, 3]
        assert rank_minibatches[1]["tokens"].tolist() == [[12, 22, 32], [13, 23, 33]]
        assert rank_minibatches[1]["advantages"].tolist() == [
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
        assert rank_minibatches[1]["lengths"].tolist() == [3, 3]

        trained_ids = [
            (await store.get_trajectory(f"t{index}")).status
            for index in range(4)
        ]
        remaining = await store.get_trajectory("t4")
        assert trained_ids == ["trained", "trained", "trained", "trained"]
        assert remaining is not None
        assert remaining.status == "completed"
        await store.close()

    asyncio.run(run())


def test_train_loop_train_once_splits_rank_local_minibatches() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            input_ids=[1, 2, 3],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=2.0,
            input_ids=[4, 5, 6, 7],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t2",
            reward=3.0,
            input_ids=[8, 9, 10, 11, 12],
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t3",
            reward=4.0,
            input_ids=[13, 14, 15],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=4,
            mini_batch_size=4,
            micro_batch_size=2,
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
        assert rank_minibatches[0]["advantages"].tolist() == [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 0.0, 0.0],
        ]
        assert rank_minibatches[0]["lengths"].tolist() == [3, 4, 5, 3]
        await store.close()

    asyncio.run(run())


def test_train_loop_builds_advantages_from_final_reward() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.5,
            input_ids=[1, 2, 3, 4],
            prompt_length=3,
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=-2.0,
            input_ids=[5, 6, 7],
            prompt_length=1,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["advantages"].tolist() == [
            [1.5, 1.5, 1.5, 1.5],
            [-2.0, -2.0, -2.0, 0.0],
        ]
        assert rank_minibatches[0]["loss_mask"].tolist() == [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        await store.close()

    asyncio.run(run())


def test_train_loop_uses_all_turns_in_completed_trajectory() -> None:
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
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t1",
            reward=-2.0,
            input_ids=[5, 6, 7],
            prompt_length=1,
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=3,
            mini_batch_size=3,
            micro_batch_size=1,
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
        assert rank_minibatches[0]["advantages"].tolist() == [
            [1.5, 1.5, 1.5, 0.0],
            [1.5, 1.5, 1.5, 1.5],
            [-2.0, -2.0, -2.0, 0.0],
        ]
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


def test_train_loop_counts_batch_size_in_turns() -> None:
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
            input_ids=[7, 8, 9],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [
            [1, 2, 3],
            [1, 2, 4],
        ]
        t0 = await store.get_trajectory("t0")
        t1 = await store.get_trajectory("t1")
        assert t0 is not None
        assert t1 is not None
        assert t0.status == "trained"
        assert t1.status == "completed"
        await store.close()

    asyncio.run(run())


def test_train_loop_waits_for_all_siblings_in_group() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await store.create_trajectory(
            Trajectory(
                trajectory_id="root",
                session_id="s0",
                parent_trajectory_id=None,
                status="active",
            )
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-a",
            reward=1.0,
            input_ids=[1, 2, 3],
        )
        await store.update_trajectory(
            Trajectory(
                trajectory_id="child-a",
                session_id="s0",
                parent_trajectory_id="root",
                status="completed",
                final_reward=1.0,
            )
        )
        await store.create_trajectory(
            Trajectory(
                trajectory_id="child-b",
                session_id="s0",
                parent_trajectory_id="root",
                status="active",
            )
        )
        await store.append_turn(
            _turn(
                "child-b",
                turn_index=0,
                prompt_length=2,
                input_ids=[4, 5, 6],
            )
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="solo",
            reward=9.0,
            input_ids=[7, 8, 9],
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is False
        assert train_manager.step_calls == []
        await store.close()

    asyncio.run(run())


def test_train_loop_trains_completed_sibling_group_together() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await store.create_trajectory(
            Trajectory(
                trajectory_id="root",
                session_id="s0",
                parent_trajectory_id=None,
                status="active",
            )
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-a",
            reward=1.0,
            input_ids=[1, 2, 3],
        )
        await store.update_trajectory(
            Trajectory(
                trajectory_id="child-a",
                session_id="s0",
                parent_trajectory_id="root",
                status="completed",
                final_reward=1.0,
            )
        )
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="child-b",
            reward=2.0,
            input_ids=[4, 5, 6, 7],
        )
        await store.update_trajectory(
            Trajectory(
                trajectory_id="child-b",
                session_id="s0",
                parent_trajectory_id="root",
                status="completed",
                final_reward=2.0,
            )
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
        )
        loop = TrainLoop(session_id="s0", store=store, train_manager=train_manager)

        trained = await loop.train_once()

        assert trained is True
        _global_step, rank_minibatches = train_manager.step_calls[0]
        assert rank_minibatches[0]["tokens"].tolist() == [[1, 2, 3, 0], [4, 5, 6, 7]]
        assert rank_minibatches[0]["advantages"].tolist() == [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
        await store.close()

    asyncio.run(run())


def test_train_loop_run_polls_until_batch_is_ready() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await _completed_trajectory(
            store,
            session_id="s0",
            trajectory_id="t0",
            reward=1.0,
            input_ids=[1, 2, 3],
        )
        await store.create_trajectory(
            Trajectory(
                trajectory_id="t1",
                session_id="s0",
                parent_trajectory_id=None,
                status="active",
            )
        )
        await store.append_turn(
            _turn(
                "t1",
                turn_index=0,
                prompt_length=2,
                input_ids=[4, 5, 6],
            )
        )

        train_manager = _FakeTrainManager(
            world_size=1,
            global_batch_size=2,
            mini_batch_size=2,
            micro_batch_size=1,
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
                parent_trajectory_id=None,
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
        assert train_manager.sync_calls == [1]
        t0 = await store.get_trajectory("t0")
        t1 = await store.get_trajectory("t1")
        assert t0 is not None
        assert t1 is not None
        assert t0.status == "trained"
        assert t1.status == "trained"
        await store.close()

    asyncio.run(run())
