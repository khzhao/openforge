# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from _script_test_utils import run_tests

from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn


def _turn(trajectory_id: str, turn_index: int) -> Turn:
    token_ids = [10, 11, 12 + turn_index, 20 + turn_index]
    return Turn(
        trajectory_id=trajectory_id,
        turn_index=turn_index,
        rollout_model_version="version-7",
        prompt_length=3,
        token_ids=token_ids,
        position_ids=list(range(len(token_ids))),
        loss_mask=[False, False, True],
    )


def test_sqlite_openforge_store_session_trajectory_and_turn_lifecycle() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "openforge.sqlite3")

            session = Session(session_id="session-0", model_name="model-a")
            root = Trajectory(
                trajectory_id="traj-0",
                session_id=session.session_id,
                parent_trajectory_id=None,
                status="active",
            )

            await store.create_session(session)
            await store.create_trajectory(root)
            await store.append_turn(_turn("traj-0", 0))
            await store.append_turn(_turn("traj-0", 1))

            loaded_session = await store.get_session("session-0")
            loaded_trajectory = await store.get_trajectory("traj-0")
            turns = await store.list_turns("traj-0")

            assert loaded_session == session
            assert loaded_trajectory == root
            assert [turn.turn_index for turn in turns] == [0, 1]
            await store.close()

    asyncio.run(run())


def test_sqlite_openforge_store_filters_by_status_and_model() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "openforge.sqlite3")

            await store.create_session(Session(session_id="s0", model_name="model-a"))
            await store.create_session(Session(session_id="s1", model_name="model-b"))

            await store.create_trajectory(
                Trajectory(
                    trajectory_id="t0",
                    session_id="s0",
                    parent_trajectory_id=None,
                    status="completed",
                    final_reward=1.0,
                )
            )
            await store.create_trajectory(
                Trajectory(
                    trajectory_id="t1",
                    session_id="s0",
                    parent_trajectory_id="t0",
                    status="active",
                )
            )
            await store.create_trajectory(
                Trajectory(
                    trajectory_id="t2",
                    session_id="s1",
                    parent_trajectory_id=None,
                    status="completed",
                    final_reward=2.0,
                )
            )

            session_trajectories = await store.list_trajectories("s0")
            active_trajectories = await store.list_trajectories("s0", status="active")
            completed_all = await store.list_completed_trajectories()
            completed_model_a = await store.list_completed_trajectories(
                model_name="model-a"
            )
            completed_limited = await store.list_completed_trajectories(limit=1)

            assert [
                trajectory.trajectory_id for trajectory in session_trajectories
            ] == [
                "t0",
                "t1",
            ]
            assert [trajectory.trajectory_id for trajectory in active_trajectories] == [
                "t1"
            ]
            assert [trajectory.trajectory_id for trajectory in completed_all] == [
                "t0",
                "t2",
            ]
            assert [trajectory.trajectory_id for trajectory in completed_model_a] == [
                "t0"
            ]
            assert len(completed_limited) == 1
            await store.close()

    asyncio.run(run())


def test_sqlite_openforge_store_updates_trajectory() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteOpenForgeStore(Path(tmpdir) / "openforge.sqlite3")

            await store.create_session(Session(session_id="s0", model_name="model-a"))
            trajectory = Trajectory(
                trajectory_id="t0",
                session_id="s0",
                parent_trajectory_id=None,
                status="active",
            )
            await store.create_trajectory(trajectory)

            await store.update_trajectory(
                Trajectory(
                    trajectory_id="t0",
                    session_id="s0",
                    parent_trajectory_id=None,
                    status="completed",
                    final_reward=3.5,
                )
            )

            updated = await store.get_trajectory("t0")
            assert updated is not None
            assert updated.status == "completed"
            assert updated.final_reward == 3.5
            await store.close()

    asyncio.run(run())


def test_sqlite_openforge_store_returns_none_for_missing_records() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")

        assert await store.get_session("missing") is None
        assert await store.get_trajectory("missing") is None
        assert await store.list_trajectories("missing") == []
        assert await store.list_turns("missing") == []

        await store.close()

    asyncio.run(run())


def test_sqlite_openforge_store_persists_reopened_data_and_orders_turns() -> None:
    async def write(path: Path) -> None:
        store = SQLiteOpenForgeStore(path)
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await store.create_trajectory(
            Trajectory(
                trajectory_id="t0",
                session_id="s0",
                parent_trajectory_id=None,
                status="completed",
                final_reward=1.0,
            )
        )
        await store.append_turn(_turn("t0", 1))
        await store.append_turn(_turn("t0", 0))
        await store.close()

    async def read(path: Path) -> None:
        store = SQLiteOpenForgeStore(path)

        session = await store.get_session("s0")
        trajectory = await store.get_trajectory("t0")
        turns = await store.list_turns("t0")

        assert session == Session(session_id="s0", model_name="model-a")
        assert trajectory is not None
        assert trajectory.status == "completed"
        assert trajectory.final_reward == 1.0
        assert [turn.turn_index for turn in turns] == [0, 1]

        await store.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "openforge.sqlite3"
        asyncio.run(write(path))
        asyncio.run(read(path))


def test_sqlite_openforge_store_rejects_duplicate_turn_indices() -> None:
    async def run() -> None:
        store = SQLiteOpenForgeStore(":memory:")
        await store.create_session(Session(session_id="s0", model_name="model-a"))
        await store.create_trajectory(
            Trajectory(
                trajectory_id="t0",
                session_id="s0",
                parent_trajectory_id=None,
                status="active",
            )
        )
        await store.append_turn(_turn("t0", 0))

        try:
            await store.append_turn(_turn("t0", 0))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("expected duplicate turn index insert to fail")
        finally:
            await store.close()

    asyncio.run(run())


def main() -> int:
    return run_tests(
        [
            test_sqlite_openforge_store_session_trajectory_and_turn_lifecycle,
            test_sqlite_openforge_store_filters_by_status_and_model,
            test_sqlite_openforge_store_updates_trajectory,
            test_sqlite_openforge_store_returns_none_for_missing_records,
            test_sqlite_openforge_store_persists_reopened_data_and_orders_turns,
            test_sqlite_openforge_store_rejects_duplicate_turn_indices,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
