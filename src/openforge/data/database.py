# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from .interfaces import OpenForgeStore
from .types import Session, Trajectory, TrajectoryStatus, Turn

__all__ = ["SQLiteOpenForgeStore"]


class SQLiteOpenForgeStore(OpenForgeStore):
    """SQLite repository for Session, Trajectory, and Turn records."""

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(resolved)

        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._initialize_schema()

    async def create_session(self, session: Session) -> None:
        async with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, model_name)
                    VALUES (?, ?)
                    """,
                    (session.session_id, session.model_name),
                )

    async def get_session(self, session_id: str) -> Session | None:
        async with self._lock:
            row = self._conn.execute(
                """
                SELECT session_id, model_name
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=str(row["session_id"]),
            model_name=str(row["model_name"]),
        )

    async def create_trajectory(self, trajectory: Trajectory) -> None:
        async with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO trajectories (
                        trajectory_id,
                        session_id,
                        parent_trajectory_id,
                        status,
                        final_reward
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory.trajectory_id,
                        trajectory.session_id,
                        trajectory.parent_trajectory_id,
                        trajectory.status,
                        trajectory.final_reward,
                    ),
                )

    async def get_trajectory(self, trajectory_id: str) -> Trajectory | None:
        async with self._lock:
            row = self._conn.execute(
                """
                SELECT trajectory_id, session_id, parent_trajectory_id, status, final_reward
                FROM trajectories
                WHERE trajectory_id = ?
                """,
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._deserialize_trajectory(row)

    async def list_trajectories(
        self,
        session_id: str,
        *,
        status: TrajectoryStatus | None = None,
    ) -> list[Trajectory]:
        query = """
            SELECT trajectory_id, session_id, parent_trajectory_id, status, final_reward
            FROM trajectories
            WHERE session_id = ?
        """
        params: list[object] = [session_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY trajectory_id ASC"

        async with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._deserialize_trajectory(row) for row in rows]

    async def update_trajectory(self, trajectory: Trajectory) -> None:
        async with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE trajectories
                    SET
                        session_id = ?,
                        parent_trajectory_id = ?,
                        status = ?,
                        final_reward = ?
                    WHERE trajectory_id = ?
                    """,
                    (
                        trajectory.session_id,
                        trajectory.parent_trajectory_id,
                        trajectory.status,
                        trajectory.final_reward,
                        trajectory.trajectory_id,
                    ),
                )

    async def list_completed_trajectories(
        self,
        *,
        limit: int | None = None,
        model_name: str | None = None,
    ) -> list[Trajectory]:
        query = """
            SELECT
                t.trajectory_id,
                t.session_id,
                t.parent_trajectory_id,
                t.status,
                t.final_reward
            FROM trajectories AS t
            JOIN sessions AS s
                ON s.session_id = t.session_id
            WHERE t.status = 'completed'
        """
        params: list[object] = []
        if model_name is not None:
            query += " AND s.model_name = ?"
            params.append(model_name)
        query += " ORDER BY t.trajectory_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        async with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._deserialize_trajectory(row) for row in rows]

    async def append_turn(self, turn: Turn) -> None:
        async with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO turns (
                        trajectory_id,
                        turn_index,
                        rollout_model_version,
                        prompt_length,
                        input_ids_json,
                        position_ids_json,
                        loss_mask_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.trajectory_id,
                        turn.turn_index,
                        turn.rollout_model_version,
                        turn.prompt_length,
                        json.dumps(turn.input_ids),
                        json.dumps(turn.position_ids),
                        json.dumps(turn.loss_mask),
                    ),
                )

    async def list_turns(self, trajectory_id: str) -> list[Turn]:
        async with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    trajectory_id,
                    turn_index,
                    rollout_model_version,
                    prompt_length,
                    input_ids_json,
                    position_ids_json,
                    loss_mask_json
                FROM turns
                WHERE trajectory_id = ?
                ORDER BY turn_index ASC
                """,
                (trajectory_id,),
            ).fetchall()
        return [self._deserialize_turn(row) for row in rows]

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    parent_trajectory_id TEXT,
                    status TEXT NOT NULL,
                    final_reward REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    trajectory_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    rollout_model_version TEXT NOT NULL,
                    prompt_length INTEGER NOT NULL,
                    input_ids_json TEXT NOT NULL,
                    position_ids_json TEXT NOT NULL,
                    loss_mask_json TEXT NOT NULL,
                    PRIMARY KEY (trajectory_id, turn_index),
                    FOREIGN KEY(trajectory_id) REFERENCES trajectories(trajectory_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS trajectories_session_status_idx
                ON trajectories(session_id, status, trajectory_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS turns_trajectory_turn_idx
                ON turns(trajectory_id, turn_index)
                """
            )

    @staticmethod
    def _deserialize_trajectory(row: sqlite3.Row) -> Trajectory:
        return Trajectory(
            trajectory_id=str(row["trajectory_id"]),
            session_id=str(row["session_id"]),
            parent_trajectory_id=row["parent_trajectory_id"],
            status=str(row["status"]),
            final_reward=row["final_reward"],
        )

    @staticmethod
    def _deserialize_turn(row: sqlite3.Row) -> Turn:
        return Turn(
            trajectory_id=str(row["trajectory_id"]),
            turn_index=int(row["turn_index"]),
            rollout_model_version=str(row["rollout_model_version"]),
            prompt_length=int(row["prompt_length"]),
            input_ids=list(json.loads(str(row["input_ids_json"]))),
            position_ids=list(json.loads(str(row["position_ids_json"]))),
            loss_mask=list(json.loads(str(row["loss_mask_json"]))),
        )
