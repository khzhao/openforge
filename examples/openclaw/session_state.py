# Copyright 2026 openforge

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

__all__ = [
    "PendingTurn",
    "TurnRecord",
    "SessionStateStore",
]


@dataclass(slots=True)
class PendingTurn:
    """One OpenClaw-facing turn awaiting next-state feedback."""

    external_session_id: str
    openforge_session_id: str
    trajectory_id: str
    model_name: str
    request_message_count: int
    assistant_text: str
    created_at: float


@dataclass(slots=True)
class TurnRecord:
    """One persisted OpenClaw turn for visibility and debugging."""

    trajectory_id: str
    external_session_id: str
    openforge_session_id: str
    model_name: str
    turn_type: str
    purpose: str
    request_messages: list[dict[str, Any]]
    normalized_messages: list[dict[str, Any]]
    assistant_text: str
    status: str
    created_at: float
    closed_at: float | None
    reward: float | None
    reason: str | None
    feedback_text: str | None


class SessionStateStore:
    """SQLite-backed mapping from OpenClaw sessions to pending OpenForge turns."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def get_pending(self, external_session_id: str) -> PendingTurn | None:
        """Return the pending turn for one external session."""
        with self._lock, sqlite3.connect(self._path) as conn:
            row = conn.execute(
                """
                SELECT
                    external_session_id,
                    openforge_session_id,
                    trajectory_id,
                    model_name,
                    request_message_count,
                    assistant_text,
                    created_at
                FROM pending_turns
                WHERE external_session_id = ?
                """,
                (external_session_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingTurn(
            external_session_id=str(row[0]),
            openforge_session_id=str(row[1]),
            trajectory_id=str(row[2]),
            model_name=str(row[3]),
            request_message_count=int(row[4]),
            assistant_text=str(row[5]),
            created_at=float(row[6]),
        )

    def save_pending(
        self,
        *,
        external_session_id: str,
        openforge_session_id: str,
        trajectory_id: str,
        model_name: str,
        request_message_count: int,
        assistant_text: str,
    ) -> PendingTurn:
        """Create or replace the pending turn for one external session."""
        pending = PendingTurn(
            external_session_id=external_session_id,
            openforge_session_id=openforge_session_id,
            trajectory_id=trajectory_id,
            model_name=model_name,
            request_message_count=request_message_count,
            assistant_text=assistant_text,
            created_at=time.time(),
        )
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT INTO pending_turns (
                    external_session_id,
                    openforge_session_id,
                    trajectory_id,
                    model_name,
                    request_message_count,
                    assistant_text,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(external_session_id) DO UPDATE SET
                    openforge_session_id = excluded.openforge_session_id,
                    trajectory_id = excluded.trajectory_id,
                    model_name = excluded.model_name,
                    request_message_count = excluded.request_message_count,
                    assistant_text = excluded.assistant_text,
                    created_at = excluded.created_at
                """,
                (
                    pending.external_session_id,
                    pending.openforge_session_id,
                    pending.trajectory_id,
                    pending.model_name,
                    pending.request_message_count,
                    pending.assistant_text,
                    pending.created_at,
                ),
            )
        return pending

    def clear_pending(self, external_session_id: str) -> None:
        """Delete the pending turn for one external session."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                "DELETE FROM pending_turns WHERE external_session_id = ?",
                (external_session_id,),
            )

    def clear_all(self) -> None:
        """Delete all middleware-side state for the OpenClaw example."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute("DELETE FROM pending_turns")
            conn.execute("DELETE FROM turn_records")

    def clear_all(self) -> None:
        """Delete all middleware-side OpenClaw example state."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute("DELETE FROM pending_turns")
            conn.execute("DELETE FROM turn_records")

    def save_turn_record(
        self,
        *,
        trajectory_id: str,
        external_session_id: str,
        openforge_session_id: str,
        model_name: str,
        turn_type: str,
        purpose: str,
        request_messages: list[dict[str, Any]],
        normalized_messages: list[dict[str, Any]],
        assistant_text: str,
    ) -> None:
        """Insert or replace one turn record."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT INTO turn_records (
                    trajectory_id,
                    external_session_id,
                    openforge_session_id,
                    model_name,
                    turn_type,
                    purpose,
                    request_messages_json,
                    normalized_messages_json,
                    assistant_text,
                    status,
                    created_at,
                    closed_at,
                    reward,
                    reason,
                    feedback_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
                ON CONFLICT(trajectory_id) DO UPDATE SET
                    external_session_id = excluded.external_session_id,
                    openforge_session_id = excluded.openforge_session_id,
                    model_name = excluded.model_name,
                    turn_type = excluded.turn_type,
                    purpose = excluded.purpose,
                    request_messages_json = excluded.request_messages_json,
                    normalized_messages_json = excluded.normalized_messages_json,
                    assistant_text = excluded.assistant_text,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (
                    trajectory_id,
                    external_session_id,
                    openforge_session_id,
                    model_name,
                    turn_type,
                    purpose,
                    json.dumps(request_messages, ensure_ascii=False),
                    json.dumps(normalized_messages, ensure_ascii=False),
                    assistant_text,
                    "pending" if purpose == "train" else "non_train",
                    time.time(),
                ),
            )

    def complete_turn_record(
        self,
        *,
        trajectory_id: str,
        reward: float,
        reason: str,
        feedback_text: str,
    ) -> None:
        """Mark a turn record as rewarded and closed."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                UPDATE turn_records
                SET
                    status = ?,
                    closed_at = ?,
                    reward = ?,
                    reason = ?,
                    feedback_text = ?
                WHERE trajectory_id = ?
                """,
                (
                    "rewarded",
                    time.time(),
                    reward,
                    reason,
                    feedback_text,
                    trajectory_id,
                ),
            )

    def mark_turn_record_status(self, *, trajectory_id: str, status: str) -> None:
        """Update the visibility status for one turn record."""
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                UPDATE turn_records
                SET
                    status = ?,
                    closed_at = CASE
                        WHEN ? IN ('failed', 'discarded') THEN ?
                        ELSE closed_at
                    END
                WHERE trajectory_id = ?
                """,
                (status, status, time.time(), trajectory_id),
            )

    def list_turn_records(self, *, limit: int = 20) -> list[TurnRecord]:
        """Return recent turn records newest-first."""
        with self._lock, sqlite3.connect(self._path) as conn:
            rows = conn.execute(
                """
                SELECT
                    trajectory_id,
                    external_session_id,
                    openforge_session_id,
                    model_name,
                    turn_type,
                    purpose,
                    request_messages_json,
                    normalized_messages_json,
                    assistant_text,
                    status,
                    created_at,
                    closed_at,
                    reward,
                    reason,
                    feedback_text
                FROM turn_records
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._deserialize_turn_record(row) for row in rows]

    def list_pending_turn_records(self, *, limit: int = 20) -> list[TurnRecord]:
        """Return recent pending train turn records."""
        with self._lock, sqlite3.connect(self._path) as conn:
            rows = conn.execute(
                """
                SELECT
                    trajectory_id,
                    external_session_id,
                    openforge_session_id,
                    model_name,
                    turn_type,
                    purpose,
                    request_messages_json,
                    normalized_messages_json,
                    assistant_text,
                    status,
                    created_at,
                    closed_at,
                    reward,
                    reason,
                    feedback_text
                FROM turn_records
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._deserialize_turn_record(row) for row in rows]

    def _initialize(self) -> None:
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_turns (
                    external_session_id TEXT PRIMARY KEY,
                    openforge_session_id TEXT NOT NULL,
                    trajectory_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    request_message_count INTEGER NOT NULL,
                    assistant_text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_records (
                    trajectory_id TEXT PRIMARY KEY,
                    external_session_id TEXT NOT NULL,
                    openforge_session_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    turn_type TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    request_messages_json TEXT NOT NULL,
                    normalized_messages_json TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    closed_at REAL,
                    reward REAL,
                    reason TEXT,
                    feedback_text TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS turn_records_external_created_idx
                ON turn_records(external_session_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS turn_records_status_created_idx
                ON turn_records(status, created_at DESC)
                """
            )

    @staticmethod
    def _deserialize_turn_record(row: sqlite3.Row | tuple[Any, ...]) -> TurnRecord:
        return TurnRecord(
            trajectory_id=str(row[0]),
            external_session_id=str(row[1]),
            openforge_session_id=str(row[2]),
            model_name=str(row[3]),
            turn_type=str(row[4]),
            purpose=str(row[5]),
            request_messages=list(json.loads(str(row[6]))),
            normalized_messages=list(json.loads(str(row[7]))),
            assistant_text=str(row[8]),
            status=str(row[9]),
            created_at=float(row[10]),
            closed_at=None if row[11] is None else float(row[11]),
            reward=None if row[12] is None else float(row[12]),
            reason=None if row[13] is None else str(row[13]),
            feedback_text=None if row[14] is None else str(row[14]),
        )
