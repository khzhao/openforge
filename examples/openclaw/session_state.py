# Copyright 2026 openforge

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PendingTurn",
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
