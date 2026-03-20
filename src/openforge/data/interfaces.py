# Copyright 2026 openforge

from __future__ import annotations

from typing import Protocol

from .types import Session, Trajectory, TrajectoryStatus, Turn

__all__ = ["OpenForgeStore"]


class OpenForgeStore(Protocol):
    """Minimal repository interface for OpenForge persistence."""

    async def create_session(self, session: Session) -> None:
        """Insert one session."""

    async def get_session(self, session_id: str) -> Session | None:
        """Load one session by id."""

    async def create_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Insert multiple trajectories."""

    async def get_trajectory(self, trajectory_id: str) -> Trajectory | None:
        """Load one trajectory by id."""

    async def get_trajectories(
        self,
        trajectory_ids: list[str],
    ) -> list[Trajectory]:
        """Load multiple trajectories by id."""

    async def list_trajectories(
        self,
        session_id: str,
        *,
        status: TrajectoryStatus | None = None,
    ) -> list[Trajectory]:
        """List trajectories for one session, optionally filtered by status."""

    async def update_trajectory(self, trajectory: Trajectory) -> None:
        """Persist a trajectory update."""

    async def list_completed_trajectories(
        self,
        *,
        limit: int | None = None,
        model_name: str | None = None,
    ) -> list[Trajectory]:
        """List completed trajectories ready for training consumption."""

    async def append_turns(self, turns: list[Turn]) -> None:
        """Insert multiple turns."""

    async def list_turns(self, trajectory_id: str) -> list[Turn]:
        """List turns for one trajectory in turn order."""
