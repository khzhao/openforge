# Copyright 2026 openforge

from __future__ import annotations

import asyncio
from uuid import uuid4

from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn
from openforge.gateway.runtime import Generation, Runtime
from openforge.gateway.types import (
    EndSessionResponse,
    EndTrajectoryResponse,
    GenerateResponse,
    RuntimeConfig,
    StartSessionResponse,
    StartTrajectoryResponse,
)

__all__ = [
    "ActiveSessionError",
    "ActiveTrajectoriesRemainError",
    "Service",
    "SessionClosedError",
    "SessionNotFoundError",
    "TrajectoryNotActiveError",
    "TrajectoryNotFoundError",
]


class ActiveSessionError(RuntimeError):
    """Raised when another session is already active."""


class SessionNotFoundError(RuntimeError):
    """Raised when a session id does not exist."""


class SessionClosedError(RuntimeError):
    """Raised when a session is no longer active."""


class TrajectoryNotFoundError(RuntimeError):
    """Raised when a trajectory id does not exist."""


class TrajectoryNotActiveError(RuntimeError):
    """Raised when a trajectory is not active."""


class ActiveTrajectoriesRemainError(RuntimeError):
    """Raised when a session still has active trajectories."""


class Service:
    """Own the session, trajectory, and turn-recording workflow."""

    def __init__(
        self,
        *,
        store: SQLiteOpenForgeStore,
        runtime: Runtime,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self._active_session_ids: set[str] = set()

    async def list_models(self) -> dict[str, object]:
        return {
            "models": self.runtime.list_models(),
            "active_model": self.runtime.current_model(),
        }

    async def current_session(self) -> StartSessionResponse | None:
        if not self._active_session_ids:
            return None

        active_session_id = next(iter(self._active_session_ids))
        session = await self.store.get_session(active_session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {active_session_id}")
        return StartSessionResponse(
            session_id=session.session_id,
            model=session.model_name,
        )

    async def start_session(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> StartSessionResponse:
        if self._active_session_ids:
            raise ActiveSessionError("another session is already active")

        model_name = runtime_config.model.model_name_or_path
        if self.runtime.current_model() not in (None, model_name):
            await self._shutdown_runtime()
        resolved_model_name = self.runtime.start(
            runtime_config=runtime_config,
        )

        session_id = self._new_id("sess")
        await self.store.create_session(
            Session(session_id=session_id, model_name=resolved_model_name)
        )
        self._active_session_ids.add(session_id)
        return StartSessionResponse(session_id=session_id, model=resolved_model_name)

    async def start_trajectory(
        self,
        *,
        session_id: str,
        parent_trajectory_id: str | None = None,
    ) -> StartTrajectoryResponse:
        await self._require_active_session(session_id)

        trajectory_id = self._new_id("traj")
        parent_turns: list[Turn] = []
        if parent_trajectory_id is not None:
            parent = await self._require_active_trajectory(
                session_id=session_id,
                trajectory_id=parent_trajectory_id,
            )
            parent_turns = await self.store.list_turns(parent.trajectory_id)

        await self.store.create_trajectory(
            Trajectory(
                trajectory_id=trajectory_id,
                session_id=session_id,
                parent_trajectory_id=parent_trajectory_id,
                status="active",
            )
        )

        for turn in parent_turns:
            await self.store.append_turn(
                Turn(
                    trajectory_id=trajectory_id,
                    turn_index=turn.turn_index,
                    rollout_model_version=turn.rollout_model_version,
                    prompt_length=turn.prompt_length,
                    input_ids=list(turn.input_ids),
                    position_ids=list(turn.position_ids),
                    loss_mask=list(turn.loss_mask),
                    old_logprobs=list(turn.old_logprobs),
                )
            )

        return StartTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            parent_trajectory_id=parent_trajectory_id,
        )

    async def generate(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        messages: list[dict[str, str]],
        sampling_params: dict[str, object] | None = None,
    ) -> GenerateResponse:
        session = await self._require_active_session(session_id)
        await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        prompt_token_ids = self.runtime.tokenize_messages(session.model_name, messages)
        turn_index = len(await self.store.list_turns(trajectory_id))
        generation = self.runtime.generate(
            session.model_name,
            prompt_token_ids=prompt_token_ids,
            sampling_params=None if sampling_params is None else dict(sampling_params),
        )
        await self.store.append_turn(
            self._build_turn(
                trajectory_id=trajectory_id,
                turn_index=turn_index,
                prompt_token_ids=prompt_token_ids,
                generation=generation,
            )
        )
        return GenerateResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            token_ids=list(generation.token_ids),
            logprobs=list(generation.logprobs),
            finish_reason=generation.finish_reason,
            rollout_model_version=generation.rollout_model_version,
        )

    async def end_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        final_reward: float,
    ) -> EndTrajectoryResponse:
        trajectory = await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        await self.store.update_trajectory(
            Trajectory(
                trajectory_id=trajectory.trajectory_id,
                session_id=trajectory.session_id,
                parent_trajectory_id=trajectory.parent_trajectory_id,
                status="completed",
                final_reward=float(final_reward),
            )
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="completed",
        )

    async def end_session(
        self,
        *,
        session_id: str,
    ) -> EndSessionResponse:
        await self._require_active_session(session_id)
        active_trajectories = await self.store.list_trajectories(
            session_id,
            status="active",
        )
        if active_trajectories:
            raise ActiveTrajectoriesRemainError(
                "all trajectories must be ended before ending the session"
            )

        self._active_session_ids.discard(session_id)
        if not self._active_session_ids:
            await self._shutdown_runtime()

        return EndSessionResponse(session_id=session_id, status="completed")

    async def _require_active_session(self, session_id: str) -> Session:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")
        if session_id not in self._active_session_ids:
            raise SessionClosedError(f"session {session_id} is not active")
        return session

    async def _require_active_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
    ) -> Trajectory:
        trajectory = await self.store.get_trajectory(trajectory_id)
        if trajectory is None or trajectory.session_id != session_id:
            raise TrajectoryNotFoundError(f"unknown trajectory_id: {trajectory_id}")
        if trajectory.status != "active":
            raise TrajectoryNotActiveError(f"trajectory {trajectory_id} is not active")
        return trajectory

    async def _shutdown_runtime(self) -> None:
        await asyncio.to_thread(self.runtime.shutdown)

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _build_turn(
        *,
        trajectory_id: str,
        turn_index: int,
        prompt_token_ids: list[int],
        generation: Generation,
    ) -> Turn:
        prompt_length = len(prompt_token_ids)
        completion_length = len(generation.token_ids)
        prompt_prediction_count = max(prompt_length - 1, 0)
        return Turn(
            trajectory_id=trajectory_id,
            turn_index=turn_index,
            rollout_model_version=generation.rollout_model_version,
            prompt_length=prompt_length,
            input_ids=[*prompt_token_ids, *generation.token_ids],
            position_ids=list(range(prompt_length + completion_length)),
            loss_mask=[False] * prompt_prediction_count + [True] * completion_length,
            old_logprobs=[0.0] * prompt_prediction_count + list(generation.logprobs),
        )
