# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from openforge.data import Session, SQLiteOpenForgeStore, Trajectory, Turn
from openforge.gateway.runtime import Generation, Runtime
from openforge.gateway.train_loop import TrainLoop
from openforge.gateway.types import (
    ChatChoice,
    ChatMessage,
    CompletionUsage,
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
        self._active_session_id: str | None = None
        self._train_loop: TrainLoop | None = None

    async def list_models(self) -> dict[str, object]:
        return {
            "models": self.runtime.list_models(),
            "active_model": self.runtime.current_model(),
        }

    async def current_session(self) -> StartSessionResponse | None:
        session_id = self._active_session_id
        if session_id is None:
            return None

        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")
        return StartSessionResponse(
            session_id=session.session_id,
            model=session.model_name,
        )

    async def start_session(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> StartSessionResponse:
        if self._active_session_id is not None:
            raise ActiveSessionError("another session is already active")

        model_name = runtime_config.model.model_name_or_path
        if self.runtime.current_model() not in (None, model_name):
            await asyncio.to_thread(self.runtime.shutdown)
        resolved_model_name = await asyncio.to_thread(
            self.runtime.start,
            runtime_config=runtime_config,
        )

        session_id = self._new_id("sess")
        await self.store.create_session(
            Session(session_id=session_id, model_name=resolved_model_name)
        )
        self._active_session_id = session_id
        assert self._train_loop is None
        self._train_loop = TrainLoop(
            session_id=session_id,
            store=self.store,
            train_manager=self.runtime.train_manager(),
        )
        self._train_loop.start()
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
                    token_ids=list(turn.token_ids),
                    position_ids=list(turn.position_ids),
                    loss_mask=list(turn.loss_mask),
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
        messages: list[ChatMessage],
        sampling_params: dict[str, object] | None = None,
    ) -> GenerateResponse:
        session = await self._require_active_session(session_id)
        await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        try:
            input_ids = await asyncio.to_thread(
                self.runtime.tokenize_messages,
                messages,
            )
        except Exception as exc:
            raise ValueError(
                f"failed to tokenize messages with chat template: {exc}"
            ) from exc
        turn_index = len(await self.store.list_turns(trajectory_id))
        generation = await asyncio.to_thread(
            self.runtime.generate,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        await self.store.append_turn(
            self._build_turn(
                trajectory_id=trajectory_id,
                turn_index=turn_index,
                input_ids=input_ids,
                generation=generation,
            )
        )
        return GenerateResponse(
            id=f"chatcmpl_{trajectory_id}_{turn_index}",
            choices=[
                ChatChoice(
                    finish_reason=generation.finish_reason,
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generation.text,
                    ),
                )
            ],
            created=int(time.time()),
            model=session.model_name,
            usage=CompletionUsage(
                completion_tokens=len(generation.token_ids),
                prompt_tokens=len(input_ids),
                total_tokens=len(input_ids) + len(generation.token_ids),
            ),
            metadata={
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "token_ids": generation.token_ids,
                "rollout_model_version": generation.rollout_model_version,
            },
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
                final_reward=final_reward,
            )
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="completed",
        )

    async def error_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
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
                status="errored",
            )
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="errored",
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

        self._active_session_id = None
        if self._train_loop is not None:
            await self._train_loop.stop()
            while await self._train_loop.train_once():
                pass
            self._train_loop = None
        await asyncio.to_thread(self.runtime.shutdown)

        return EndSessionResponse(session_id=session_id, status="completed")

    async def shutdown(self) -> None:
        if self._train_loop is not None:
            await self._train_loop.stop()
            self._train_loop = None
        self._active_session_id = None
        await asyncio.to_thread(self.runtime.shutdown)

    async def _require_active_session(self, session_id: str) -> Session:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")
        if session_id != self._active_session_id:
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

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _build_turn(
        *,
        trajectory_id: str,
        turn_index: int,
        input_ids: list[int],
        generation: Generation,
    ) -> Turn:
        prompt_length = len(input_ids)
        completion_length = len(generation.token_ids)
        prompt_prediction_count = max(prompt_length - 1, 0)
        return Turn(
            trajectory_id=trajectory_id,
            turn_index=turn_index,
            rollout_model_version=generation.rollout_model_version,
            prompt_length=prompt_length,
            token_ids=[*input_ids, *generation.token_ids],
            position_ids=list(range(prompt_length + completion_length)),
            loss_mask=[False] * prompt_prediction_count + [True] * completion_length,
        )
