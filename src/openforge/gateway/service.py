# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from uuid import uuid4

from openforge.data import OpenForgeStore, Session, Trajectory, Turn
from openforge.gateway.runtime import Generation, Runtime
from openforge.gateway.train_loop import TrainLoop
from openforge.gateway.types import (
    ChatChoice,
    ChatMessage,
    CompletionUsage,
    EndSessionResponse,
    EndTrajectoriesResponse,
    EndTrajectoryResponse,
    ExportCheckpointResponse,
    GenerateResponse,
    RuntimeConfig,
    StartSessionResponse,
    StartTrajectoryGroupsResponse,
    StartTrajectoryResponse,
)

__all__ = ["Service"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingGenerate:
    session_id: str
    trajectory_id: str
    group_id: str | None
    messages: list[ChatMessage]
    sampling_params: dict[str, object]
    future: asyncio.Future[GenerateResponse]
    batch_key: tuple[str, str]


@dataclass(slots=True)
class _PendingFinish:
    trajectory: Trajectory
    turns: list[Turn]


class Service:
    """Own the session, trajectory, and turn-recording workflow."""

    GENERATE_BATCH_MAX_SIZE = 320
    GENERATE_BATCH_MAX_WAIT_SECONDS = 0.02

    def __init__(
        self,
        *,
        store: OpenForgeStore,
        runtime: Runtime,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self._active_session_id: str | None = None
        self._active_session_model_name: str | None = None
        self._active_trajectories: dict[str, Trajectory] = {}
        self._active_turns: dict[str, list[Turn]] = {}
        self._train_loop: TrainLoop | None = None
        self._generate_parallelism = 1
        self._generate_lock = asyncio.Lock()
        self._generate_workers = 0
        self._pending_generates: list[_PendingGenerate] = []
        self._finish_lock = asyncio.Lock()
        self._finish_task: asyncio.Task[None] | None = None
        self._pending_finishes: list[_PendingFinish] = []

    async def list_models(self) -> dict[str, object]:
        return {
            "models": self.runtime.list_models(),
            "active_model": self.runtime.current_model(),
        }

    async def current_session(self) -> StartSessionResponse | None:
        session_id = self._active_session_id
        if session_id is None:
            return None

        policy_version = 0 if self._train_loop is None else self._train_loop.policy_version
        model_name = self._active_session_model_name
        if model_name is not None:
            return StartSessionResponse(
                session_id=session_id,
                model=model_name,
                policy_version=policy_version,
            )

        session = await self.store.get_session(session_id)
        if session is None:
            raise Exception(f"unknown session_id: {session_id}")
        return StartSessionResponse(
            session_id=session.session_id,
            model=session.model_name,
            policy_version=policy_version,
        )

    async def start_session(
        self,
        *,
        runtime_config: RuntimeConfig,
    ) -> StartSessionResponse:
        if self._active_session_id is not None:
            raise Exception("another session is already active")

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
        self._active_session_model_name = resolved_model_name
        self._active_trajectories = {}
        self._active_turns = {}
        self._generate_parallelism = runtime_config.rollout.num_engine_replicas
        self._generate_workers = 0
        assert self._train_loop is None
        self._train_loop = TrainLoop(
            session_id=session_id,
            store=self.store,
            train_manager=self.runtime.train_manager(),
        )
        self._train_loop.start()
        return StartSessionResponse(
            session_id=session_id,
            model=resolved_model_name,
            policy_version=0,
        )

    async def start_trajectory(
        self,
        *,
        session_id: str,
        group_id: str | None = None,
    ) -> StartTrajectoryResponse:
        await self._require_active_session(session_id)
        trajectory_id = self._new_id("traj")
        self._active_trajectories[trajectory_id] = Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            group_id=group_id,
            status="active",
        )
        self._active_turns[trajectory_id] = []
        return StartTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            group_id=group_id,
        )

    async def start_trajectory_groups(
        self,
        *,
        session_id: str,
        counts: list[int],
        group_ids: list[str | None],
    ) -> StartTrajectoryGroupsResponse:
        await self._require_active_session(session_id)
        if len(counts) != len(group_ids):
            raise Exception("counts must align with group_ids")
        if not counts:
            return StartTrajectoryGroupsResponse(session_id=session_id, trajectory_ids=[])

        trajectory_ids_per_group: list[list[str]] = []
        for count, group_id in zip(counts, group_ids, strict=True):
            if count <= 0:
                raise Exception("count must be >= 1")
            trajectory_ids = [self._new_id("traj") for _ in range(count)]
            for trajectory_id in trajectory_ids:
                trajectory = Trajectory(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    group_id=group_id,
                    status="active",
                )
                self._active_trajectories[trajectory_id] = trajectory
                self._active_turns[trajectory_id] = []
            trajectory_ids_per_group.append(trajectory_ids)
        return StartTrajectoryGroupsResponse(
            session_id=session_id,
            trajectory_ids=trajectory_ids_per_group,
        )

    async def generate(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        group_id: str | None = None,
        messages: list[ChatMessage],
        sampling_params: dict[str, object] | None = None,
    ) -> GenerateResponse:
        if session_id != self._active_session_id:
            await self._require_active_session(session_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerateResponse] = loop.create_future()
        request = _PendingGenerate(
            session_id=session_id,
            trajectory_id=trajectory_id,
            group_id=group_id,
            messages=messages,
            sampling_params={} if sampling_params is None else dict(sampling_params),
            future=future,
            batch_key=(
                session_id,
                self._sampling_params_key(
                    {} if sampling_params is None else sampling_params
                ),
            ),
        )
        async with self._generate_lock:
            self._pending_generates.append(request)
            self._ensure_generate_workers_locked()
        return await future

    async def end_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
        final_reward: float,
    ) -> EndTrajectoryResponse:
        await self.end_trajectories(
            session_id=session_id,
            trajectory_ids=[trajectory_id],
            final_rewards=[final_reward],
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="completed",
        )

    async def end_trajectories(
        self,
        *,
        session_id: str,
        trajectory_ids: list[str],
        final_rewards: list[float],
    ) -> EndTrajectoriesResponse:
        if len(trajectory_ids) != len(final_rewards):
            raise Exception("trajectory_ids must align with final_rewards")
        if not trajectory_ids:
            return EndTrajectoriesResponse(
                session_id=session_id,
                trajectory_ids=[],
                status="completed",
            )

        trajectories_to_update: list[Trajectory] = []
        for trajectory_id, final_reward in zip(
            trajectory_ids,
            final_rewards,
            strict=True,
        ):
            trajectory = self._active_trajectories.get(trajectory_id)
            if trajectory is None:
                stored_trajectory = await self.store.get_trajectory(trajectory_id)
                if stored_trajectory is None or stored_trajectory.session_id != session_id:
                    raise Exception(f"unknown trajectory_id: {trajectory_id}")
                if stored_trajectory.status == "completed":
                    continue
                raise Exception(f"trajectory {trajectory_id} is not active")
            if trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            if trajectory.status != "active":
                raise Exception(f"trajectory {trajectory_id} is not active")
            trajectories_to_update.append(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    group_id=trajectory.group_id,
                    status="completed",
                    final_reward=final_reward,
                )
            )
        await self._finish_trajectories(trajectories_to_update)
        return EndTrajectoriesResponse(
            session_id=session_id,
            trajectory_ids=[
                trajectory.trajectory_id for trajectory in trajectories_to_update
            ],
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
        await self._finish_trajectories(
            [
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    group_id=trajectory.group_id,
                    status="failed",
                )
            ]
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="failed",
        )

    async def error_trajectories(
        self,
        *,
        session_id: str,
        trajectory_ids: list[str],
    ) -> EndTrajectoriesResponse:
        if not trajectory_ids:
            return EndTrajectoriesResponse(
                session_id=session_id,
                trajectory_ids=[],
                status="failed",
            )

        trajectories_by_id = await self._require_active_trajectories(
            session_id=session_id,
            trajectory_ids=trajectory_ids,
        )
        trajectories_to_update: list[Trajectory] = []
        for trajectory_id in trajectory_ids:
            trajectory = trajectories_by_id[trajectory_id]
            trajectories_to_update.append(
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    group_id=trajectory.group_id,
                    status="failed",
                )
            )
        await self._finish_trajectories(trajectories_to_update)
        return EndTrajectoriesResponse(
            session_id=session_id,
            trajectory_ids=[
                trajectory.trajectory_id for trajectory in trajectories_to_update
            ],
            status="failed",
        )

    async def discard_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
    ) -> EndTrajectoryResponse:
        trajectory = await self._require_active_trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
        )
        await self._finish_trajectories(
            [
                Trajectory(
                    trajectory_id=trajectory.trajectory_id,
                    session_id=trajectory.session_id,
                    group_id=trajectory.group_id,
                    status="discarded",
                )
            ]
        )
        return EndTrajectoryResponse(
            session_id=session_id,
            trajectory_id=trajectory_id,
            status="discarded",
        )

    async def export_checkpoint(
        self,
        *,
        session_id: str,
    ) -> ExportCheckpointResponse:
        await self._require_active_session(session_id)
        await self._drain_finished_trajectories()
        if self._active_trajectories:
            raise Exception(
                "all trajectories must be ended before exporting a checkpoint"
            )
        assert self._train_loop is not None, "train loop must exist for active session"
        policy_version = self._train_loop.policy_version
        checkpoint_path = await asyncio.to_thread(
            self.runtime.train_manager().export_checkpoint,
            policy_version=policy_version,
        )
        return ExportCheckpointResponse(
            session_id=session_id,
            policy_version=policy_version,
            checkpoint_path=checkpoint_path,
        )

    async def end_session(
        self,
        *,
        session_id: str,
    ) -> EndSessionResponse:
        await self._require_active_session(session_id)
        await self._drain_finished_trajectories()
        if any(
            trajectory.session_id == session_id
            for trajectory in self._active_trajectories.values()
        ):
            raise Exception(
                "all trajectories must be ended before ending the session"
            )

        self._active_session_id = None
        self._active_session_model_name = None
        self._active_trajectories = {}
        self._active_turns = {}
        if self._train_loop is not None:
            await self._train_loop.stop()
            while await self._train_loop.train_once():
                pass
            self._train_loop = None
        await asyncio.to_thread(self.runtime.shutdown)

        return EndSessionResponse(session_id=session_id, status="completed")

    async def shutdown(self) -> None:
        await self._drain_finished_trajectories()
        if self._train_loop is not None:
            await self._train_loop.stop()
            self._train_loop = None
        self._active_session_id = None
        self._active_session_model_name = None
        self._active_trajectories = {}
        self._active_turns = {}
        await asyncio.to_thread(self.runtime.shutdown)

    def _ensure_generate_workers_locked(self) -> None:
        while (
            self._pending_generates
            and self._generate_workers < self._generate_parallelism
        ):
            self._generate_workers += 1
            asyncio.create_task(self._flush_generate_queue())

    def _ensure_finish_task_locked(self) -> None:
        task = self._finish_task
        if task is not None and not task.done():
            return
        self._finish_task = asyncio.create_task(self._flush_finish_queue())

    async def _flush_generate_queue(self) -> None:
        await asyncio.sleep(self.GENERATE_BATCH_MAX_WAIT_SECONDS)
        while True:
            async with self._generate_lock:
                if not self._pending_generates:
                    self._generate_workers -= 1
                    return
                batch = self._take_generate_batch_locked()
            try:
                outputs = await self._execute_generate_batch(
                    session_id=batch[0].session_id,
                    trajectory_ids=[request.trajectory_id for request in batch],
                    group_ids=[request.group_id for request in batch],
                    messages_per_item=[request.messages for request in batch],
                    sampling_params=batch[0].sampling_params,
                )
            except Exception as exc:
                logger.exception(
                    "gateway batched generate failed for %s trajectory(s)",
                    len(batch),
                )
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                for request, output in zip(batch, outputs, strict=True):
                    if not request.future.done():
                        request.future.set_result(output)

    async def _flush_finish_queue(self) -> None:
        await asyncio.sleep(self.GENERATE_BATCH_MAX_WAIT_SECONDS)
        while True:
            async with self._finish_lock:
                if not self._pending_finishes:
                    self._finish_task = None
                    return
                batch = self._pending_finishes
                self._pending_finishes = []
            await self.store.create_trajectories(
                [item.trajectory for item in batch]
            )
            await self.store.append_turns(
                [turn for item in batch for turn in item.turns]
            )

    def _take_generate_batch_locked(self) -> list[_PendingGenerate]:
        assert self._pending_generates
        batch_key = self._pending_generates[0].batch_key
        batch: list[_PendingGenerate] = []
        remaining: list[_PendingGenerate] = []
        for request in self._pending_generates:
            if (
                request.batch_key == batch_key
                and len(batch) < self.GENERATE_BATCH_MAX_SIZE
            ):
                batch.append(request)
                continue
            remaining.append(request)
        self._pending_generates = remaining
        return batch

    async def _execute_generate_batch(
        self,
        *,
        session_id: str,
        trajectory_ids: list[str],
        group_ids: list[str | None],
        messages_per_item: list[list[ChatMessage]],
        sampling_params: dict[str, object],
    ) -> list[GenerateResponse]:
        session = await self._require_active_session(session_id)
        for trajectory_id, group_id in zip(trajectory_ids, group_ids, strict=True):
            trajectory = self._active_trajectories.get(trajectory_id)
            if trajectory is None:
                self._active_trajectories[trajectory_id] = Trajectory(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    group_id=group_id,
                    status="active",
                )
                self._active_turns[trajectory_id] = []
                continue
            if trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            if trajectory.status != "active":
                raise Exception(f"trajectory {trajectory_id} is not active")
        turn_indexes = [
            len(self._active_turns[trajectory_id]) for trajectory_id in trajectory_ids
        ]
        input_ids_per_item = await asyncio.to_thread(
            self._tokenize_messages_batch_deduped,
            messages_per_item,
        )

        generations = await asyncio.to_thread(
            self.runtime.generate_batch,
            input_ids=input_ids_per_item,
            sampling_params=sampling_params,
        )

        turns_to_append: list[Turn] = []
        outputs: list[GenerateResponse] = []
        for trajectory_id, input_ids, turn_index, generation in zip(
            trajectory_ids,
            input_ids_per_item,
            turn_indexes,
            generations,
            strict=True,
        ):
            turns_to_append.append(
                self._build_turn(
                    trajectory_id=trajectory_id,
                    turn_index=turn_index,
                    input_ids=input_ids,
                    generation=generation,
                )
            )
            outputs.append(
                self._build_generate_response(
                    session_id=session_id,
                    session_model_name=session.model_name,
                    trajectory_id=trajectory_id,
                    turn_index=turn_index,
                    input_ids=input_ids,
                    generation=generation,
                )
            )
        for trajectory_id, turn in zip(trajectory_ids, turns_to_append, strict=True):
            self._active_turns[trajectory_id].append(turn)
        return outputs

    def _tokenize_messages_batch_deduped(
        self,
        message_batches: list[list[ChatMessage]],
    ) -> list[list[int]]:
        if not message_batches:
            return []
        unique_batches: list[list[ChatMessage]] = []
        unique_keys: dict[str, int] = {}
        key_indexes: list[int] = []
        for messages in message_batches:
            key = self._messages_key(messages)
            index = unique_keys.get(key)
            if index is None:
                index = len(unique_batches)
                unique_keys[key] = index
                unique_batches.append(messages)
            key_indexes.append(index)
        unique_input_ids = self.runtime.tokenize_messages_batch(unique_batches)
        return [list(unique_input_ids[index]) for index in key_indexes]

    @staticmethod
    def _messages_key(messages: list[ChatMessage]) -> str:
        return json.dumps(
            [message.model_dump(mode="json") for message in messages],
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _sampling_params_key(sampling_params: dict[str, object]) -> str:
        return json.dumps(
            sampling_params,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    async def _require_active_session(self, session_id: str) -> Session:
        active_session_id = self._active_session_id
        active_model_name = self._active_session_model_name
        if session_id == active_session_id and active_model_name is not None:
            return Session(session_id=session_id, model_name=active_model_name)
        session = await self.store.get_session(session_id)
        if session is None:
            raise Exception(f"unknown session_id: {session_id}")
        if session_id != active_session_id:
            raise Exception(f"session {session_id} is not active")
        self._active_session_model_name = session.model_name
        return session

    async def _require_active_trajectory(
        self,
        *,
        session_id: str,
        trajectory_id: str,
    ) -> Trajectory:
        trajectory = self._active_trajectories.get(trajectory_id)
        if trajectory is not None:
            if trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            return trajectory
        trajectory = await self.store.get_trajectory(trajectory_id)
        if trajectory is None or trajectory.session_id != session_id:
            raise Exception(f"unknown trajectory_id: {trajectory_id}")
        raise Exception(f"trajectory {trajectory_id} is not active")

    async def _require_active_trajectories(
        self,
        *,
        session_id: str,
        trajectory_ids: list[str],
    ) -> dict[str, Trajectory]:
        missing = [
            trajectory_id
            for trajectory_id in trajectory_ids
            if trajectory_id not in self._active_trajectories
        ]
        stored = {
            trajectory.trajectory_id: trajectory
            for trajectory in await self.store.get_trajectories(missing)
        }
        for trajectory_id in trajectory_ids:
            trajectory = self._active_trajectories.get(trajectory_id)
            if trajectory is None:
                stored_trajectory = stored.get(trajectory_id)
                if (
                    stored_trajectory is None
                    or stored_trajectory.session_id != session_id
                ):
                    raise Exception(f"unknown trajectory_id: {trajectory_id}")
                raise Exception(f"trajectory {trajectory_id} is not active")
            if trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            if trajectory.status != "active":
                raise Exception(f"trajectory {trajectory_id} is not active")
        return {
            trajectory_id: self._active_trajectories[trajectory_id]
            for trajectory_id in trajectory_ids
        }

    async def _finish_trajectories(self, trajectories: list[Trajectory]) -> None:
        pending: list[_PendingFinish] = []
        for trajectory in trajectories:
            pending.append(
                _PendingFinish(
                    trajectory=trajectory,
                    turns=self._active_turns.pop(trajectory.trajectory_id),
                )
            )
            del self._active_trajectories[trajectory.trajectory_id]
        async with self._finish_lock:
            self._pending_finishes.extend(pending)
            self._ensure_finish_task_locked()

    async def _drain_finished_trajectories(self) -> None:
        while True:
            async with self._finish_lock:
                task = self._finish_task
                pending = bool(self._pending_finishes)
            if task is None:
                if not pending:
                    return
                async with self._finish_lock:
                    self._ensure_finish_task_locked()
                    task = self._finish_task
            assert task is not None
            await task

    @staticmethod
    def _build_generate_response(
        *,
        session_id: str,
        session_model_name: str,
        trajectory_id: str,
        turn_index: int,
        input_ids: list[int],
        generation: Generation,
    ) -> GenerateResponse:
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
            model=session_model_name,
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
