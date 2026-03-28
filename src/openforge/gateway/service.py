# Copyright 2026 openforge

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from uuid import uuid4

from openforge.data import OpenForgeStore, Session, Trajectory, Turn
from openforge.gateway.runtime import Generation, Runtime
from openforge.gateway.types import (
    AssistantMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    ChatCompletionChoice,
    ChatCompletionCreateRequest,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionTool,
    ChatCompletionToolChoice,
    CompletionUsage,
    EndSessionResponse,
    EndTrajectoriesResponse,
    EndTrajectoryResponse,
    ExportCheckpointResponse,
    RuntimeConfig,
    StartSessionResponse,
    StartTrajectoryGroupsResponse,
    StartTrajectoryResponse,
    TrajectoryStatusesResponse,
    TrajectoryStatusInfo,
)
from openforge.logging import SessionLogger

__all__ = ["Service"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingGenerate:
    session_id: str
    trajectory_id: str
    group_id: str | None
    messages: list[ChatCompletionMessage]
    tools: list[ChatCompletionTool] | None
    sampling_params: dict[str, object]
    future: asyncio.Future[ChatCompletionResponse]
    batch_key: tuple[str, str, str]


class Service:
    """Own the session, trajectory, and turn-recording workflow."""

    GENERATE_BATCH_MAX_SIZE = 320
    GENERATE_BATCH_MAX_SIZE_PER_REPLICA = 64
    GENERATE_BATCH_MAX_WAIT_SECONDS = 0.02
    _TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        flags=re.DOTALL,
    )

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
        self._active_turn_counts: dict[str, int] = {}
        self._generate_lock = asyncio.Lock()
        self._generate_task: asyncio.Task[None] | None = None
        self._pending_generates: list[_PendingGenerate] = []
        self._generate_batch_max_size = self.GENERATE_BATCH_MAX_SIZE
        self._runtime_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openforge-gateway-runtime",
        )
        self._session_logger = SessionLogger()

    async def current_session(self) -> StartSessionResponse | None:
        session_id = self._active_session_id
        if session_id is None:
            return None

        policy_version = self.runtime.train().policy_version
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

    async def trajectory_statuses(
        self,
        *,
        session_id: str,
        trajectory_ids: list[str],
    ) -> TrajectoryStatusesResponse:
        await self._require_active_session(session_id)
        if not trajectory_ids:
            return TrajectoryStatusesResponse(session_id=session_id, trajectories=[])

        stored = {
            trajectory.trajectory_id: trajectory
            for trajectory in await self.store.get_trajectories(trajectory_ids)
        }
        trajectories: list[TrajectoryStatusInfo] = []
        for trajectory_id in trajectory_ids:
            trajectory = self._active_trajectories.get(trajectory_id)
            if trajectory is None:
                trajectory = stored.get(trajectory_id)
            if trajectory is None or trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            trajectories.append(
                TrajectoryStatusInfo(
                    trajectory_id=trajectory_id,
                    status=trajectory.status,
                )
            )
        return TrajectoryStatusesResponse(
            session_id=session_id,
            trajectories=trajectories,
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
            await self.runtime.shutdown()
        resolved_model_name = await self._run_blocking_call(
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
        self._active_turn_counts = {}
        rollout_replicas = sum(
            int(engine_group.replicas)
            for engine_group in runtime_config.rollout.engine_groups
        )
        self._generate_batch_max_size = max(
            1,
            min(
                self.GENERATE_BATCH_MAX_SIZE,
                rollout_replicas * self.GENERATE_BATCH_MAX_SIZE_PER_REPLICA,
            ),
        )
        train_runtime = self.runtime.train()
        self._session_logger.start(
            session_id=session_id,
            runtime_config=runtime_config,
        )
        train_runtime.set_update_callback(self._session_logger.record_train_update)
        train_runtime.start_session(
            session_id=session_id,
            store=self.store,
        )
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
        trajectory = Trajectory(
            trajectory_id=self._new_id("traj"),
            session_id=session_id,
            group_id=group_id,
            status="active",
            expected_group_size=1,
        )
        await self._record_active_trajectories([trajectory])
        return StartTrajectoryResponse(
            trajectory_id=trajectory.trajectory_id,
            session_id=session_id,
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
            return StartTrajectoryGroupsResponse(
                session_id=session_id, trajectory_ids=[]
            )

        trajectory_ids_per_group: list[list[str]] = []
        trajectories_to_create: list[Trajectory] = []
        for count, group_id in zip(counts, group_ids, strict=True):
            if count <= 0:
                raise Exception("count must be >= 1")
            trajectory_ids = [self._new_id("traj") for _ in range(count)]
            for trajectory_id in trajectory_ids:
                trajectories_to_create.append(
                    Trajectory(
                        trajectory_id=trajectory_id,
                        session_id=session_id,
                        group_id=group_id,
                        status="active",
                        expected_group_size=count,
                    )
                )
            trajectory_ids_per_group.append(trajectory_ids)
        await self._record_active_trajectories(trajectories_to_create)
        return StartTrajectoryGroupsResponse(
            session_id=session_id,
            trajectory_ids=trajectory_ids_per_group,
        )

    async def generate(
        self,
        *,
        request: ChatCompletionCreateRequest,
    ) -> ChatCompletionResponse:
        session_id = request.openforge.session_id
        trajectory_id = request.openforge.trajectory_id
        group_id = request.openforge.group_id
        session = await self._require_active_session(session_id)
        if request.model != session.model_name:
            raise Exception(
                f"request model {request.model!r} does not match active session model "
                f"{session.model_name!r}"
            )

        if request.stream:
            raise Exception("stream is not supported")
        if request.n not in (None, 1):
            raise Exception("n must be 1")
        if request.frequency_penalty not in (None, 0.0):
            raise Exception("frequency_penalty is not supported")
        if request.presence_penalty not in (None, 0.0):
            raise Exception("presence_penalty is not supported")

        tools = self._resolve_tools(request.tools, request.tool_choice)
        sampling_params = self._sampling_params_from_request(request)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ChatCompletionResponse] = loop.create_future()
        pending = _PendingGenerate(
            session_id=session_id,
            trajectory_id=trajectory_id,
            group_id=group_id,
            messages=list(request.messages),
            tools=tools,
            sampling_params=sampling_params,
            future=future,
            batch_key=(
                session_id,
                self._sampling_params_key(sampling_params),
                self._tools_key(tools),
            ),
        )
        async with self._generate_lock:
            self._pending_generates.append(pending)
            self._ensure_generate_task_locked()
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
            final_rewards=[final_reward],
            trajectory_ids=[trajectory_id],
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
                if (
                    stored_trajectory is None
                    or stored_trajectory.session_id != session_id
                ):
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
                    expected_group_size=trajectory.expected_group_size,
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
                    expected_group_size=trajectory.expected_group_size,
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
                    expected_group_size=trajectory.expected_group_size,
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
                    expected_group_size=trajectory.expected_group_size,
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
        if self._active_trajectories:
            raise Exception(
                "all trajectories must be ended before exporting a checkpoint"
            )
        policy_version, checkpoint_path = await self._run_blocking_call(
            self.runtime.train().export_checkpoint,
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
        if any(
            trajectory.session_id == session_id
            for trajectory in self._active_trajectories.values()
        ):
            raise Exception("all trajectories must be ended before ending the session")

        self._active_session_id = None
        self._active_session_model_name = None
        self._active_trajectories = {}
        self._active_turn_counts = {}
        self._generate_batch_max_size = self.GENERATE_BATCH_MAX_SIZE
        self._session_logger.flush(force=True)
        self._session_logger.finish()
        await self.runtime.train().end_session()
        await self.runtime.shutdown()

        return EndSessionResponse(session_id=session_id, status="completed")

    async def shutdown(self) -> None:
        self._active_session_id = None
        self._active_session_model_name = None
        self._active_trajectories = {}
        self._active_turn_counts = {}
        self._generate_batch_max_size = self.GENERATE_BATCH_MAX_SIZE
        self._session_logger.finish()
        try:
            await self.runtime.shutdown()
        finally:
            self._runtime_executor.shutdown(wait=False, cancel_futures=True)

    async def status(self) -> dict[str, object]:
        runtime_status = {
            "train": {},
            "rollout": {},
            "cluster": {},
        }
        if self._active_session_id is not None:
            runtime_status = await self._run_blocking_call(self.runtime.status)
        train_status = runtime_status["train"]
        rollout_status = runtime_status["rollout"]
        cluster_status = runtime_status["cluster"]
        self._session_logger.record_runtime_status(
            rollout_status=rollout_status,
            cluster_status=cluster_status,
        )
        self._session_logger.flush()
        return self._session_logger.snapshot(
            train_status=train_status,
            rollout_status=rollout_status,
            cluster_status=cluster_status,
        )

    def _ensure_generate_task_locked(self) -> None:
        task = self._generate_task
        if task is not None and not task.done():
            return
        self._generate_task = asyncio.create_task(self._flush_generate_queue())

    async def _flush_generate_queue(self) -> None:
        await asyncio.sleep(self.GENERATE_BATCH_MAX_WAIT_SECONDS)
        while True:
            async with self._generate_lock:
                if not self._pending_generates:
                    self._generate_task = None
                    return
                batch = self._take_generate_batch_locked()
            try:
                outputs = await self._execute_generate_batch(
                    session_id=batch[0].session_id,
                    trajectory_ids=[request.trajectory_id for request in batch],
                    group_ids=[request.group_id for request in batch],
                    messages_per_item=[request.messages for request in batch],
                    tools=batch[0].tools,
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

    def _take_generate_batch_locked(self) -> list[_PendingGenerate]:
        assert self._pending_generates
        batch_key = self._pending_generates[0].batch_key
        batch: list[_PendingGenerate] = []
        remaining: list[_PendingGenerate] = []
        for request in self._pending_generates:
            if (
                request.batch_key == batch_key
                and len(batch) < self._generate_batch_max_size
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
        messages_per_item: list[list[ChatCompletionMessage]],
        tools: list[ChatCompletionTool] | None,
        sampling_params: dict[str, object],
    ) -> list[ChatCompletionResponse]:
        batch_started_monotonic = time.monotonic()
        session = await self._require_active_session(session_id)
        new_trajectories: list[Trajectory] = []
        missing_trajectory_ids = [
            trajectory_id
            for trajectory_id in trajectory_ids
            if trajectory_id not in self._active_trajectories
        ]
        stored_trajectories = {
            trajectory.trajectory_id: trajectory
            for trajectory in await self.store.get_trajectories(missing_trajectory_ids)
        }
        for trajectory_id, group_id in zip(trajectory_ids, group_ids, strict=True):
            trajectory = self._active_trajectories.get(trajectory_id)
            if trajectory is None:
                stored_trajectory = stored_trajectories.get(trajectory_id)
                if stored_trajectory is not None:
                    if stored_trajectory.session_id != session_id:
                        raise Exception(f"unknown trajectory_id: {trajectory_id}")
                    raise Exception(f"trajectory {trajectory_id} is not active")
                new_trajectories.append(
                    Trajectory(
                        trajectory_id=trajectory_id,
                        session_id=session_id,
                        group_id=group_id,
                        status="active",
                        expected_group_size=1,
                    )
                )
                continue
            if trajectory.session_id != session_id:
                raise Exception(f"unknown trajectory_id: {trajectory_id}")
            if trajectory.status != "active":
                raise Exception(f"trajectory {trajectory_id} is not active")
        await self._record_active_trajectories(new_trajectories)
        turn_indexes = [
            self._active_turn_counts[trajectory_id] for trajectory_id in trajectory_ids
        ]
        (
            input_ids_per_item,
            generations,
            tokenize_dedupe_hits,
        ) = await self._run_blocking_call(
            self._tokenize_and_generate_batch,
            trajectory_ids,
            messages_per_item,
            tools,
            sampling_params,
        )

        turns_to_append: list[Turn] = []
        outputs: list[ChatCompletionResponse] = []
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
        await self.store.append_turns(turns_to_append)
        for trajectory_id in trajectory_ids:
            turn_count = self._active_turn_counts.get(trajectory_id)
            if turn_count is not None:
                self._active_turn_counts[trajectory_id] = turn_count + 1
        async with self._generate_lock:
            pending_generate_count = len(self._pending_generates)
        self._session_logger.record_generations(
            input_ids_per_item=input_ids_per_item,
            generations=generations,
            latency_seconds=time.monotonic() - batch_started_monotonic,
            pending_generate_count=pending_generate_count,
            tokenize_dedupe_hits=tokenize_dedupe_hits,
        )
        self._session_logger.flush()
        return outputs

    def _tokenize_messages_batch_deduped(
        self,
        message_batches: list[list[ChatCompletionMessage]],
        tools: list[ChatCompletionTool] | None,
    ) -> tuple[list[list[int]], int]:
        if not message_batches:
            return [], 0
        unique_batches: list[list[ChatCompletionMessage]] = []
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
        unique_input_ids = self.runtime.tokenize_messages_batch(
            unique_batches,
            tools=tools,
        )
        return [list(unique_input_ids[index]) for index in key_indexes], (
            len(message_batches) - len(unique_batches)
        )

    def _tokenize_and_generate_batch(
        self,
        trajectory_ids: list[str],
        message_batches: list[list[ChatCompletionMessage]],
        tools: list[ChatCompletionTool] | None,
        sampling_params: dict[str, object],
    ) -> tuple[list[list[int]], list[Generation], int]:
        input_ids_per_item, tokenize_dedupe_hits = (
            self._tokenize_messages_batch_deduped(
                message_batches,
                tools,
            )
        )
        generations = self.runtime.generate_batch(
            trajectory_ids=trajectory_ids,
            input_ids=input_ids_per_item,
            sampling_params=sampling_params,
        )
        return input_ids_per_item, generations, tokenize_dedupe_hits

    async def _run_blocking_call(self, fn: object, /, *args: object, **kwargs: object):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._runtime_executor,
            partial(fn, *args, **kwargs),
        )

    @staticmethod
    def _messages_key(messages: list[ChatCompletionMessage]) -> str:
        return json.dumps(
            [
                message.model_dump(mode="json", exclude_none=True)
                for message in messages
            ],
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

    @staticmethod
    def _tools_key(tools: list[ChatCompletionTool] | None) -> str:
        if tools is None:
            return ""
        return json.dumps(
            [tool.model_dump(mode="json", exclude_none=True) for tool in tools],
            sort_keys=True,
            separators=(",", ":"),
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
        if not trajectories:
            return
        for trajectory in trajectories:
            await self.store.update_trajectory(trajectory)
        # This is a short control-plane release call. Keeping it inline avoids
        # executor hangs in short-lived async test runners.
        self.runtime.release_trajectories(
            [trajectory.trajectory_id for trajectory in trajectories],
        )
        for trajectory in trajectories:
            self._active_turn_counts.pop(trajectory.trajectory_id, None)
            self._active_trajectories.pop(trajectory.trajectory_id, None)

    async def _record_active_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        if not trajectories:
            return
        await self.store.create_trajectories(trajectories)
        for trajectory in trajectories:
            self._active_trajectories[trajectory.trajectory_id] = trajectory
            self._active_turn_counts[trajectory.trajectory_id] = 0

    @staticmethod
    def _build_generate_response(
        *,
        session_id: str,
        session_model_name: str,
        trajectory_id: str,
        turn_index: int,
        input_ids: list[int],
        generation: Generation,
    ) -> ChatCompletionResponse:
        content, tool_calls = Service._parse_tool_calls(
            generation.text,
            trajectory_id=trajectory_id,
            turn_index=turn_index,
        )
        finish_reason = generation.finish_reason
        if tool_calls is not None:
            finish_reason = "tool_calls"

        return ChatCompletionResponse(
            id=f"chatcmpl_{trajectory_id}_{turn_index}",
            choices=[
                ChatCompletionChoice(
                    finish_reason=finish_reason,
                    index=0,
                    message=AssistantMessage(
                        content=content,
                        tool_calls=tool_calls,
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
    def _parse_tool_calls(
        text: str,
        *,
        trajectory_id: str,
        turn_index: int,
    ) -> tuple[str | None, list[AssistantToolCall] | None]:
        matches = list(Service._TOOL_CALL_PATTERN.finditer(text))
        if not matches:
            return text, None

        tool_calls: list[AssistantToolCall] = []
        for index, match in enumerate(matches):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                return text, None
            tool_calls.append(
                AssistantToolCall(
                    id=f"call_{trajectory_id}_{turn_index}_{index}",
                    function=AssistantToolCallFunction(
                        name=str(payload["name"]),
                        arguments=json.dumps(payload["arguments"]),
                    ),
                )
            )

        content = Service._TOOL_CALL_PATTERN.sub("", text).strip()
        if not content:
            content = None
        return content, tool_calls

    @staticmethod
    def _resolve_tools(
        tools: list[ChatCompletionTool] | None,
        tool_choice: ChatCompletionToolChoice | None,
    ) -> list[ChatCompletionTool] | None:
        if tools is None:
            return None
        if tool_choice is None:
            return tools
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "required"}:
                return tools
            if tool_choice == "none":
                return None

        selected_name = tool_choice.function.name
        selected_tools = [tool for tool in tools if tool.function.name == selected_name]
        if not selected_tools:
            raise Exception(f"unknown tool_choice function: {selected_name}")
        return selected_tools

    @staticmethod
    def _sampling_params_from_request(
        request: ChatCompletionCreateRequest,
    ) -> dict[str, object]:
        sampling_params: dict[str, object] = {}
        if request.temperature is not None:
            sampling_params["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_params["top_p"] = request.top_p
        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            sampling_params["repetition_penalty"] = request.repetition_penalty

        max_tokens = request.max_completion_tokens
        if max_tokens is None:
            max_tokens = request.max_tokens
        elif request.max_tokens not in (None, max_tokens):
            raise Exception("max_tokens must match max_completion_tokens")
        if max_tokens is not None:
            sampling_params["max_new_tokens"] = max_tokens

        if request.stop is not None:
            if isinstance(request.stop, str):
                sampling_params["stop"] = [request.stop]
            else:
                sampling_params["stop"] = list(request.stop)
        if request.seed is not None:
            sampling_params["seed"] = request.seed
        return sampling_params

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
            rollout_log_probs=[0.0] * prompt_prediction_count
            + generation.rollout_log_probs,
        )
