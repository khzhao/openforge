# Copyright 2026 openforge

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, TypeVar

from fastapi import FastAPI, HTTPException

from openforge import active_state
from openforge.configs.models import GatewayServerConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import Runtime
from openforge.gateway.service import Service
from openforge.gateway.types import (
    ChatCompletionCreateRequest,
    ChatCompletionResponse,
    DiscardTrajectoryRequest,
    EndSessionRequest,
    EndSessionResponse,
    EndTrajectoriesRequest,
    EndTrajectoriesResponse,
    EndTrajectoryRequest,
    EndTrajectoryResponse,
    ErrorTrajectoriesRequest,
    ErrorTrajectoryRequest,
    ExportCheckpointRequest,
    ExportCheckpointResponse,
    ModelCard,
    ModelListResponse,
    StartSessionRequest,
    StartSessionResponse,
    StartTrajectoryGroupsRequest,
    StartTrajectoryGroupsResponse,
    StartTrajectoryRequest,
    StartTrajectoryResponse,
    TrajectoryStatusesRequest,
    TrajectoryStatusesResponse,
)

__all__ = ["create_app"]

ResponseT = TypeVar("ResponseT")


def _build_store(cfg: GatewayServerConfig) -> SQLiteOpenForgeStore:
    if cfg.data.path is None:
        return SQLiteOpenForgeStore(":memory:")
    return SQLiteOpenForgeStore(Path(cfg.data.path))


async def _invoke(operation: Awaitable[ResponseT]) -> ResponseT:
    try:
        return await operation
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(
    config: GatewayServerConfig,
) -> FastAPI:
    """Create the OpenForge gateway app."""
    store = _build_store(config)
    runtime = Runtime(cfg=config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        pid = os.getpid()
        active_state.save_active_gateway(
            host=config.gateway.host,
            port=config.gateway.port,
            pid=pid,
        )
        try:
            yield
        finally:
            await service.shutdown()
            await store.close()
            active_state.clear_active_gateway(expected_pid=pid)

    service = Service(store=store, runtime=runtime)
    app = FastAPI(title="OpenForge Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/info", response_model=GatewayServerConfig)
    async def info() -> GatewayServerConfig:
        return config

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[ModelCard(id=model_id) for model_id in runtime.list_models()]
        )

    @app.get("/current_session", response_model=StartSessionResponse)
    async def current_session() -> StartSessionResponse:
        session = await service.current_session()
        if session is None:
            raise HTTPException(status_code=404, detail="no active session")
        return session

    @app.get("/status")
    async def status() -> dict[str, object]:
        return await _invoke(service.status())

    @app.post("/trajectory_statuses", response_model=TrajectoryStatusesResponse)
    async def trajectory_statuses(
        payload: TrajectoryStatusesRequest,
    ) -> TrajectoryStatusesResponse:
        return await _invoke(
            service.trajectory_statuses(
                session_id=payload.session_id,
                trajectory_ids=payload.trajectory_ids,
            )
        )

    @app.post("/start_session", response_model=StartSessionResponse)
    async def start_session(payload: StartSessionRequest) -> StartSessionResponse:
        response = await _invoke(
            service.start_session(
                runtime_config=payload.runtime,
            )
        )
        active_state.save_active_session(
            session_id=response.session_id,
            runtime=payload.runtime,
        )
        return response

    @app.post("/start_trajectory", response_model=StartTrajectoryResponse)
    async def start_trajectory(
        payload: StartTrajectoryRequest,
    ) -> StartTrajectoryResponse:
        return await _invoke(
            service.start_trajectory(
                session_id=payload.session_id,
                group_id=payload.group_id,
            )
        )

    @app.post("/start_trajectory_groups", response_model=StartTrajectoryGroupsResponse)
    async def start_trajectory_groups(
        payload: StartTrajectoryGroupsRequest,
    ) -> StartTrajectoryGroupsResponse:
        return await _invoke(
            service.start_trajectory_groups(
                session_id=payload.session_id,
                counts=payload.counts,
                group_ids=payload.group_ids,
            )
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        payload: ChatCompletionCreateRequest,
    ) -> ChatCompletionResponse:
        return await _invoke(service.generate(request=payload))

    @app.post("/end_trajectory", response_model=EndTrajectoryResponse)
    async def end_trajectory(
        payload: EndTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        return await _invoke(
            service.end_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                final_reward=payload.final_reward,
            )
        )

    @app.post("/end_trajectories", response_model=EndTrajectoriesResponse)
    async def end_trajectories(
        payload: EndTrajectoriesRequest,
    ) -> EndTrajectoriesResponse:
        return await _invoke(
            service.end_trajectories(
                session_id=payload.session_id,
                trajectory_ids=payload.trajectory_ids,
                final_rewards=payload.final_rewards,
            )
        )

    @app.post("/error_trajectory", response_model=EndTrajectoryResponse)
    async def error_trajectory(
        payload: ErrorTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        return await _invoke(
            service.error_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
            )
        )

    @app.post("/error_trajectories", response_model=EndTrajectoriesResponse)
    async def error_trajectories(
        payload: ErrorTrajectoriesRequest,
    ) -> EndTrajectoriesResponse:
        return await _invoke(
            service.error_trajectories(
                session_id=payload.session_id,
                trajectory_ids=payload.trajectory_ids,
            )
        )

    @app.post("/discard_trajectory", response_model=EndTrajectoryResponse)
    async def discard_trajectory(
        payload: DiscardTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        return await _invoke(
            service.discard_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
            )
        )

    @app.post("/end_session", response_model=EndSessionResponse)
    async def end_session(payload: EndSessionRequest) -> EndSessionResponse:
        response = await _invoke(service.end_session(session_id=payload.session_id))
        active_state.clear_active_session()
        return response

    @app.post("/export_checkpoint", response_model=ExportCheckpointResponse)
    async def export_checkpoint(
        payload: ExportCheckpointRequest,
    ) -> ExportCheckpointResponse:
        return await _invoke(service.export_checkpoint(session_id=payload.session_id))

    return app
