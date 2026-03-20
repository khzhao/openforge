# Copyright 2026 openforge

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from openforge.configs.models import GatewayServerConfig
from openforge.data import SQLiteOpenForgeStore
from openforge.gateway.runtime import ModelBusyError, Runtime, UnsupportedModelError
from openforge.gateway.service import (
    ActiveSessionError,
    ActiveTrajectoriesRemainError,
    Service,
    SessionClosedError,
    SessionNotFoundError,
    TrajectoryNotActiveError,
    TrajectoryNotFoundError,
)
from openforge.gateway.types import (
    DiscardTrajectoryRequest,
    EndTrajectoriesRequest,
    EndTrajectoriesResponse,
    ErrorTrajectoriesRequest,
    ExportCheckpointRequest,
    ExportCheckpointResponse,
    EndSessionRequest,
    EndSessionResponse,
    EndTrajectoryRequest,
    EndTrajectoryResponse,
    ErrorTrajectoryRequest,
    GenerateRequest,
    GenerateResponse,
    ModelsResponse,
    StartSessionRequest,
    StartSessionResponse,
    StartTrajectoryGroupsRequest,
    StartTrajectoryGroupsResponse,
    StartTrajectoryRequest,
    StartTrajectoryResponse,
)

__all__ = ["create_app"]


def _build_store(cfg: GatewayServerConfig) -> SQLiteOpenForgeStore:
    if cfg.data.path is None:
        return SQLiteOpenForgeStore(":memory:")
    return SQLiteOpenForgeStore(Path(cfg.data.path))


def create_app(
    config: GatewayServerConfig,
) -> FastAPI:
    """Create the OpenForge gateway app."""
    store = _build_store(config)
    runtime = Runtime(cfg=config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await service.shutdown()
            await store.close()

    service = Service(store=store, runtime=runtime)
    app = FastAPI(title="OpenForge Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        return await service.list_models()

    @app.get("/current_session", response_model=StartSessionResponse)
    async def current_session() -> StartSessionResponse:
        session = await service.current_session()
        if session is None:
            raise HTTPException(status_code=404, detail="no active session")
        return session

    @app.post("/start_session", response_model=StartSessionResponse)
    async def start_session(payload: StartSessionRequest) -> StartSessionResponse:
        try:
            return await service.start_session(
                runtime_config=payload.runtime,
            )
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ActiveSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/start_trajectory", response_model=StartTrajectoryResponse)
    async def start_trajectory(
        payload: StartTrajectoryRequest,
    ) -> StartTrajectoryResponse:
        try:
            return await service.start_trajectory(
                session_id=payload.session_id,
                group_id=payload.group_id,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/start_trajectory_groups", response_model=StartTrajectoryGroupsResponse)
    async def start_trajectory_groups(
        payload: StartTrajectoryGroupsRequest,
    ) -> StartTrajectoryGroupsResponse:
        try:
            return await service.start_trajectory_groups(
                session_id=payload.session_id,
                counts=payload.counts,
                group_ids=payload.group_ids,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            return await service.generate(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                group_id=payload.group_id,
                messages=payload.messages,
                sampling_params=payload.sampling_params,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/end_trajectory", response_model=EndTrajectoryResponse)
    async def end_trajectory(
        payload: EndTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        try:
            return await service.end_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                final_reward=payload.final_reward,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/end_trajectories", response_model=EndTrajectoriesResponse)
    async def end_trajectories(
        payload: EndTrajectoriesRequest,
    ) -> EndTrajectoriesResponse:
        try:
            return await service.end_trajectories(
                session_id=payload.session_id,
                trajectory_ids=payload.trajectory_ids,
                final_rewards=payload.final_rewards,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/error_trajectory", response_model=EndTrajectoryResponse)
    async def error_trajectory(
        payload: ErrorTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        try:
            return await service.error_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/error_trajectories", response_model=EndTrajectoriesResponse)
    async def error_trajectories(
        payload: ErrorTrajectoriesRequest,
    ) -> EndTrajectoriesResponse:
        try:
            return await service.error_trajectories(
                session_id=payload.session_id,
                trajectory_ids=payload.trajectory_ids,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/discard_trajectory", response_model=EndTrajectoryResponse)
    async def discard_trajectory(
        payload: DiscardTrajectoryRequest,
    ) -> EndTrajectoryResponse:
        try:
            return await service.discard_trajectory(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/end_session", response_model=EndSessionResponse)
    async def end_session(payload: EndSessionRequest) -> EndSessionResponse:
        try:
            return await service.end_session(session_id=payload.session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ActiveTrajectoriesRemainError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/export_checkpoint", response_model=ExportCheckpointResponse)
    async def export_checkpoint(
        payload: ExportCheckpointRequest,
    ) -> ExportCheckpointResponse:
        try:
            return await service.export_checkpoint(session_id=payload.session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ActiveTrajectoriesRemainError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
