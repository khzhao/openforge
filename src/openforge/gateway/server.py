# Copyright 2026 openforge

from __future__ import annotations

import asyncio
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
    EndSessionRequest,
    EndSessionResponse,
    EndTrajectoryRequest,
    EndTrajectoryResponse,
    GenerateRequest,
    GenerateResponse,
    ModelsResponse,
    StartSessionRequest,
    StartSessionResponse,
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
    gateway_config = config.gateway
    store = _build_store(config)
    runtime = Runtime(cfg=config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.shutdown)
            await store.close()

    service = Service(store=store, runtime=runtime)
    app = FastAPI(title="OpenForge Gateway", lifespan=lifespan)
    app.state.config = gateway_config
    app.state.runtime = runtime
    app.state.store = store
    app.state.service = service

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        return ModelsResponse.model_validate(await service.list_models())

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
                parent_trajectory_id=payload.parent_trajectory_id,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionClosedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TrajectoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TrajectoryNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            return await service.generate(
                session_id=payload.session_id,
                trajectory_id=payload.trajectory_id,
                messages=[message.model_dump() for message in payload.messages],
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
        except UnsupportedModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelBusyError as exc:
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

    return app
