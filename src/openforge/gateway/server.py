# Copyright 2026 openforge

from typing import Any

from fastapi import FastAPI, HTTPException

from openforge.configs import GatewayConfig


def create_app(config: GatewayConfig) -> FastAPI:
    """Create a minimal OpenAI-compatible gateway app."""
    app = FastAPI(title="OpenForge Gateway")
    app.state.config = config

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/models")
    async def list_models() -> list[str]:
        return ["Qwen/Qwen2.5-0.5B-Instruct"]

    @app.post("/v1/chat/completions")
    async def chat_completions(_: dict[str, Any]) -> dict[str, Any]:
        raise HTTPException(status_code=501, detail="Not implemented")

    return app
