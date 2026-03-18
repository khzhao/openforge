# Copyright 2026 openforge

from pydantic import BaseModel, ConfigDict


class OpenForgeBaseModel(BaseModel):
    """Base class for OpenForge config models."""

    model_config = ConfigDict(extra="forbid")
