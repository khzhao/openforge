# Copyright 2026 openforge

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    """Configuration for the Gateway server."""

    host: str = "0.0.0.0"
    port: int = 8000
    backend_url: str = "http://localhost:8080/v1"
