# Copyright 2026 openforge

from __future__ import annotations

import sys

import openforge.gateway.main as gateway_main
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


def test_gateway_main_applies_cli_overrides(monkeypatch) -> None:
    cfg = _server_config()
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        gateway_main.GatewayServerConfig,
        "from_yaml",
        classmethod(lambda cls, path: cfg),
    )
    monkeypatch.setattr(gateway_main, "create_app", lambda config: "app")
    monkeypatch.setattr(
        gateway_main.uvicorn,
        "run",
        lambda app, host, port: seen.update(
            {"app": app, "host": host, "port": port}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gateway",
            "--config",
            "examples/gateway_server.yaml",
            "--data-path",
            "/tmp/openforge-cli.db",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--gpus-per-node",
            "4",
            "--cpus-per-node",
            "32",
        ],
    )

    gateway_main.main()

    assert cfg.data.path == "/tmp/openforge-cli.db"
    assert cfg.gateway.host == "0.0.0.0"
    assert cfg.gateway.port == 9001
    assert cfg.cluster.gpus_per_node == 4
    assert cfg.cluster.cpus_per_node == 32
    assert seen == {"app": "app", "host": "0.0.0.0", "port": 9001}
