# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

import sys
from unittest.mock import patch

from _script_test_utils import install_test_stubs, run_tests

install_test_stubs()

import openforge.gateway.main as gateway_main
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig


def _server_config() -> GatewayServerConfig:
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=1, cpus_per_node=1),
    )


def test_gateway_main_applies_cli_overrides() -> None:
    cfg = _server_config()
    seen: dict[str, object] = {}

    with patch.object(
        gateway_main.GatewayServerConfig,
        "from_yaml",
        classmethod(lambda cls, path: cfg),
    ):
        with patch.object(gateway_main, "create_app", lambda config: "app"):
            with patch.object(
                gateway_main.uvicorn,
                "run",
                lambda app, host, port: seen.update(
                    {"app": app, "host": host, "port": port}
                ),
            ):
                with patch.object(
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
                ):
                    gateway_main.main()

    assert cfg.data.path == "/tmp/openforge-cli.db"
    assert cfg.gateway.host == "0.0.0.0"
    assert cfg.gateway.port == 9001
    assert cfg.cluster.gpus_per_node == 4
    assert cfg.cluster.cpus_per_node == 32
    assert seen == {"app": "app", "host": "0.0.0.0", "port": 9001}


def main() -> int:
    return run_tests([test_gateway_main_applies_cli_overrides])


if __name__ == "__main__":
    raise SystemExit(main())
