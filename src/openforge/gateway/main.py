# Copyright 2026 openforge

import argparse

import uvicorn

from openforge.configs.models import GatewayServerConfig
from openforge.gateway.server import create_app


def main() -> None:
    """Run the gateway API server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Gateway server YAML config file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override gateway.host from the config file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override gateway.port from the config file.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Override cluster.gpus_per_node from the config file.",
    )
    parser.add_argument(
        "--cpus-per-node",
        type=int,
        default=None,
        help="Override cluster.cpus_per_node from the config file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data.path from the config file.",
    )
    args = parser.parse_args()

    cfg = GatewayServerConfig.from_yaml(args.config)
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.host is not None:
        cfg.gateway.host = args.host
    if args.port is not None:
        cfg.gateway.port = args.port
    if args.gpus_per_node is not None:
        cfg.cluster.gpus_per_node = args.gpus_per_node
    if args.cpus_per_node is not None:
        cfg.cluster.cpus_per_node = args.cpus_per_node
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.gateway.host, port=cfg.gateway.port)


if __name__ == "__main__":
    main()
