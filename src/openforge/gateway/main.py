# Copyright 2026 openforge

import argparse

import uvicorn

from openforge.configs import OpenForgeConfig
from openforge.gateway.server import create_app


def main() -> None:
    """Run the gateway API server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to OpenForge YAML config file.",
    )
    args = parser.parse_args()

    cfg = OpenForgeConfig.from_yaml(args.config)
    app = create_app(cfg.gateway)
    uvicorn.run(app, host=cfg.gateway.host, port=cfg.gateway.port)


if __name__ == "__main__":
    main()
