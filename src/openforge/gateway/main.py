# Copyright 2026 openforge

import uvicorn

from openforge.configs import GatewayConfig
from openforge.gateway.server import create_app


def main() -> None:
    """Run the gateway API server."""
    config = GatewayConfig()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
