# Copyright 2026 openforge

"""Run a real openforge.ninja smoke against an active gateway session.

This script drives generation through a registered ninja agent against an
already-running gateway that already has an active session.

It validates the user-facing ninja API, not raw gateway HTTP calls.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from test_gateway_real_e2e import (
    make_artifact_dir,
)

from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig
from openforge.ninja import register


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ninja smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def build_gateway_config(
    *,
    gateway_host: str,
    gateway_port: int,
) -> GatewayServerConfig:
    """Build the gateway config passed to ``openforge.ninja``."""
    return GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host=gateway_host, port=gateway_port),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=1,
            cpus_per_node=1,
        ),
    )


def main() -> int:
    """Run the real ninja E2E smoke."""
    args = parse_args()
    artifact_dir = make_artifact_dir(args.artifact_dir)
    summary_path = artifact_dir / "summary.json"

    gateway_config = build_gateway_config(
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
    )

    responses: list[dict[str, Any]] = []

    @register(gateway_config)
    def agent(client, *, prompt: str, reward: float) -> float:
        response = client.generate(
            [{"role": "user", "content": prompt}],
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": 8,
            },
        )
        responses.append(response)
        return reward

    first_reward = agent(prompt="Say hello in four words.", reward=0.0)
    assert first_reward == 0.0
    first_response = responses[-1]
    first_version = str(first_response["metadata"]["rollout_model_version"])

    updated_response = None
    deadline = time.monotonic() + args.request_timeout
    while time.monotonic() < deadline:
        reward = agent(prompt="Say hello in four words.", reward=0.0)
        assert reward == 0.0
        candidate = responses[-1]
        version = str(candidate["metadata"]["rollout_model_version"])
        if version != first_version:
            updated_response = candidate
            break
        time.sleep(1.0)

    assert updated_response is not None, "rollout version never changed"

    summary = {
        "artifact_dir": str(artifact_dir),
        "first_response": first_response,
        "updated_response": updated_response,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
