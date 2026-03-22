# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import time

import openforge.ninja as ninja
from openforge.benchmarks.gsm8k import build_gsm8k_prompt
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import DataConfig, GatewayConfig, GatewayServerConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Ninja batching benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Run a simple batched generate benchmark through Ninja."""
    args = parse_args()
    gateway_config = GatewayServerConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host=args.gateway_host, port=args.gateway_port),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=1,
            cpus_per_node=1,
        ),
    )

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }
    prompt = build_gsm8k_prompt(
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April "
        "and May?"
    )

    @ninja.agent(gateway_config)
    def agent(client, prompt_text: str) -> float:
        client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_completion_tokens=sampling_params["max_new_tokens"],
        )
        return 0.0

    prompts = [prompt for _ in range(args.episodes)]

    started_at = time.perf_counter()
    agent.sample(
        requests=[{"prompt_text": prompt_text} for prompt_text in prompts],
    )
    elapsed_seconds = time.perf_counter() - started_at
    print(
        json.dumps(
            {
                "episodes": len(prompts),
                "elapsed_seconds": elapsed_seconds,
                "episodes_per_second": len(prompts) / elapsed_seconds,
            }
        )
    )


if __name__ == "__main__":
    main()
