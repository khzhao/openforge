# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train_gsm8k_ninja_rl import (
    build_gateway_config,
    build_runtime_config,
    get_free_port,
    require_visible_gpus,
)
from openforge.benchmarks.gsm8k import build_gsm8k_prompt
from openforge.ninja import register


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="/home/guo/kzhao/github/openforge/artifacts/ninja-register-generate-smoke",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cpus-per-node", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    visible_gpus = require_visible_gpus(4)
    gateway_config = build_gateway_config(
        artifact_dir=artifact_dir,
        gateway_host="127.0.0.1",
        gateway_port=get_free_port("127.0.0.1"),
        visible_gpus=visible_gpus,
        cpus_per_node=args.cpus_per_node,
    )
    runtime_config = build_runtime_config(
        model_path=args.model_path,
        train_gpus=1,
        rollout_replicas=3,
        gpus_per_replica=1,
        checkpoint_root=artifact_dir / "checkpoints",
        train_batch_size=256,
        group_size=5,
        ppo_mini_batch_size_prompts=64,
        ppo_micro_batch_size_per_gpu=4,
        ppo_epochs=4,
        max_new_tokens=args.max_new_tokens,
        learning_rate=1.0e-6,
        kl_coef=1.0e-3,
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

    @register(gateway_config, runtime_config)
    def agent(client, prompt_text: str) -> float:
        client.generate(
            [{"role": "user", "content": prompt_text}],
            sampling_params=sampling_params,
        )
        return 0.0

    prompts = [prompt for _ in range(args.episodes)]
    def run() -> float:
        started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            list(executor.map(agent, prompts))
        return time.perf_counter() - started_at

    elapsed_seconds = float(agent.run(run))

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
