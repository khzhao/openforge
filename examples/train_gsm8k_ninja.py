# Copyright 2026 openforge

from __future__ import annotations

import json

from gsm8k_common import (
    parse_train_args,
    prepare_train_setup,
    response_text,
    save_summary,
)

from openforge.benchmarks.gsm8k import compute_gsm8k_score
from openforge.ninja import register, run_train


def main() -> int:
    """Run the shared GSM8K Ninja training example."""
    args = parse_train_args()
    setup = prepare_train_setup(args)
    sampling_params = setup["sampling_params"]

    @register(
        gateway_config=setup["gateway_config"],
        runtime_config=setup["runtime_config"],
    )
    def user_agent(client, *, prompt: str, ground_truth: str) -> float:
        response = client.generate(
            [{"role": "user", "content": prompt}],
            sampling_params=sampling_params,
        )
        text = response_text(response)
        return float(
            compute_gsm8k_score(
                text,
                ground_truth,
                method="strict",
                format_score=0.0,
            )
        )

    summary = {
        **setup["summary"],
        **run_train(
            user_agent,
            inputs=setup["inputs"],
            group_size=args.group_size,
            epochs=args.total_epochs,
            seed=args.seed,
            parallelism=args.train_group_parallelism,
            retries=args.train_group_retries,
            wait_timeout=args.wait_timeout,
            max_updates=args.max_updates,
        ),
    }
    save_summary(setup["summary_path"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
