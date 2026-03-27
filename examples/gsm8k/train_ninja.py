# Copyright 2026 openforge

from __future__ import annotations

import json

from examples.gsm8k.common import (
    parse_train_args,
    prepare_train_setup,
    response_text,
)
from examples.gsm8k.task import compute_gsm8k_score
from examples.shared import print_train_update, run_train, save_summary

import openforge.ninja as ninja


def main() -> int:
    """Run the shared GSM8K Ninja training example."""
    args = parse_train_args()
    setup = prepare_train_setup(args)
    sampling_params = setup["sampling_params"]

    @ninja.agent()
    def user_agent(client, *, prompt: str, ground_truth: str) -> float:
        response = client.chat.completions.create(
            model=setup["runtime_config"].model.model_name_or_path,
            messages=[{"role": "user", "content": prompt}],
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            top_k=sampling_params["top_k"],
            repetition_penalty=sampling_params["repetition_penalty"],
            max_completion_tokens=sampling_params["max_new_tokens"],
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
            runtime_config=setup["runtime_config"],
            inputs=setup["inputs"],
            group_size=args.group_size,
            epochs=args.total_epochs,
            seed=args.seed,
            parallelism=args.train_group_parallelism,
            retries=args.train_group_retries,
            wait_timeout=args.wait_timeout,
            max_updates=args.max_updates,
            progress_callback=print_train_update,
        ),
    }
    save_summary(setup["summary_path"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
