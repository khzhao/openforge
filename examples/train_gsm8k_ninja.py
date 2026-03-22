# Copyright 2026 openforge

from __future__ import annotations

import json

from gsm8k_common import (
    parse_train_args,
    prepare_train_setup,
    response_text,
    run_train,
    save_summary,
)

import openforge.ninja as ninja
from openforge.benchmarks.gsm8k import compute_gsm8k_score


def main() -> int:
    """Run the shared GSM8K Ninja training example."""
    args = parse_train_args()
    setup = prepare_train_setup(args)
    sampling_params = setup["sampling_params"]

    def report_progress(update: dict[str, object]) -> None:
        print(
            "TRAIN_UPDATE",
            json.dumps(update, sort_keys=True),
            flush=True,
        )

    train_kwargs = {
        "gateway_target": setup["gateway_target"],
        "runtime_config": setup["runtime_config"],
        "inputs": setup["inputs"],
        "group_size": args.group_size,
        "epochs": args.total_epochs,
        "seed": args.seed,
        "retries": args.train_group_retries,
        "wait_timeout": args.wait_timeout,
        "max_updates": args.max_updates,
        "progress_callback": report_progress,
    }
    if args.train_group_parallelism is not None:
        train_kwargs["parallelism"] = args.train_group_parallelism

    @ninja.agent()
    def user_agent(*, prompt: str, ground_truth: str) -> float:
        response = ninja.generate(prompt, sampling_params=sampling_params)
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
        **run_train(user_agent, **train_kwargs),
    }
    save_summary(setup["summary_path"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
