# Copyright 2026 openforge

from __future__ import annotations

import json
import logging

import openforge.ninja as ninja
from examples.livecodebench_lcb_v6.common import (
    parse_train_args,
    prepare_train_setup,
)
from examples.livecodebench_lcb_v6.task import compute_livecodebench_score
from examples.shared import print_train_update, run_train, save_summary

_LOG = logging.getLogger(__name__)


def main() -> int:
    """Run the LiveCodeBench Ninja training example."""
    args = parse_train_args()
    if args.validation_every_updates != 0:
        raise ValueError(
            "validation-every-updates is not implemented for the LiveCodeBench "
            "LCB v6 example"
        )
    setup = prepare_train_setup(args)
    sampling_params = setup["sampling_params"]

    @ninja.agent()
    def user_agent(
        client,
        *,
        prompt: str,
        reward_spec: dict[str, object],
        problem_id: str,
    ) -> float:
        response = client.chat.completions.create(
            model=setup["runtime_config"].model.model_name_or_path,
            messages=[{"role": "user", "content": prompt}],
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            top_k=sampling_params["top_k"],
            repetition_penalty=sampling_params["repetition_penalty"],
            max_completion_tokens=sampling_params["max_new_tokens"],
        )
        text = str(response.choices[0].message.content or "")
        try:
            return float(compute_livecodebench_score(text, dict(reward_spec)))
        except Exception:
            _LOG.exception(
                "livecodebench reward evaluation failed for problem_id=%s",
                problem_id,
            )
            return 0.0

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
