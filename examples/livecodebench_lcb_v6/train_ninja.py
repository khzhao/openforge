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
from examples.shared import (
    plan_train_batches,
    print_train_update,
    print_validation_update,
    save_summary,
)

_LOG = logging.getLogger(__name__)
_VALIDATION_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
}


def main() -> int:
    """Run the LiveCodeBench Ninja training example."""
    args = parse_train_args()
    if args.validation_every_updates < 0:
        raise ValueError("validation-every-updates must be >= 0")
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

    validation_agent = None
    if setup["validation_path"] is not None:

        @ninja.agent()
        def validation_user_agent(
            client,
            *,
            prompt: str,
            reward_spec: dict[str, object],
            problem_id: str,
        ) -> float:
            response = client.chat.completions.create(
                model=setup["runtime_config"].model.model_name_or_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=_VALIDATION_SAMPLING_PARAMS["temperature"],
                top_p=_VALIDATION_SAMPLING_PARAMS["top_p"],
                top_k=_VALIDATION_SAMPLING_PARAMS["top_k"],
                repetition_penalty=sampling_params["repetition_penalty"],
                max_completion_tokens=sampling_params["max_new_tokens"],
            )
            text = str(response.choices[0].message.content or "")
            try:
                return float(compute_livecodebench_score(text, dict(reward_spec)))
            except Exception:
                _LOG.exception(
                    "livecodebench validation reward evaluation failed for problem_id=%s",
                    problem_id,
                )
                return 0.0

        validation_agent = validation_user_agent

    train_plan = plan_train_batches(
        runtime_config=setup["runtime_config"],
        inputs=setup["inputs"],
        group_size=args.group_size,
        epochs=args.total_epochs,
        seed=args.seed,
        max_updates=args.max_updates,
    )
    train_updates: list[dict[str, object]] = []
    validation_updates: list[dict[str, object]] = []
    validation_path = setup["validation_path"]
    for update_index, batch_inputs in enumerate(
        train_plan["update_inputs"],
        start=1,
    ):
        assert isinstance(batch_inputs, list)
        train_update = ninja.train(
            user_agent,
            inputs=batch_inputs,
            group_size=args.group_size,
            concurrency=args.train_group_parallelism,
            retries=args.train_group_retries,
            wait_timeout=args.wait_timeout,
        )
        train_event = {
            **train_update,
            "policy_version": train_update["final_policy_version"],
            "update_index": update_index,
        }
        train_updates.append(train_event)
        print_train_update(train_event)
        if (
            validation_agent is not None
            and validation_path is not None
            and args.validation_every_updates > 0
            and update_index % args.validation_every_updates == 0
        ):
            validation_update = ninja.validate(
                validation_agent,
                file_path=str(validation_path),
                wait_timeout=args.wait_timeout,
            )
            validation_event = {
                **validation_update,
                "update_index": update_index,
            }
            validation_updates.append(validation_event)
            print_validation_update(validation_event)

    final_checkpoint = user_agent.save()
    summary = {
        **setup["summary"],
        "completed_updates": len(train_updates),
        "expected_updates": train_plan["expected_updates"],
        "final_checkpoint": final_checkpoint,
        "final_policy_version": final_checkpoint["policy_version"],
        "last_train_update": train_updates[-1] if train_updates else None,
        "last_validation_update": (
            validation_updates[-1] if validation_updates else None
        ),
        "validation_sampling": (
            None if validation_agent is None else dict(_VALIDATION_SAMPLING_PARAMS)
        ),
        "prompt_groups_per_update": train_plan["prompt_groups_per_update"],
        "train_groups": train_plan["train_groups"],
        "train_groups_consumed": len(train_updates)
        * int(train_plan["prompt_groups_per_update"]),
        "train_groups_dropped": train_plan["train_groups_dropped"],
        "train_updates": train_updates,
        "validation_updates": validation_updates,
    }
    save_summary(setup["summary_path"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
