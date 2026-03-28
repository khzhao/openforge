# Copyright 2026 openforge

from __future__ import annotations

import json

import openforge.ninja as ninja
from examples.gsm8k.common import (
    parse_train_args,
    prepare_train_setup,
    response_text,
)
from examples.gsm8k.task import compute_gsm8k_score
from examples.shared import (
    plan_train_batches,
    print_train_update,
    print_validation_update,
    save_summary,
)


def main() -> int:
    """Run the shared GSM8K Ninja training example."""
    args = parse_train_args()
    if args.validation_every_updates < 0:
        raise ValueError("validation-every-updates must be >= 0")
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
            validation_path is not None
            and args.validation_every_updates > 0
            and update_index % args.validation_every_updates == 0
        ):
            validation_update = ninja.validate(
                user_agent,
                file_path=str(validation_path),
                wait_timeout=args.wait_timeout,
            )
            validation_event = {
                **validation_update,
                "update_index": update_index,
            }
            validation_updates.append(validation_event)
            print_validation_update(validation_event)

    summary = {
        **setup["summary"],
        "completed_updates": len(train_updates),
        "expected_updates": train_plan["expected_updates"],
        "final_checkpoint": user_agent.save(),
        "final_policy_version": user_agent.policy_version(),
        "last_train_update": train_updates[-1] if train_updates else None,
        "last_validation_update": (
            validation_updates[-1] if validation_updates else None
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
