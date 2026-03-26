# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

from _script_test_utils import run_tests


def _install_service_import_stubs() -> None:
    runtime = ModuleType("openforge.gateway.runtime")

    @dataclass(slots=True)
    class Generation:
        text: str
        token_ids: list[int]
        rollout_model_version: int
        finish_reason: str = "stop"

    class Runtime:
        pass

    runtime.Generation = Generation
    runtime.Runtime = Runtime
    sys.modules["openforge.gateway.runtime"] = runtime

    train_loop = ModuleType("openforge.train.loop")

    class TrainLoop:
        pass

    train_loop.TrainLoop = TrainLoop
    sys.modules["openforge.train.loop"] = train_loop


_install_service_import_stubs()

from openforge.gateway.service import Service
from openforge.gateway.runtime import Generation


def test_build_generate_response_parses_complete_tool_call() -> None:
    response = Service._build_generate_response(
        session_id="sess_1",
        session_model_name="model-a",
        trajectory_id="traj_1",
        turn_index=0,
        input_ids=[1, 2, 3],
        generation=Generation(
            text=(
                "thinking\n"
                "<tool_call>"
                '{"name":"search","arguments":{"query":"2+2"}}'
                "</tool_call>"
            ),
            token_ids=[10, 11],
            rollout_model_version=1,
            finish_reason="stop",
        ),
    )
    assert response.model_dump(mode="json", exclude_none=True)["choices"] == [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [
                    {
                        "id": "call_traj_1_0_0",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "2+2"}',
                        },
                    }
                ],
            },
        }
    ]


def test_build_generate_response_keeps_truncated_tool_call_as_text() -> None:
    response = Service._build_generate_response(
        session_id="sess_1",
        session_model_name="model-a",
        trajectory_id="traj_1",
        turn_index=0,
        input_ids=[1, 2, 3],
        generation=Generation(
            text='before <tool_call>{"name":"search","arguments":{"query":"2+2"}}',
            token_ids=[10, 11],
            rollout_model_version=1,
            finish_reason="length",
        ),
    )
    assert response.model_dump(mode="json", exclude_none=True)["choices"] == [
        {
            "finish_reason": "length",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    'before <tool_call>{"name":"search","arguments":{"query":"2+2"}}'
                ),
            },
        }
    ]


def main() -> int:
    return run_tests(
        [
            test_build_generate_response_parses_complete_tool_call,
            test_build_generate_response_keeps_truncated_tool_call_as_text,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
