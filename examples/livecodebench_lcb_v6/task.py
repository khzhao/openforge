# Copyright 2026 openforge

from __future__ import annotations

import base64
import copy
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

__all__ = [
    "build_livecodebench_prompt",
    "decode_livecodebench_private_test_cases",
    "compute_livecodebench_score",
    "evaluate_livecodebench_response",
    "extract_python_code",
    "sample_livecodebench_train_reward_spec",
]

_CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and "
    "passes all tests. The time limit is 1 second. You may start by outlining "
    "your thought process. In the end, please provide the complete code in a "
    "code block enclosed with ``` ```."
)
_FUNCTION_RUNNER = """\
from __future__ import annotations

import importlib.util
import json
import resource
import sys
import traceback

def _to_safe_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_safe_jsonable(item) for key, item in value.items()}
    return repr(value)


payload = json.load(sys.stdin)
cpu_seconds = max(1, int(float(payload.get("timeout_seconds", 5.0))))
memory_bytes = max(256, int(payload.get("memory_limit_mb", 1024))) * 1024 * 1024
file_bytes = 8 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location("submission", payload["submission_path"])
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load submission")
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    fn = getattr(module, payload["entry_point"])
    result = fn(*payload["args"], **payload["kwargs"])
    pred = _to_safe_jsonable(result)
    expected = _to_safe_jsonable(payload["expected"])
    json.dump(
        {
            "passed": pred == expected,
            "pred": pred,
        },
        sys.stdout,
    )
except BaseException:
    json.dump({"passed": False, "error": traceback.format_exc(limit=4)}, sys.stdout)
"""
_STDIO_RUNNER = """\
from __future__ import annotations

import math
import os
import resource
import runpy
import sys

cpu_seconds = max(1, int(math.ceil(float(os.environ["OPENFORGE_TIMEOUT_SECONDS"]))))
memory_bytes = max(256, int(os.environ["OPENFORGE_MEMORY_LIMIT_MB"])) * 1024 * 1024
file_bytes = 8 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
runpy.run_path(sys.argv[1], run_name="__main__")
"""


def _parse_signature(starter_code: str) -> str:
    if "def " in starter_code:
        return "def " + starter_code.split("def ", 1)[1].split("Input\n", 1)[0].strip()
    return starter_code.strip()


def build_livecodebench_prompt(
    question: str,
    *,
    starter_code: str | None = None,
    time_limit_seconds: float | None = None,
) -> str:
    """Build the wrapped LiveCodeBench prompt text."""
    del time_limit_seconds
    parts = [_CODE_PROMPT, "", question.rstrip()]
    if starter_code:
        parts.extend(
            [
                "",
                "Your solution should have the following signature: ```python",
                _parse_signature(starter_code),
                "```",
            ]
        )
    return "\n".join(parts).strip()


def _format_time_limit_sentence(time_limit_seconds: float | None) -> str:
    # Kept only for compatibility with older callers.
    if time_limit_seconds is None:
        return ""
    if float(time_limit_seconds).is_integer():
        whole_seconds = int(time_limit_seconds)
        unit = "second" if whole_seconds == 1 else "seconds"
        return f" The time limit is {whole_seconds} {unit}."
    return f" The time limit is {time_limit_seconds:g} seconds."


def extract_python_code(response_text: str) -> str:
    """Extract the longest fenced code block."""
    matches = _CODE_BLOCK_PATTERN.findall(response_text)
    if not matches:
        return ""
    return max((code for _, code in matches), key=len).strip()


def decode_livecodebench_private_test_cases(
    encoded_data: str,
    *,
    fn_name: str = "",
) -> dict[str, Any]:
    """Decode raw LiveCodeBench private test cases into a normalized bundle."""
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    tests = json.loads(original_data)
    if not tests:
        return {
            "inputs": [],
            "outputs": [],
            "testtype": "stdin",
            "fn_name": fn_name,
        }
    return {
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
        "testtype": tests[0].get("testtype", "stdin"),
        "fn_name": fn_name,
    }


def sample_livecodebench_train_reward_spec(
    reward_spec: dict[str, Any],
    *,
    keep_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample a reduced train-time test set from a full reward spec."""
    tests = list(reward_spec["tests"])
    if not tests:
        return dict(reward_spec)
    keep_count = max(1, int(len(tests) * keep_fraction))
    rng = random.Random(seed)
    keep_indexes = sorted(rng.sample(range(len(tests)), keep_count))
    reduced = copy.deepcopy(reward_spec)
    reduced["tests"] = [tests[index] for index in keep_indexes]
    return reduced


def evaluate_livecodebench_response(
    response_text: str,
    reward_spec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one code response and return SDPO-style score plus feedback."""
    code = extract_python_code(response_text)
    if not code:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": True,
            "timed_out": False,
            "truncated": False,
            "feedback": "Incorrect Format: Put your code inside a ```python ... ``` block.",
            "passed_tests": 0,
            "total_tests": 0,
        }

    task_type = str(reward_spec["task_type"])
    tests = list(reward_spec["tests"])
    if not tests:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": code,
            "incorrect_format": False,
            "timed_out": False,
            "truncated": False,
            "feedback": "No tests were provided.",
            "passed_tests": 0,
            "total_tests": 0,
        }

    timeout_seconds = float(reward_spec.get("timeout_seconds", 5.0))
    memory_limit_mb = int(reward_spec.get("memory_limit_mb", 1024))
    sparse_rewards = bool(reward_spec.get("sparse_rewards", False))
    max_feedback_failures = int(reward_spec.get("max_feedback_failures", 3))

    test_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="openforge-livecodebench-") as tmpdir:
        workdir = Path(tmpdir)
        submission_path = workdir / "submission.py"
        submission_path.write_text(code, encoding="utf-8")

        for test_case in tests:
            if task_type == "function":
                result = _run_function_test(
                    submission_path=submission_path,
                    entry_point=str(reward_spec["entry_point"]),
                    test_case=dict(test_case),
                    timeout_seconds=timeout_seconds,
                    memory_limit_mb=memory_limit_mb,
                )
            elif task_type == "stdin_stdout":
                result = _run_stdio_test(
                    submission_path=submission_path,
                    test_case=dict(test_case),
                    timeout_seconds=timeout_seconds,
                    memory_limit_mb=memory_limit_mb,
                )
            else:
                raise ValueError(f"unsupported task_type: {task_type}")
            test_results.append(result)

    passed_tests = sum(int(result["passed"]) for result in test_results)
    total_tests = len(test_results)
    acc = passed_tests / total_tests
    score = 1.0 if passed_tests == total_tests else 0.0 if sparse_rewards else acc
    failed_feedback = [
        str(result["feedback"])
        for result in test_results
        if not result["passed"] and result.get("feedback")
    ]
    truncated = len(failed_feedback) > max_feedback_failures
    if failed_feedback:
        feedback_lines = failed_feedback[:max_feedback_failures]
        if truncated:
            feedback_lines.append(
                f"... plus {len(failed_feedback) - max_feedback_failures} more failing tests."
            )
        feedback = "\n\n".join(feedback_lines)
    else:
        feedback = ""

    return {
        "score": float(score),
        "acc": float(acc),
        "pred": code,
        "incorrect_format": False,
        "timed_out": any(bool(result.get("timed_out")) for result in test_results),
        "truncated": truncated,
        "feedback": feedback,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
    }


def compute_livecodebench_score(
    response_text: str,
    reward_spec: dict[str, Any],
) -> float:
    """Return the scalar reward from a richer LiveCodeBench evaluation result."""
    return float(evaluate_livecodebench_response(response_text, reward_spec)["score"])


def _run_function_test(
    *,
    submission_path: Path,
    entry_point: str,
    test_case: dict[str, Any],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    runner_path = submission_path.with_name("judge_function.py")
    runner_path.write_text(_FUNCTION_RUNNER, encoding="utf-8")
    payload = {
        "submission_path": str(submission_path),
        "entry_point": entry_point,
        "args": list(test_case.get("args", [])),
        "kwargs": dict(test_case.get("kwargs", {})),
        "expected": test_case["expected"],
        "timeout_seconds": timeout_seconds,
        "memory_limit_mb": memory_limit_mb,
    }
    completed = _run_python_file(
        argv=[str(runner_path)],
        input_text=json.dumps(payload),
        timeout_seconds=timeout_seconds + 1.0,
        cwd=submission_path.parent,
    )
    if completed is None:
        return {
            "passed": False,
            "pred": None,
            "timed_out": True,
            "feedback": (
                "Timed out while executing function test.\n"
                f"args={payload['args']}\nkwargs={payload['kwargs']}\n"
                f"expected={payload['expected']!r}"
            ),
        }
    if completed.returncode != 0:
        return {
            "passed": False,
            "pred": None,
            "timed_out": False,
            "feedback": (
                "Judge process failed while executing function test.\n"
                f"stderr={completed.stderr.strip()}"
            ),
        }
    try:
        result = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "passed": False,
            "pred": None,
            "timed_out": False,
            "feedback": "Judge returned invalid JSON for a function test.",
        }
    if bool(result.get("passed", False)):
        return {
            "passed": True,
            "pred": result.get("pred"),
            "timed_out": False,
            "feedback": "",
        }
    error = str(result.get("error", "")).strip()
    actual = result.get("pred")
    details = [
        "Function test failed.",
        f"args={payload['args']}",
        f"kwargs={payload['kwargs']}",
        f"expected={payload['expected']!r}",
    ]
    if "pred" in result:
        details.append(f"actual={actual!r}")
    if error:
        details.append(f"error={error}")
    return {
        "passed": False,
        "pred": actual,
        "timed_out": False,
        "feedback": "\n".join(details),
    }


def _run_stdio_test(
    *,
    submission_path: Path,
    test_case: dict[str, Any],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    runner_path = submission_path.with_name("judge_stdio.py")
    runner_path.write_text(_STDIO_RUNNER, encoding="utf-8")
    completed = _run_python_file(
        argv=[str(runner_path), str(submission_path)],
        input_text=str(test_case["stdin"]),
        timeout_seconds=timeout_seconds + 1.0,
        cwd=submission_path.parent,
        env_updates={
            "OPENFORGE_TIMEOUT_SECONDS": str(timeout_seconds),
            "OPENFORGE_MEMORY_LIMIT_MB": str(memory_limit_mb),
        },
    )
    if completed is None:
        return {
            "passed": False,
            "pred": None,
            "timed_out": True,
            "feedback": (
                "Timed out while executing stdin/stdout test.\n"
                f"stdin={test_case['stdin']!r}\nexpected={test_case['stdout']!r}"
            ),
        }
    if completed.returncode != 0:
        return {
            "passed": False,
            "pred": completed.stdout,
            "timed_out": False,
            "feedback": (
                "Program crashed during stdin/stdout test.\n"
                f"stdin={test_case['stdin']!r}\n"
                f"stderr={completed.stderr.strip()}"
            ),
        }
    actual = _normalize_reference_text_output(completed.stdout)
    expected = _normalize_reference_text_output(str(test_case["stdout"]))
    if actual == expected:
        return {
            "passed": True,
            "pred": completed.stdout,
            "timed_out": False,
            "feedback": "",
        }
    return {
        "passed": False,
        "pred": completed.stdout,
        "timed_out": False,
        "feedback": (
            "stdin/stdout test failed.\n"
            f"stdin={test_case['stdin']!r}\n"
            f"actual={actual!r}\nexpected={expected!r}"
        ),
    }


def _run_python_file(
    *,
    argv: list[str],
    input_text: str,
    timeout_seconds: float,
    cwd: Path,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [sys.executable, "-I", *argv],
            input=input_text,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout_seconds,
            env=_judge_environment(env_updates),
        )
    except subprocess.TimeoutExpired:
        return None


def _judge_environment(env_updates: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    if env_updates is not None:
        env.update(env_updates)
    return env


def _normalize_text_output(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _normalize_reference_text_output(text: str) -> str:
    normalized = text.strip()
    if normalized.endswith("-"):
        normalized = normalized[: normalized.rfind("-")].rstrip()
    return normalized.replace("\r", "").replace("\n", " ").strip()
