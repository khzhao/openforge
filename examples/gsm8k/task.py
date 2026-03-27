# Copyright 2026 openforge

from __future__ import annotations

import re

__all__ = [
    "GSM8K_FINAL_ANSWER_INSTRUCTION",
    "build_gsm8k_prompt",
    "compute_gsm8k_score",
    "extract_gsm8k_ground_truth",
    "extract_gsm8k_solution",
]

GSM8K_FINAL_ANSWER_INSTRUCTION = (
    'Let\'s think step by step and output the final answer after "####".'
)
_SOLUTION_CLIP_CHARS = 300


def build_gsm8k_prompt(question: str) -> str:
    """Append the verl-style final-answer formatting instruction."""
    return f"{question.rstrip()} {GSM8K_FINAL_ANSWER_INSTRUCTION}"


def extract_gsm8k_ground_truth(answer: str) -> str:
    """Extract the official GSM8k target answer from one dataset solution."""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    if solution is None:
        raise ValueError("GSM8k answer is missing a final `#### <number>` target")
    return solution.group(1).replace(",", "").replace("$", "")


def extract_gsm8k_solution(
    solution_str: str,
    *,
    method: str = "strict",
) -> str | None:
    """Extract a predicted final answer using verl's GSM8k rules."""
    if method not in {"strict", "flexible"}:
        raise ValueError("method must be either 'strict' or 'flexible'")

    clipped = solution_str[-_SOLUTION_CLIP_CHARS:]
    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", clipped)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")

    answers = re.findall(r"(\-?[0-9\.\,]+)", clipped)
    if not answers:
        return None
    for candidate in reversed(answers):
        if candidate not in {"", "."}:
            return candidate.replace(",", "").replace("$", "")
    return None


def compute_gsm8k_score(
    solution_str: str,
    ground_truth: str,
    *,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Compute GSM8k reward using the verl strict/flexible extractor."""
    answer = extract_gsm8k_solution(solution_str, method=method)
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return float(score)
    return float(format_score)
