# Copyright 2026 openforge
# ruff: noqa: D103

from __future__ import annotations

import sys
from pathlib import Path

from _script_test_utils import run_tests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.search_r1 import train_ninja as search_r1


def _reset_search_state() -> None:
    search_r1._SEARCH_CACHE.clear()
    search_r1._SEARCH_NEXT_REQUEST_AT = 0.0


def test_search_wikipedia_sends_user_agent_and_formats_hits() -> None:
    _reset_search_state()
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "query": {
                    "search": [
                        {
                            "title": "Paris",
                            "snippet": "<span>Capital</span> of France",
                        },
                        {
                            "title": "France",
                            "snippet": "Country in <b>Europe</b>",
                        },
                    ]
                }
            }

    def fake_get(url: str, *, headers, params, timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    original_get = search_r1.httpx.get
    search_r1.httpx.get = fake_get
    try:
        result = search_r1._search_wikipedia(
            query="capital of France",
            top_k=2,
            timeout=30.0,
        )
    finally:
        search_r1.httpx.get = original_get

    assert captured["url"] == "https://en.wikipedia.org/w/api.php"
    assert captured["headers"] == search_r1._SEARCH_HEADERS
    assert captured["params"] == {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": "capital of France",
        "srlimit": 2,
        "utf8": 1,
    }
    assert captured["timeout"] == 30.0
    assert result == "1. Paris: Capital of France\n2. France: Country in Europe"


def test_search_wikipedia_retries_retryable_status_codes() -> None:
    _reset_search_state()
    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code == 200:
                return None
            request = search_r1.httpx.Request("GET", "https://example.com")
            response = search_r1.httpx.Response(self.status_code, request=request)
            raise search_r1.httpx.HTTPStatusError(
                "retryable",
                request=request,
                response=response,
            )

        def json(self) -> dict[str, object]:
            return {
                "query": {
                    "search": [
                        {"title": "Paris", "snippet": "Capital of France"},
                    ]
                }
            }

    def fake_get(url: str, *, headers, params, timeout: float):
        calls["count"] += 1
        if calls["count"] < 3:
            return FakeResponse(429)
        return FakeResponse(200)

    original_get = search_r1.httpx.get
    original_sleep = search_r1.time.sleep
    search_r1.httpx.get = fake_get
    search_r1.time.sleep = lambda _seconds: None
    try:
        result = search_r1._search_wikipedia(
            query="capital of France",
            top_k=1,
            timeout=30.0,
        )
    finally:
        search_r1.httpx.get = original_get
        search_r1.time.sleep = original_sleep

    assert calls["count"] == 3
    assert result == "1. Paris: Capital of France"


def test_search_wikipedia_returns_failure_result_when_backend_stays_unavailable() -> None:
    _reset_search_state()

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            request = search_r1.httpx.Request("GET", "https://example.com")
            response = search_r1.httpx.Response(self.status_code, request=request)
            raise search_r1.httpx.HTTPStatusError(
                "retryable",
                request=request,
                response=response,
            )

    def fake_get(url: str, *, headers, params, timeout: float):
        return FakeResponse(429)

    original_get = search_r1.httpx.get
    original_sleep = search_r1.time.sleep
    search_r1.httpx.get = fake_get
    search_r1.time.sleep = lambda _seconds: None
    try:
        result = search_r1._search_wikipedia(
            query="capital of France",
            top_k=1,
            timeout=30.0,
        )
    finally:
        search_r1.httpx.get = original_get
        search_r1.time.sleep = original_sleep

    assert result == search_r1._SEARCH_FAILURE_RESULT


def test_compute_reward_matches_search_r1_shape() -> None:
    reward_model = {"ground_truth": {"target": ["Paris"]}}
    response = (
        "<think>I should search.</think>"
        "<search>capital of France</search>"
        "<information>Paris is the capital city of France.</information>"
        "<think>I have the answer.</think>"
        "<answer>Paris</answer>"
    )
    assert search_r1._compute_reward(response, reward_model) == 1.0


def test_compute_reward_gives_structure_credit_without_correct_answer() -> None:
    reward_model = {"ground_truth": {"target": ["Paris"]}}
    response = (
        "<think>I should search.</think>"
        "<search>capital of France</search>"
        "<information>Paris is the capital city of France.</information>"
        "<think>I will answer incorrectly.</think>"
        "<answer>Lyon</answer>"
    )
    assert search_r1._compute_reward(response, reward_model) == 0.3


def test_messages_from_prompt_preserves_chat_message_lists() -> None:
    prompt = [{"role": "user", "content": "Question"}]
    assert search_r1._messages_from_prompt(prompt) == prompt


def main() -> int:
    return run_tests(
        [
            test_search_wikipedia_sends_user_agent_and_formats_hits,
            test_search_wikipedia_retries_retryable_status_codes,
            test_search_wikipedia_returns_failure_result_when_backend_stays_unavailable,
            test_compute_reward_matches_search_r1_shape,
            test_compute_reward_gives_structure_credit_without_correct_answer,
            test_messages_from_prompt_preserves_chat_message_lists,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
