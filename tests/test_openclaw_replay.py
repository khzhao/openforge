# Copyright 2026 openforge

from __future__ import annotations

from types import SimpleNamespace

from openforge import active_state

from examples.openclaw.replay.gsm8k_replay import (
    _build_session_id,
    _default_max_completion_tokens,
    _default_model,
)


def test_gsm8k_replay_default_model_uses_active_runtime(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(
        model=SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    )
    monkeypatch.setattr(active_state, "load_active_runtime_config", lambda: runtime)

    assert _default_model() == "Qwen/Qwen2.5-0.5B-Instruct"


def test_gsm8k_replay_session_id_includes_epoch_and_episode() -> None:
    session_id = _build_session_id(
        mode="train",
        epoch_index=3,
        episode_index=27,
    )

    assert session_id.startswith("gsm8k-train-epoch003-episode00027-")


def test_gsm8k_replay_default_max_completion_tokens_uses_active_runtime(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(
        rollout=SimpleNamespace(
            request=SimpleNamespace(max_new_tokens=256),
        )
    )
    monkeypatch.setattr(active_state, "load_active_runtime_config", lambda: runtime)

    assert _default_max_completion_tokens() == 256
