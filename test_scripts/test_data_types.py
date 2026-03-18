# Copyright 2026 openforge

from __future__ import annotations

import pytest

from openforge.data import Session, Trajectory, Turn


def test_session_and_trajectory_construct() -> None:
    session = Session(session_id="session-0", model_name="model-a")
    trajectory = Trajectory(
        trajectory_id="traj-0",
        session_id=session.session_id,
        parent_trajectory_id=None,
        status="active",
    )

    assert session.session_id == "session-0"
    assert trajectory.is_active is True
    assert trajectory.is_terminal is False


def test_completed_trajectory_can_store_final_reward() -> None:
    trajectory = Trajectory(
        trajectory_id="traj-0",
        session_id="session-0",
        parent_trajectory_id="traj-parent",
        status="completed",
        final_reward=1.5,
    )

    assert trajectory.final_reward == 1.5
    assert trajectory.is_terminal is True


def test_active_trajectory_cannot_store_final_reward() -> None:
    with pytest.raises(ValueError, match="final_reward"):
        Trajectory(
            trajectory_id="traj-0",
            session_id="session-0",
            parent_trajectory_id=None,
            status="active",
            final_reward=1.0,
        )


@pytest.mark.parametrize("status", ["forked", "completed", "trained", "failed"])
def test_terminal_trajectory_statuses_are_terminal(status: str) -> None:
    trajectory = Trajectory(
        trajectory_id="traj-0",
        session_id="session-0",
        parent_trajectory_id="traj-parent",
        status=status,
        final_reward=1.0 if status != "forked" else None,
    )

    assert trajectory.is_active is False
    assert trajectory.is_terminal is True


def test_turn_exposes_prompt_and_completion_token_views() -> None:
    turn = Turn(
        trajectory_id="traj-0",
        turn_index=2,
        rollout_model_version=3,
        prompt_length=3,
        input_ids=[10, 11, 12, 20, 21],
        position_ids=[0, 1, 2, 3, 4],
        loss_mask=[False, False, True, True],
        old_logprobs=[-1.0, -1.1, -0.1, -0.2],
    )

    assert turn.prompt_token_ids == [10, 11, 12]
    assert turn.completion_token_ids == [20, 21]


def test_turn_validates_lengths() -> None:
    with pytest.raises(ValueError, match="position_ids"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version=0,
            prompt_length=1,
            input_ids=[1, 2],
            position_ids=[0],
            loss_mask=[True],
            old_logprobs=[-0.1],
        )


def test_turn_rejects_negative_turn_index() -> None:
    with pytest.raises(ValueError, match="turn_index"):
        Turn(
            trajectory_id="traj-0",
            turn_index=-1,
            rollout_model_version=0,
            prompt_length=1,
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[True],
            old_logprobs=[-0.1],
        )


def test_turn_rejects_invalid_prompt_length() -> None:
    with pytest.raises(ValueError, match="prompt_length"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version=0,
            prompt_length=3,
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[True],
            old_logprobs=[-0.1],
        )


def test_turn_rejects_invalid_loss_mask_length() -> None:
    with pytest.raises(ValueError, match="loss_mask"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version=0,
            prompt_length=1,
            input_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            loss_mask=[True],
            old_logprobs=[-0.1, -0.2],
        )


def test_turn_rejects_invalid_old_logprobs_length() -> None:
    with pytest.raises(ValueError, match="old_logprobs"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version=0,
            prompt_length=1,
            input_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            loss_mask=[False, True],
            old_logprobs=[-0.1],
        )
