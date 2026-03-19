# Copyright 2026 openforge

from __future__ import annotations

import pytest

from openforge.data import Session, Trajectory, Turn


def test_session_and_trajectory_construct() -> None:
    """Construct the core session and active trajectory records."""
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
    """Allow completed trajectories to store a final reward."""
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
    """Reject final rewards on active trajectories."""
    with pytest.raises(ValueError, match="final_reward"):
        Trajectory(
            trajectory_id="traj-0",
            session_id="session-0",
            parent_trajectory_id=None,
            status="active",
            final_reward=1.0,
        )


@pytest.mark.parametrize("status", ["completed", "trained", "failed"])
def test_terminal_trajectory_statuses_are_terminal(status: str) -> None:
    """Mark completed, trained, and failed trajectories as terminal."""
    trajectory = Trajectory(
        trajectory_id="traj-0",
        session_id="session-0",
        parent_trajectory_id="traj-parent",
        status=status,
        final_reward=1.0,
    )

    assert trajectory.is_active is False
    assert trajectory.is_terminal is True


def test_turn_exposes_prompt_and_completion_token_views() -> None:
    """Expose prompt and completion token slices from one stored turn."""
    turn = Turn(
        trajectory_id="traj-0",
        turn_index=2,
        rollout_model_version="v3",
        prompt_length=3,
        input_ids=[10, 11, 12, 20, 21],
        position_ids=[0, 1, 2, 3, 4],
        loss_mask=[False, False, True, True],
    )

    assert turn.prompt_token_ids == [10, 11, 12]
    assert turn.completion_token_ids == [20, 21]


def test_turn_validates_lengths() -> None:
    """Reject position arrays that do not align with input ids."""
    with pytest.raises(ValueError, match="position_ids"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version="default",
            prompt_length=1,
            input_ids=[1, 2],
            position_ids=[0],
            loss_mask=[True],
        )


def test_turn_rejects_negative_turn_index() -> None:
    """Reject negative turn indices."""
    with pytest.raises(ValueError, match="turn_index"):
        Turn(
            trajectory_id="traj-0",
            turn_index=-1,
            rollout_model_version="default",
            prompt_length=1,
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[True],
        )


def test_turn_rejects_invalid_prompt_length() -> None:
    """Reject prompt lengths outside the input id range."""
    with pytest.raises(ValueError, match="prompt_length"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version="default",
            prompt_length=3,
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[True],
        )


def test_turn_rejects_invalid_loss_mask_length() -> None:
    """Reject loss masks that do not cover every predicted token."""
    with pytest.raises(ValueError, match="loss_mask"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version="default",
            prompt_length=1,
            input_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            loss_mask=[True],
        )


def test_turn_rejects_empty_rollout_model_version() -> None:
    """Reject an empty rollout model version."""
    with pytest.raises(ValueError, match="rollout_model_version"):
        Turn(
            trajectory_id="traj-0",
            turn_index=0,
            rollout_model_version="",
            prompt_length=1,
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[True],
        )
