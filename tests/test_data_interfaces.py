# Copyright 2026 openforge

from __future__ import annotations

from openforge.data import OpenForgeStore


def test_openforge_store_exposes_minimal_repository_methods() -> None:
    expected_methods = {
        "create_session",
        "get_session",
        "create_trajectory",
        "get_trajectory",
        "list_trajectories",
        "update_trajectory",
        "list_completed_trajectories",
        "append_turn",
        "list_turns",
    }

    assert expected_methods.issubset(OpenForgeStore.__dict__)


def test_openforge_store_does_not_expose_workflow_methods() -> None:
    unexpected_methods = {
        "record_generation",
        "end_session",
        "mark_trained",
        "mark_failed",
    }

    assert unexpected_methods.isdisjoint(OpenForgeStore.__dict__)
