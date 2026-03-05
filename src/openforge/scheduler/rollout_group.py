# Copyright 2026 openforge

from openforge.configs import OpenForgeConfig


class RolloutGroup:
    """Ray actor group for rollout workers."""

    def __init__(self, cfg: OpenForgeConfig) -> None:
        self.cfg = cfg
