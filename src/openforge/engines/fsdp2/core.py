# Copyright 2026 openforge


from openforge.configs import OpenForgeConfig


class FSDP2Core:
    """Core FSDP2 engine that drives training loop."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        pass
