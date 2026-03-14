# Copyright 2026 openforge

from loguru import logger

from openforge.rollout.types import EngineAddr, EngineSpec
from openforge.utils.networking import get_free_port
from openforge.utils.ray import get_current_ray_node_ip_address

__all__ = ["Engine"]


class Engine:
    """A thin wrapper around the SGLang runtime."""

    def initialize(self, spec: EngineSpec) -> None:
        self.spec = spec

    def get_ip_addr(self) -> str:
        return get_current_ray_node_ip_address()

    def get_free_port(self, start: int, block_size: int) -> int:
        return get_free_port(start=start, block_size=block_size)

    def get_post_launch_addr(self) -> EngineAddr:
        return self.addr

    def get_post_init_spec(self) -> EngineSpec:
        return self.spec

    def launch(self, addr: EngineAddr) -> None:
        self.addr = addr
        logger.info(f"Launching SGLang server on {addr}")
