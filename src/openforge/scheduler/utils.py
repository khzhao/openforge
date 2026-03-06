# Copyright 2026 openforge

import os

import ray

from openforge.configs import PlacementStrategy
from openforge.utils.ray import get_current_ray_node_ip_address

_RAY_PLACEMENT_STRATEGIES = {
    PlacementStrategy.PACK: "STRICT_PACK",
    PlacementStrategy.SPREAD: "STRICT_SPREAD",
}


def normalize_placement_strategy(
    strategy: PlacementStrategy | str,
) -> PlacementStrategy:
    """Normalize config/runtime placement values to the shared enum."""
    return (
        strategy
        if isinstance(strategy, PlacementStrategy)
        else PlacementStrategy(strategy)
    )


def ray_placement_group_strategy(strategy: PlacementStrategy | str) -> str:
    """Map the shared placement enum to Ray placement-group strategy names."""
    return _RAY_PLACEMENT_STRATEGIES[normalize_placement_strategy(strategy)]


@ray.remote(num_gpus=1)
class CanaryWorker:
    """A worker meant to inspect state."""

    def get_cuda_visible_devices(self) -> list[str]:
        """Get the CUDA visible devices of the current node."""
        return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

    def get_gpu_ids(self) -> list[str]:
        """Get the GPU IDs of the current node."""
        return ray.get_gpu_ids()

    def get_node_ip_address(self) -> str:
        """Get the IP address of the current node."""
        return get_current_ray_node_ip_address()


@ray.remote
class LockWorker:
    """A lock that can be acquired and released."""

    def __init__(self):
        self._locked = False

    def acquire(self) -> bool:
        """Try to acquire the lock."""
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self) -> None:
        """Release the lock."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
