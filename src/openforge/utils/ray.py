# Copyright 2026 openforge

import os

import ray

from openforge.configs.topology import PlacementStrategy

_RAY_PLACEMENT_STRATEGIES = {
    PlacementStrategy.PACK: "STRICT_PACK",
    PlacementStrategy.SPREAD: "STRICT_SPREAD",
}


def get_current_ray_node_ip_address() -> str:
    """Get the IP address of the current Ray node."""
    assert ray.is_initialized(), (
        "Ray must be initialized before getting the node IP address"
    )
    address = ray.util.get_node_ip_address()
    return address


def get_current_ray_gpu_ids() -> list[int]:
    """Get the physical GPU ids assigned to the current Ray worker."""
    assert ray.is_initialized(), (
        "Ray must be initialized before getting accelerator ids"
    )
    gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    return [int(gpu_id) for gpu_id in gpu_ids]


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
    """Just get the GPU IDs visible to a worker."""

    def cuda_visible_devices(self) -> str:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

    def get_node_ip_address(self) -> str:
        return get_current_ray_node_ip_address()
