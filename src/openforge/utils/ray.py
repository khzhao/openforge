# Copyright 2026 openforge

import ray


def get_current_ray_node_ip_address() -> str:
    """Get the IP address of the current Ray node."""
    assert ray.is_initialized(), (
        "Ray must be initialized before getting the node IP address"
    )
    address = ray.util.get_node_ip_address()
    return address
