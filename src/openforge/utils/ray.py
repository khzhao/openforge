# Copyright 2026 openforge

import os

import ray
import torch
from loguru import logger
from ray.util.placement_group import placement_group

from openforge.configs.models import OpenForgeConfig

NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


@ray.remote(num_gpus=1)
class CanaryWorker:
    """Just get the GPU IDs visible to a worker."""

    def cuda_visible_devices(self) -> str:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

    def get_node_ip_address(self) -> str:
        return get_current_ray_node_ip_address()

    def get_gpu_id(self) -> int:
        gpu_ids = get_current_ray_gpu_ids()
        assert len(gpu_ids) == 1, "Expected 1 GPU per worker"
        return gpu_ids[0]

    def get_ip_and_gpu_id(self):
        ip_addr = self.get_node_ip_address()
        gpu_id = self.get_gpu_id()
        return ip_addr, gpu_id


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


def get_current_physical_gpu_id() -> str:
    """Get the physical GPU id of the current CUDA device."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def ray_noset_visible_devices() -> bool:
    """Return True when Ray accelerator visibility rewriting is disabled."""
    return any(
        os.environ.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
    )


def _sort_key(x):
    """Sort bundles by numeric IPv4 address, then GPU id."""
    _, node_ip, gpu_id = x
    return (tuple(map(int, node_ip.split("."))), gpu_id)


def create_placement_groups(cfg: OpenForgeConfig):
    """Determine how to place workers on GPUS given the various roles.

    We are assuming a fully disaggregated placement and division between
    train and rollout workers. In the future we plan to support other modes
    including the following:
    - Debug Train Only
    - Rollout Only
    - Colocated Train and Rollout
    - Disaggregated Train and Rollout
    """
    # 1. Get total number of GPUs requested by the user
    train_gpus = cfg.train.total_gpus
    rollout_gpus = cfg.rollout.total_gpus
    total_gpus = train_gpus + rollout_gpus
    logger.info(f"Total GPUs requested: {total_gpus}")
    cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    available_gpus = float(ray.available_resources().get("GPU", 0.0))
    logger.info(
        "Ray GPU resources: cluster={} available={}",
        cluster_gpus,
        available_gpus,
    )
    if cluster_gpus < total_gpus:
        raise RuntimeError(
            "Ray did not detect enough GPUs for this runtime: "
            f"requested {total_gpus}, cluster reported {cluster_gpus}. "
            "Check CUDA visibility and GPU access on this machine."
        )

    # 2. Create canary workers to get the cluster topology, then kill them
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    canary_workers = []
    for i in range(len(bundles)):
        canary_workers.append(
            CanaryWorker.options(
                placement_group=pg, placement_group_bundle_index=i
            ).remote()
        )

    ip_and_gpu_ids = ray.get(
        [worker.get_ip_and_gpu_id.remote() for worker in canary_workers]
    )
    logger.info(f"IP and GPU IDs: {ip_and_gpu_ids}")
    for canary_worker in canary_workers:
        ray.kill(canary_worker)

    bundle_infos = [
        (i, ip_and_gpu_ids[i][0], ip_and_gpu_ids[i][1]) for i in range(len(bundles))
    ]
    sorted_bundle_infos = sorted(bundle_infos, key=_sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [ip_and_gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    logger.info("Placement group bundle mapping (logical → actual):")
    for logical_bundle_idx, actual_bundle_idx in enumerate(pg_reordered_bundle_indices):
        ip, physical_gpu_id = ip_and_gpu_ids[actual_bundle_idx]
        logger.info(
            "  logical_bundle_idx={:2d} actual_bundle_idx={:2d} node={} physical_gpu_id={}",
            logical_bundle_idx,
            actual_bundle_idx,
            ip,
            physical_gpu_id,
        )

    return {
        "actor": (
            pg,
            pg_reordered_bundle_indices[:train_gpus],
            pg_reordered_gpu_ids[:train_gpus],
        ),
        "rollout": (
            pg,
            pg_reordered_bundle_indices[train_gpus:],
            pg_reordered_gpu_ids[train_gpus:],
        ),
    }
