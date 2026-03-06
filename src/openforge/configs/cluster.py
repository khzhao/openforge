# Copyright 2026 openforge

from __future__ import annotations

from pydantic import RootModel, model_validator

from .base import OpenForgeBaseModel


class NodePoolConfig(OpenForgeBaseModel):
    """Homogeneous node pool available to OpenForge workloads."""

    node_pool: str
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int

    @model_validator(mode="after")
    def _validate_pool(self) -> NodePoolConfig:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if self.num_gpus_per_node <= 0:
            raise ValueError("num_gpus_per_node must be > 0")
        if self.num_cpus_per_node < 0:
            raise ValueError("num_cpus_per_node must be >= 0")
        return self

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node

    @property
    def total_cpus(self) -> int:
        return self.num_nodes * self.num_cpus_per_node


class ClusterConfig(RootModel[list[NodePoolConfig]]):
    """Top-level physical cluster inventory."""

    @model_validator(mode="after")
    def _validate_cluster(self) -> ClusterConfig:
        if not self.root:
            raise ValueError("cluster must not be empty")

        names = [pool.node_pool for pool in self.root]
        if len(names) != len(set(names)):
            raise ValueError("cluster node-pool names must be unique")
        return self

    @property
    def node_pools(self) -> list[NodePoolConfig]:
        return self.root

    def __iter__(self):
        return iter(self.root)

    def get_pool(self, node_pool: str) -> NodePoolConfig:
        for pool in self.root:
            if pool.node_pool == node_pool:
                return pool
        raise ValueError(f"Unknown cluster node pool: {node_pool}")
