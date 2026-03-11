# Copyright 2026 openforge

from pydantic import model_validator

from .base import OpenForgeBaseModel


class ClusterConfig(OpenForgeBaseModel):
    """Top-level physical cluster inventory."""

    num_nodes: int
    gpus_per_node: int
    cpus_per_node: int

    @model_validator(mode="after")
    def _validate_cluster(self) -> "ClusterConfig":
        if self.num_nodes <= 0:
            raise ValueError("cluster.num_nodes must be > 0")
        if self.gpus_per_node <= 0:
            raise ValueError("cluster.gpus_per_node must be > 0")
        if self.cpus_per_node < 0:
            raise ValueError("cluster.cpus_per_node must be >= 0")
        return self

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def total_cpus(self) -> int:
        return self.num_nodes * self.cpus_per_node
