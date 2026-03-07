# Copyright 2026 openforge

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from .base import OpenForgeBaseModel
from .cluster import ClusterConfig
from .topology import ParallelismConfig, PlacementConfig


class MixedPrecisionConfig(OpenForgeBaseModel):
    """Mixed precision configuration for FSDP2."""

    param_dtype: Literal["float32", "float16", "bfloat16"]
    reduce_dtype: Literal["float32", "float16", "bfloat16"]


class OffloadConfig(OpenForgeBaseModel):
    """CPU offload configuration for FSDP2."""

    mode: Literal["none", "cpu"]
    pin_memory: bool


class AMPConfig(OpenForgeBaseModel):
    """Automatic Mixed Precision configuration."""

    enabled: bool
    precision: Literal["float32", "float16", "bfloat16"]
    use_grad_scaler: bool


class OptimizerConfig(OpenForgeBaseModel):
    """Optimizer hyperparameters."""

    lr: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    max_grad_norm: float


class SchedulerConfig(OpenForgeBaseModel):
    """Learning rate scheduler settings."""

    type: Literal["constant", "cosine", "linear"]
    warmup_steps: int
    min_lr: float
    num_cycles: float
    min_lr_rate: float | None = None


class FSDP2Config(OpenForgeBaseModel):
    """FSDP2 backend configuration."""

    gradient_checkpointing: bool
    reshard_after_forward: bool
    shard_modules: list[str] | None = None
    mixed_precision: MixedPrecisionConfig
    offload: OffloadConfig
    amp: AMPConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


class MegatronConfig(OpenForgeBaseModel):
    """Megatron backend configuration."""

    def model_post_init(self) -> None:
        raise NotImplementedError("Megatron backend is not implemented yet")


@dataclass(slots=True)
class ResolvedTrainWorker:
    """Concrete per-rank worker allocation derived from TrainConfig."""

    rank: int
    node_pool: str
    cpus: int
    placement: PlacementConfig


@dataclass(slots=True)
class ResolvedTrainTopology:
    """Concrete training world expansion for runtime code."""

    node_pool: str
    placement: PlacementConfig
    cpus_per_worker: int
    parallelism: ParallelismConfig
    num_nodes: int
    num_gpus_per_node: int
    num_cpus_per_node: int
    workers: list[ResolvedTrainWorker]

    @property
    def world_size(self) -> int:
        return len(self.workers)

    @property
    def total_gpus(self) -> int:
        return self.world_size

    @property
    def total_cpus(self) -> int:
        return self.world_size * self.cpus_per_worker


@dataclass(slots=True)
class SerializedPolicyWeights:
    """Live rollout weight update payload serialized for SGLang."""

    step: int
    policy_version: int
    load_format: str
    serialized_weight_buckets: list[str]


class TrainConfig(OpenForgeBaseModel):
    """Configuration for the training process."""

    backend: Literal["fsdp2", "megatron"]
    backend_config: FSDP2Config | MegatronConfig

    global_batch_size: int
    mini_batch_size: int
    micro_batch_size: int
    checkpoints_dir: str

    cpus_per_worker: int
    parallelism: ParallelismConfig
    placement: PlacementConfig

    @model_validator(mode="after")
    def _validate_train_config(self) -> TrainConfig:
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be > 0")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if self.global_batch_size % self.mini_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by mini_batch_size")
        if self.mini_batch_size % self.micro_batch_size != 0:
            raise ValueError("mini_batch_size must be divisible by micro_batch_size")
        if self.cpus_per_worker < 0:
            raise ValueError("cpus_per_worker must be >= 0")

        if self.backend == "fsdp2":
            if not isinstance(self.backend_config, FSDP2Config):
                raise ValueError(
                    "backend_config must be FSDP2Config when backend is fsdp2"
                )
            if (
                self.parallelism.pipeline_parallel_size > 1
                or self.parallelism.tensor_parallel_size > 1
                or self.parallelism.context_parallel_size > 1
                or self.parallelism.expert_parallel_size > 1
            ):
                raise ValueError(
                    "FSDP2 does not support pipeline, tensor, context, or expert parallelism"
                )

        if self.backend == "megatron" and not isinstance(
            self.backend_config, MegatronConfig
        ):
            raise ValueError(
                "backend_config must be MegatronConfig when backend is megatron"
            )
        return self

    def resolve(self, cluster: ClusterConfig) -> ResolvedTrainTopology:
        pool = cluster.get_pool(self.placement.node_pool)
        total_gpus = self.total_gpus
        total_cpus = self.total_cpus

        if total_gpus > pool.total_gpus:
            raise ValueError(
                f"train topology requests {total_gpus} GPUs from node pool "
                f"{pool.node_pool}, but only {pool.total_gpus} are available"
            )
        if total_cpus > pool.total_cpus:
            raise ValueError(
                f"train topology requests {total_cpus} CPUs from node pool "
                f"{pool.node_pool}, but only {pool.total_cpus} are available"
            )

        workers = [
            ResolvedTrainWorker(
                rank=rank,
                node_pool=pool.node_pool,
                cpus=self.cpus_per_worker,
                placement=self.placement,
            )
            for rank in range(self.parallelism.world_size)
        ]
        return ResolvedTrainTopology(
            node_pool=pool.node_pool,
            placement=self.placement,
            cpus_per_worker=self.cpus_per_worker,
            parallelism=self.parallelism,
            num_nodes=pool.num_nodes,
            num_gpus_per_node=pool.num_gpus_per_node,
            num_cpus_per_node=pool.num_cpus_per_node,
            workers=workers,
        )

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(self.mini_batch_size / self.micro_batch_size)

    @property
    def num_workers(self) -> int:
        return self.parallelism.world_size

    @property
    def total_gpus(self) -> int:
        return self.num_workers

    @property
    def total_cpus(self) -> int:
        return self.num_workers * self.cpus_per_worker
