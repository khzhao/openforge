# Copyright 2026 openforge


from openforge.configs import OpenForgeConfig


class ActorCriticRefGroup:
    """A wrapper about actor-critic-ref workers."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
    ):
        self.cfg = cfg
        self.num_nodes = cfg.train.num_nodes
        self.num_gpus_per_node = cfg.train.num_gpus_per_node
        self.num_cpus_per_node = cfg.train.num_cpus_per_node
        self.data_parallel_size = cfg.train.data_parallel_size
        self.pipeline_parallel_size = cfg.train.pipeline_parallel_size
        self.tensor_parallel_size = cfg.train.tensor_parallel_size
