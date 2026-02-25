# Copyright 2026 openforge

from datetime import timedelta

import torch.distributed as dist


class ActorCriticRefWorker:
    """A worker for actor-critic-ref.

    We intentionally do not wrap this class with a ray.remote decorator. This
    will allow to write test cases for this class much more easily.

    [2026-02-25] @kzhao:
    We will restrict this class to colocate the actor/critic/ref
    workers together into the same group. We will also assume that
    everything is trained on a singular node consisting of 4-8 GPUs.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        self.rank = rank
        self.world_size = world_size

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                init_method=f"tcp://{master_addr}:{master_port}",
                timeout=timedelta(seconds=30),
            )
