# Copyright 2026 openforge


from openforge.configs import OpenForgeConfig


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
        cfg: OpenForgeConfig,
        rank: int,
        world_size: int,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
