# Copyright 2026 openforge

from typing import Sequence

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs.models import OpenForgeConfig
from openforge.configs.rollout import RolloutEndpoint
from openforge.configs.topology import PlacementStrategy
from openforge.policy.types import PolicyArtifactRef
from openforge.rollout.types import RolloutWorkerSpec
from openforge.rollout.worker import RolloutWorker
from openforge.utils.ray import ray_placement_group_strategy

__all__ = ["RolloutWorkerGroup"]


class RolloutWorkerGroup:
    """Ray actor group that owns the configured rollout engine fleet."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        strategy: PlacementStrategy | str | None = None,
    ) -> None:
        self.cfg = cfg
        self._resolved = cfg.rollout.resolve(cfg.cluster)
        self._placement_groups: list[PlacementGroup] = []
        self._workers: list[ray.actor.ActorHandle] = []
        self._engine_endpoints: list[RolloutEndpoint] = []

        self._create_workers(strategy=strategy)

    def initialize(self) -> list[RolloutEndpoint]:
        specs = [self._build_worker_spec(engine) for engine in self._resolved.engines]
        self._engine_endpoints = ray.get(
            [
                worker.initialize.remote(spec)
                for worker, spec in zip(self._workers, specs, strict=True)
            ]
        )
        return self.routable_endpoints()

    def shutdown(self) -> None:
        ray.get([worker.shutdown.remote() for worker in self._workers])
        self._workers.clear()
        self._engine_endpoints.clear()

        for pg in self._placement_groups:
            ray.util.remove_placement_group(pg)
        self._placement_groups.clear()

    def engine_endpoints(self) -> list[RolloutEndpoint]:
        self._engine_endpoints = ray.get(
            [worker.endpoint.remote() for worker in self._workers]
        )
        return self._engine_endpoints

    def routable_endpoints(self) -> list[RolloutEndpoint]:
        return [
            endpoint
            for endpoint in self.engine_endpoints()
            if endpoint.role == "regular"
            and endpoint.url is not None
            and endpoint.healthy
        ]

    def load_policy_artifact(
        self,
        artifact: PolicyArtifactRef,
        *,
        engine_ids: Sequence[int] | None = None,
    ) -> list[RolloutEndpoint]:
        target_workers = self._target_workers(engine_ids)
        ray.get(
            [worker.load_policy_artifact.remote(artifact) for worker in target_workers]
        )
        self._refresh_after_update()
        return self.routable_endpoints()

    def pause_generation(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
        mode: str = "abort",
    ) -> None:
        ray.get(
            [
                worker.pause_generation.remote(mode=mode)
                for worker in self._target_workers(engine_ids)
            ]
        )

    def continue_generation(
        self,
        *,
        engine_ids: Sequence[int] | None = None,
    ) -> None:
        ray.get(
            [
                worker.continue_generation.remote()
                for worker in self._target_workers(engine_ids)
            ]
        )

    def _create_workers(
        self,
        *,
        strategy: PlacementStrategy | str | None,
    ) -> None:
        remote_worker = ray.remote(RolloutWorker)
        resolved_by_group: dict[str, list] = {}
        for engine in self._resolved.engines:
            resolved_by_group.setdefault(engine.group_name, []).append(engine)

        for group in self.cfg.rollout.engines:
            group_strategy = strategy or group.placement.strategy

            bundles = [
                {"GPU": group.gpus_per_engine, "CPU": group.cpus_per_engine}
            ] * group.replicas
            pg = placement_group(
                bundles,
                strategy=ray_placement_group_strategy(group_strategy),
            )
            ray.get(pg.ready())
            self._placement_groups.append(pg)

            resolved_engines = sorted(
                resolved_by_group[group.name],
                key=lambda engine: engine.replica_index,
            )
            for bundle_index, _engine in enumerate(resolved_engines):
                worker = remote_worker.options(
                    num_gpus=group.gpus_per_engine,
                    num_cpus=group.cpus_per_engine,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_index,
                    ),
                ).remote()
                self._workers.append(worker)

    def _build_worker_spec(self, engine) -> RolloutWorkerSpec:
        host = ray.get(self._workers[engine.engine_id].resolve_node_ip_address.remote())
        port = ray.get(
            self._workers[engine.engine_id].allocate_port.remote(
                start=30000 + (engine.engine_id * 10)
            )
        )
        return RolloutWorkerSpec(
            cfg=self.cfg,
            engine=engine,
            host=host,
            port=port,
            bootstrap_port=None,
            policy_version=None,
        )

    def _target_workers(
        self,
        engine_ids: Sequence[int] | None,
    ) -> list[ray.actor.ActorHandle]:
        if engine_ids is None:
            return list(self._workers)
        engine_id_set = set(engine_ids)
        return [
            worker
            for worker, engine in zip(
                self._workers, self._resolved.engines, strict=True
            )
            if engine.engine_id in engine_id_set
        ]

    def _refresh_after_update(self) -> None:
        self._engine_endpoints = self.engine_endpoints()
