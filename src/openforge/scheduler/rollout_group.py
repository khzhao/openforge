# Copyright 2026 openforge

from __future__ import annotations

from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openforge.configs import (
    ExportedPolicy,
    OpenForgeConfig,
    PlacementStrategy,
    RolloutEndpoint,
)
from openforge.scheduler.utils import ray_placement_group_strategy
from openforge.scheduler.workers import PDRouterWorker, RolloutWorker

if TYPE_CHECKING:
    from openforge.scheduler.actor_ref_group import ActorRefGroup


class RolloutGroup:
    """Ray actor group that owns the configured rollout engine fleet."""

    def __init__(
        self,
        cfg: OpenForgeConfig,
        strategy: PlacementStrategy | str | None = None,
    ) -> None:
        self.cfg = cfg
        self._resolved = cfg.rollout.resolve(cfg.cluster)
        self._placement_groups: list[ray.util.placement_group.PlacementGroup] = []
        self._workers: list[ray.actor.ActorHandle] = []
        self._router_worker: ray.actor.ActorHandle | None = None
        self._engine_endpoints: list[RolloutEndpoint] = []
        self._router_endpoint: RolloutEndpoint | None = None

        self._create_workers(strategy=strategy)

    def initialize(self) -> list[RolloutEndpoint]:
        """Initialize all rollout engines and optional PD router."""
        self._engine_endpoints = ray.get(
            [
                worker.initialize.remote(self.cfg, engine=engine)
                for worker, engine in zip(
                    self._workers, self._resolved.engines, strict=True
                )
            ]
        )

        if self.cfg.rollout.engine_topology == "pd":
            self._router_endpoint = self._initialize_router()
            return [self._router_endpoint]

        self._router_endpoint = None
        return self.routable_endpoints()

    def shutdown(self) -> None:
        """Stop all rollout subprocesses and release Ray placement groups."""
        if self._router_worker is not None:
            ray.get(self._router_worker.shutdown.remote())
            self._router_worker = None
            self._router_endpoint = None

        ray.get([worker.shutdown.remote() for worker in self._workers])
        self._workers.clear()
        self._engine_endpoints.clear()

        for pg in self._placement_groups:
            ray.util.remove_placement_group(pg)
        self._placement_groups.clear()

    def engine_endpoints(self) -> list[RolloutEndpoint]:
        """Return the current per-engine endpoint view."""
        self._engine_endpoints = ray.get(
            [worker.endpoint.remote() for worker in self._workers]
        )
        return self._engine_endpoints

    def routable_endpoints(self) -> list[RolloutEndpoint]:
        """Return the upstream endpoints that a gateway should target."""
        if self.cfg.rollout.engine_topology == "pd":
            if self._router_worker is None:
                return []
            assert self._router_endpoint is not None
            self._router_endpoint = ray.get(self._router_worker.endpoint.remote())
            return [self._router_endpoint] if self._router_endpoint.healthy else []

        return [
            endpoint
            for endpoint in self.engine_endpoints()
            if endpoint.role == "regular"
            and endpoint.url is not None
            and endpoint.healthy
        ]

    def load_weights_from_policy(self, policy: ExportedPolicy) -> list[RolloutEndpoint]:
        """Restart rollout engines from an exported training policy."""
        self._engine_endpoints = ray.get(
            [worker.load_weights_from_policy.remote(policy) for worker in self._workers]
        )

        if self.cfg.rollout.engine_topology == "pd":
            self._router_endpoint = self._restart_router()

        return self.routable_endpoints()

    def load_latest_weights_from_train(self) -> list[RolloutEndpoint]:
        """Load the most recent exported training policy into rollout."""
        latest = self.cfg.train.read_exported_policy(latest=True)
        if latest is None:
            raise FileNotFoundError(
                "no rollout policy export found under train.checkpoints_dir"
            )
        return self.load_weights_from_policy(latest)

    def sync_from_train(
        self,
        train_group: ActorRefGroup,
        *,
        step: int,
        policy_version: int,
    ) -> ExportedPolicy:
        """Export from training and then restart rollout from that artifact."""
        policy = train_group.export_policy_for_rollout(
            step=step,
            policy_version=policy_version,
        )
        self.load_weights_from_policy(policy)
        return policy

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

    def _initialize_router(self) -> RolloutEndpoint:
        engine_endpoints = self.engine_endpoints()
        prefill_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "prefill"
        ]
        decode_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "decode"
        ]

        remote_router = ray.remote(PDRouterWorker)
        self._router_worker = remote_router.options(num_cpus=0).remote()
        return ray.get(
            self._router_worker.initialize.remote(
                prefill_endpoints=prefill_endpoints,
                decode_endpoints=decode_endpoints,
                checkpoints_dir=self.cfg.train.checkpoints_dir,
            )
        )

    def _restart_router(self) -> RolloutEndpoint:
        if self._router_worker is None:
            return self._initialize_router()

        engine_endpoints = self.engine_endpoints()
        prefill_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "prefill"
        ]
        decode_endpoints = [
            endpoint for endpoint in engine_endpoints if endpoint.role == "decode"
        ]
        return ray.get(
            self._router_worker.restart.remote(
                prefill_endpoints=prefill_endpoints,
                decode_endpoints=decode_endpoints,
            )
        )
