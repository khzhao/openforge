# Copyright 2026 openforge

from _manager_weight_sync_common import DEFAULT_MODEL, run_weight_sync_e2e


if __name__ == "__main__":
    raise SystemExit(
        run_weight_sync_e2e(
            model_path=DEFAULT_MODEL,
            train_gpus=1,
            rollout_replicas=1,
            gpus_per_replica=1,
            cpus_per_rollout_replica=1,
            base_policy_version=402,
            sync_modes=("distributed",),
        )
    )
