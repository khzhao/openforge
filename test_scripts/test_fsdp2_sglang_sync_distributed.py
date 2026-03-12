# Copyright 2026 openforge

from _fsdp2_sglang_sync_common import run_sync_test


if __name__ == "__main__":
    raise SystemExit(run_sync_test("distributed", default_policy_version=403))
