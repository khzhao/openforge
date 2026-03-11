# Copyright 2026 openforge

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch.distributed.fsdp import FSDPModule
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openforge.policy.types import PolicyArtifactRef

from .checkpoint import get_full_state_dict

if TYPE_CHECKING:
    from .base import FSDP2Engine


def export_policy_artifact(
    backend: "FSDP2Engine",
    *,
    step: int,
    policy_version: int,
) -> PolicyArtifactRef | None:
    """Export the current FSDP2 model as a disk-backed inference artifact."""
    model = cast(FSDPModule, backend.main_model)
    full_state = get_full_state_dict(model)
    if backend.rank != 0:
        return None

    checkpoints_dir = Path(backend.cfg.train.checkpoints)
    artifact_dir = checkpoints_dir / (
        f"policy_step_{step:08d}.version_{policy_version:08d}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(
        backend.cfg.model.model_name_or_path,
        trust_remote_code=True,
    )
    with torch.device("meta"):
        export_model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=True,
        )
    export_model.save_pretrained(
        artifact_dir,
        state_dict=full_state,
        safe_serialization=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        backend.cfg.model.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(artifact_dir)
    return PolicyArtifactRef(
        step=step,
        policy_version=policy_version,
        path=str(artifact_dir),
        load_format="safetensors",
    )
