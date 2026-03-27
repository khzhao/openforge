# Copyright 2026 openforge
# ruff: noqa: D103, E402

from __future__ import annotations

from _script_test_utils import install_test_stubs, run_tests

install_test_stubs()

from openforge.configs.algo import GRPOConfig
from openforge.configs.cluster import ClusterConfig
from openforge.configs.models import (
    DataConfig,
    GatewayConfig,
    GatewayServerConfig,
    ModelConfig,
    OpenForgeConfig,
)
from openforge.configs.rollout import (
    RolloutConfig,
    RolloutEngineGroupConfig,
    SGLangRequestConfig,
)
from openforge.configs.train import (
    AMPConfig,
    FSDP2Config,
    MixedPrecisionConfig,
    OffloadConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
)
from openforge.configs.topology import ParallelismConfig
from openforge.rollout.sglang.utils import generate_sglang_server_args
from openforge.rollout.sglang.types import EngineAddr, EngineSpec


def _cfg() -> OpenForgeConfig:
    return OpenForgeConfig(
        data=DataConfig(path=None),
        gateway=GatewayConfig(host="127.0.0.1", port=8000),
        cluster=ClusterConfig(num_nodes=1, gpus_per_node=2, cpus_per_node=8),
        algo=GRPOConfig(kl_coef=0.0),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            reference_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            attn_implementation="flash_attention_2",
        ),
        train=TrainConfig(
            backend="fsdp2",
            config=FSDP2Config(
                gradient_checkpointing=False,
                reshard_after_forward=False,
                mixed_precision=MixedPrecisionConfig(
                    param_dtype="bfloat16",
                    reduce_dtype="float32",
                ),
                offload=OffloadConfig(mode="none", pin_memory=False),
                amp=AMPConfig(
                    enabled=False,
                    precision="float32",
                    use_grad_scaler=False,
                ),
                optim=OptimizerConfig(
                    lr=1.0e-6,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1.0e-8,
                    weight_decay=0.0,
                    max_grad_norm=1.0,
                ),
                scheduler=SchedulerConfig(
                    type="constant",
                    warmup_steps=0,
                    min_lr=0.0,
                    num_cycles=0.5,
                ),
            ),
            global_batch_size=8,
            mini_batch_size=4,
            micro_batch_size=1,
            ppo_epochs=1,
            max_rollout_policy_lag=0,
            checkpoints="./checkpoints",
            cpus_per_worker=1,
            parallel=ParallelismConfig(
                data_parallel_size=1,
                fsdp_parallel_size=1,
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                context_parallel_size=1,
                expert_parallel_size=1,
            ),
        ),
        rollout=RolloutConfig(
            backend="sglang",
            request=SGLangRequestConfig(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_new_tokens=32,
                stop=[],
                stop_token_ids=[],
                skip_special_tokens=True,
                no_stop_trim=False,
            ),
            engine_groups=[
                RolloutEngineGroupConfig(
                    name="regular",
                    worker_type="regular",
                    replicas=1,
                    num_gpus_per_replica=1,
                    num_cpus_per_replica=1,
                    parallelism=ParallelismConfig(
                        data_parallel_size=1,
                        fsdp_parallel_size=1,
                        pipeline_parallel_size=1,
                        tensor_parallel_size=1,
                        context_parallel_size=1,
                        expert_parallel_size=1,
                    ),
                    enable_memory_saver=False,
                )
            ],
        ),
    )


def test_generate_sglang_server_args_starts_with_weight_version_zero() -> None:
    spec = EngineSpec(
        cfg=_cfg(),
        engine_name="regular-0",
        worker_type="regular",
        node_rank=0,
        num_nodes=1,
        engine_rank=0,
        gpu_rank_offset=0,
        base_gpu_id=0,
        num_gpus=1,
        num_cpus=1,
        parallelism=ParallelismConfig(
            data_parallel_size=1,
            fsdp_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            context_parallel_size=1,
            expert_parallel_size=1,
        ),
        pg=None,
        bundle_indices=[0],
        gpu_ids=[0],
        enable_memory_saver=False,
        sglang_server_overrides={},
    )
    addr = EngineAddr(
        host="127.0.0.1",
        port=31000,
        dist_init_addr="127.0.0.1:32000",
        nccl_port=33000,
    )

    server_args = generate_sglang_server_args(spec, addr)

    assert server_args.weight_version == "0"


def main() -> int:
    return run_tests([test_generate_sglang_server_args_starts_with_weight_version_zero])


if __name__ == "__main__":
    raise SystemExit(main())
