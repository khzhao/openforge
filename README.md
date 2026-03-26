<p align="center">
  <img src="./assets/openforge-ninja.png" alt="OpenForge banner" width="960" />
</p>

<h1 align="center"><code>openforge</code></h1>

<p align="center">
  <strong>Gateway-first post-training for LLM agents.</strong>
</p>

<p align="center">
  Train the agent, not the glue code.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#openforge-ninja-api">Ninja API</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#why-it-feels-different">Why OpenForge</a>
</p>

<p align="center">
  Bring up one live runtime, attach a small Python agent through <code>ninja</code>,
  and keep rollout, training, checkpoints, and session lifecycle behind one surface.
</p>

<p align="center">
  Gateway-first RL • Minimal agent API • Config-first setup
</p>

## Correspondence

- Kevin Zhao: `kzhao16 [at] gmail [dot] com`
- Zhaoran Wang: `zhaoran.wang [at] u [dot] northwestern [dot] edu`

## OpenForge

Most post-training repos ask you to edit launch scripts, shell state, and
framework internals at the same time. OpenForge is built around a simpler
contract:

1. Start a gateway from YAML.
2. Start a training session from YAML.
3. Define a Python agent with `@ninja.agent()`.
4. Run it directly or train it with `ninja.train(...)`.

That is the catch phrase of the repo in practice: **train the agent, not the
glue code.**

## OpenForge Ninja API

This is the center of the project:

```python
import openforge.ninja as ninja

@ninja.agent()
def agent(client, *, prompt: str, target: str) -> float:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    reward = ...
    return reward

summary = ninja.train(
    agent,
    inputs=[{"prompt": "2 + 2 = ?", "target": "4"}],
    group_size=8,
)
```

The important API points are:

- `agent(...)`
  Runs one trajectory through the active gateway and returns one reward.
- `agent.sample(requests=[...], group_size=N)`
  Collects one or more stored trajectories without waiting for a train update.
- `ninja.train(agent, inputs=[...], group_size=N)`
  Runs grouped trajectories, stores them, and waits for those trajectories to be
  trained before returning a summary.

If a gateway and session are active, `@ninja.agent()` discovers them
automatically. If you need to target a different gateway explicitly, pass a
`GatewayServerConfig` as the decorator argument, for example
`@ninja.agent(gateway_config)`.

The active environment is recorded on the local machine at:

- `$OPENFORGE_CACHE_HOME/openforge/active_gateway.json`
- `~/.cache/openforge/active_gateway.json` when `OPENFORGE_CACHE_HOME` is unset

## Why It Feels Different

- **Gateway-first control plane.** Runtime ownership, session lifecycle,
  generation, checkpoint export, and metadata live behind one API.
- **One public CLI.** OpenForge installs a single user-facing command,
  `openforge`, for gateway and session lifecycle.
- **Small Python surface.** If you can write a reward function around
  `client.chat.completions.create(...)`, you can train with OpenForge through
  `ninja.train(...)`.
- **Shared local discovery.** Once a gateway and session are active, Ninja
  scripts attach automatically. You do not need to thread gateway/session flags
  through every example.
- **Config-first workflow.** Cluster, model, rollout, and training settings
  live in YAML. The CLI starts things from config; it does not become a second
  config system.

## System At A Glance

| Layer | What it does |
| --- | --- |
| `gateway` | Runs the FastAPI control plane, owns the live runtime, exposes `/health`, `/info`, session lifecycle, generation, and checkpoint export. |
| `ninja` | The user-facing Python API for agent code. It registers agents, routes OpenAI-style chat-completions calls through the active gateway/session, and exposes grouped training via `ninja.train(...)`. |
| `active_state` | Stores the active gateway and active session on the local machine so scripts can auto-discover the current environment. |
| `benchmarks` | Prompt-building and scoring helpers used by the bundled examples and benchmarks. |
| `configs` | Pydantic-backed YAML models for gateway, cluster, model, rollout, and training setup. |
| `data` | SQLite-backed storage for sessions and trajectories. |
| `rollout` / `train` | SGLang rollout management and FSDP2 training infrastructure. |
| `examples` | Runnable configs, training scripts, shared helpers, and the batching benchmark. |

## Installation

### Requirements

- Linux
- Python 3.10+
- NVIDIA GPU(s) with a working CUDA stack
- `uv`

### Install

```bash
uv venv --python 3.10
source .venv/bin/activate

uv sync --dev
uv pip install -r requirements.txt
```

`uv pip install -r requirements.txt` is required for the runtime stack.

OpenForge exposes one CLI:

```bash
openforge --help
```

The module entrypoint still works too:

```bash
python -m openforge.cli.main --help
```

## Configuration

Start from the bundled example files:

- `examples/gsm8k/gateway.yaml`
- `examples/gsm8k/runtime.yaml`

The gateway config controls:

- gateway host and port
- data path
- cluster shape

The runtime config controls:

- algorithm
- model
- rollout engine groups
- training topology and checkpoints

The CLI only accepts config paths. If you want to change behavior, change the
YAML.

The bundled GSM8K example is configured for:

- 1 node
- 4 GPUs per node
- 1 FSDP2 training worker
- 3 SGLang rollout replicas
- GRPO on `Qwen/Qwen2.5-0.5B-Instruct`

There is also a Search-R1-style example configured for:

- GRPO + TIS (`grpo_tis`)
- `Qwen/Qwen2.5-3B-Instruct`
- one Python `search` tool inside a multi-turn Ninja agent

OpenForge runs on Ray, but the default behavior is simple:

- if `RAY_ADDRESS` is unset, `session start` creates a local Ray runtime
- if `RAY_ADDRESS` is set, `session start` attaches to that cluster

To attach to an existing Ray cluster instead of using a local one:

```bash
export RAY_ADDRESS="ray://<head-node>:10001"
```

## Quick Start

### Fastest Path

```bash
bash examples/gsm8k/run_ninja_train.sh
```

This wrapper starts the gateway, starts a session, runs the GSM8K Ninja
training example, and cleans up on exit.

### Manual Flow

Shell 1:

```bash
openforge gateway start --config examples/gsm8k/gateway.yaml
```

Shell 2:

```bash
openforge session start --runtime-config examples/gsm8k/runtime.yaml
```

Shell 3:

```bash
PYTHONUNBUFFERED=1 python -m examples.gsm8k.train_ninja \
  --artifact-dir /tmp/openforge-gsm8k-train \
  --total-epochs 15
```

The training script attaches to the active gateway and active session
automatically and uses `ninja.train(...)` under the hood.

`session start` is the heavy step. It creates or attaches to Ray, allocates the
runtime, and may take a few minutes on a cold start.

If you want to use an existing Ray cluster instead of the local default, export
`RAY_ADDRESS` before running `session start`.

When you are done:

```bash
openforge session stop
openforge gateway stop
```

### Search-R1-Style Example

The Search-R1 example keeps the same gateway/session flow, but the agent body is
multi-turn and uses an explicit search tool:

```bash
bash examples/search_r1/run_ninja_train.sh
```

Important details:

- the runtime config is `examples/search_r1/runtime.yaml`
- the example runtime uses 4 GPUs as `2` train workers plus `2` rollout replicas
- the agent script is `python -m examples.search_r1.train_ninja`
- the wrapper uses `examples/search_r1/gateway.yaml` for an isolated gateway DB
- the built-in prompt-data default is `/home/guo/kzhao/data/kzhao/search_r1_train.parquet`
- the built-in field defaults are `input-key=prompt` and `label-key=reward_model`
- `session start` is the slow step on a cold `Qwen/Qwen2.5-3B-Instruct` run

If your Search-R1 parquet lives somewhere else, you can still override it:

```bash
bash examples/search_r1/run_ninja_train.sh \
  --prompt-data /path/to/train.parquet
```

## Repository Layout

```text
.
├── assets/
│   └── openforge-ninja.png
├── examples/
│   ├── benchmark_ninja_register_batching.py
│   ├── shared.py
│   ├── gsm8k/
│   │   ├── common.py
│   │   ├── gateway.yaml
│   │   ├── run_ninja_train.sh
│   │   ├── runtime.yaml
│   │   └── train_ninja.py
│   └── search_r1/
│       ├── gateway.yaml
│       ├── run_ninja_train.sh
│       ├── runtime.yaml
│       └── train_ninja.py
├── src/openforge/
│   ├── active_state.py
│   ├── algo/
│   ├── benchmarks/
│   ├── cli/
│   ├── configs/
│   ├── data/
│   ├── gateway/
│   ├── ninja/
│   ├── rollout/
│   │   ├── router/
│   │   └── sglang/
│   ├── train/
│   │   ├── fsdp2/
│   │   └── server/
│   ├── runtime.py
│   └── utils/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

Key paths:

- `src/openforge/cli/main.py`
  The only public CLI entrypoint.
- `src/openforge/ninja/__init__.py`
  Agent registration, request execution, checkpoint export, and grouped
  training.
- `src/openforge/gateway/`
  FastAPI server, runtime/service lifecycle, and gateway-facing types.
- `src/openforge/rollout/`
  Rollout orchestration plus router and SGLang integrations.
- `src/openforge/train/`
  Training manager/runtime code, the FSDP2 backend, and the train-side server.
- `src/openforge/benchmarks/gsm8k.py`
  GSM8K prompt building and scoring helpers used by the examples.
- `examples/shared.py`
  Shared artifact helpers and the outer `ninja.train(...)` scheduling loop used
  by the bundled examples.
- `tests/`
  Unit, integration, and real-runtime coverage for CLI, gateway, rollout,
  training, and Ninja flows.
- `checkpoints/`
  A generated local output directory that may appear after training runs. It is
  gitignored and not part of the source layout.

## Development

```bash
ruff format src tests examples
ruff check src tests examples
python tests/test_active_state.py
python tests/test_cli.py
python tests/test_gateway_service.py
python tests/test_ninja.py
```

## Current Scope

OpenForge is intentionally focused today: gateway-driven post-training with
GRPO and GRPO+TIS, SGLang rollout, FSDP2 training, and the bundled GSM8K plus
Search-R1-style examples. If you want a small agent API on top of a real
runtime stack, that is the point of this repo.
