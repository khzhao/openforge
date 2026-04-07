# OpenClaw Example

This example runs a small middleware service between OpenClaw and an active OpenForge gateway:

`OpenClaw -> examples/openclaw -> OpenForge`

The middleware accepts normal OpenAI-style `POST /v1/chat/completions` requests from OpenClaw, creates one OpenForge trajectory per trainable turn, and closes the previous pending trajectory when the next request arrives for the same OpenClaw session.

Internally it talks to the OpenForge gateway through Ninja's session client layer from one small `app.py`.

## What This Example Does

- Accepts OpenClaw provider traffic on `/v1/chat/completions`
- Accepts `stream=true` requests and re-emits the final reply as OpenAI-style SSE chunks
- Requires `X-Session-Id` and uses `X-Turn-Type` to classify turns
- Maps one OpenClaw session to one pending OpenForge train trajectory
- Uses the next request as delayed feedback for the previous assistant turn
- Uses a validation-purpose judge call to score the most recent pending `main` turn
- Forwards the actual generation request to the active OpenForge gateway

## What This Example Does Not Do

- No true upstream token streaming; streaming responses are synthesized from the completed upstream reply
- No OPD or combined RL
- No hidden K-sample shadow rollouts for GRPO
- No production-grade reward model
- No distributed session store

This is an integration example, not a full OpenClaw-RL port.

## Prerequisites

1. Start an OpenForge gateway.
2. Start an OpenForge session.
3. Run this middleware.
4. Point OpenClaw at the middleware instead of the OpenForge gateway directly.

The middleware should run on a different port from the OpenForge gateway. In this example:

- OpenForge gateway: `127.0.0.1:8042`
- OpenClaw middleware: `127.0.0.1:8012`

The middleware discovers the active OpenForge gateway automatically when it runs on the same machine. You can also set an explicit upstream base URL:

```bash
export OPENFORGE_GATEWAY_BASE_URL="http://127.0.0.1:8042"
```

Local starter configs are included in this folder:

- [gateway.yaml](/home/guo/kzhao/github/openforge/examples/openclaw/gateway.yaml)
- [runtime.yaml](/home/guo/kzhao/github/openforge/examples/openclaw/runtime.yaml)
- [openclaw.json.example](/home/guo/kzhao/github/openforge/examples/openclaw/openclaw.json.example)
- [extensions/rl-training-headers/index.ts](/home/guo/kzhao/github/openforge/examples/openclaw/extensions/rl-training-headers/index.ts)
- [install_extension.sh](/home/guo/kzhao/github/openforge/examples/openclaw/install_extension.sh)

Example launcher flow:

```bash
openforge gateway start --config examples/openclaw/gateway.yaml
openforge session start --runtime-config examples/openclaw/runtime.yaml
OPENCLAW_MIDDLEWARE_PORT=8012 OPENFORGE_GATEWAY_BASE_URL=http://127.0.0.1:8042 python -m examples.openclaw.app
```

For replay-driven GSM8K experiments, reuse the same launcher with a lighter
runtime:

```bash
OPENFORGE_RUNTIME_CONFIG=examples/openclaw/replay/gsm8k_runtime.yaml \
  bash examples/openclaw/run_demo.sh
```

`run_demo.sh` starts the shipped gateway config on `127.0.0.1:8042`, writes both
SQLite paths into a fresh experiment directory, and prints the exact dashboard
command for that run.

Visibility dashboard:

```bash
python -m examples.openclaw.tui \
  --state-db /path/to/openclaw-state.sqlite3 \
  --gateway-yaml examples/openclaw/gateway.yaml \
  --gateway-base-url http://127.0.0.1:8042
```

`--state-db` is required. Pass either `--gateway-yaml` or `--gateway-db`; when
you launch through `run_demo.sh`, use the `--gateway-db` path that the script
prints.

Link the bundled OpenClaw extension into your local OpenClaw install:

```bash
bash examples/openclaw/install_extension.sh
```

Run a replay-driven GSM8K train pass against the middleware:

```bash
python -m examples.openclaw.replay.gsm8k_replay \
  --mode train \
  --state-db /tmp/openclaw-gsm8k-demo/openclaw-state.sqlite3 \
  --max-examples 1280 \
  --epochs 2
```

`gsm8k_replay.py` now defaults to the active OpenForge session model, so
switching runtimes does not require a separate `--model` flag unless you want to
override it explicitly.

The replay runtime at
[examples/openclaw/replay/gsm8k_runtime.yaml](/home/guo/kzhao/github/openforge/examples/openclaw/replay/gsm8k_runtime.yaml)
now mirrors the original GSM8K benchmark's `Qwen/Qwen2.5-0.5B-Instruct` model
family and batch shape. Use `--epochs` to replay the same split multiple times
so training can continue across repeated passes of the data.

## Run

```bash
OPENCLAW_MIDDLEWARE_PORT=8012 OPENFORGE_GATEWAY_BASE_URL=http://127.0.0.1:8042 python -m examples.openclaw.app
```

By default the middleware serves on `127.0.0.1:8012`, which keeps it separate
from the OpenForge gateway on `127.0.0.1:8042`.

Optional environment variables:

- `OPENCLAW_MIDDLEWARE_HOST`
- `OPENCLAW_MIDDLEWARE_PORT`
- `OPENFORGE_GATEWAY_BASE_URL`
- `OPENCLAW_STATE_DB`
- `OPENCLAW_JUDGE_M`
- `OPENCLAW_JUDGE_MAX_TOKENS`
- `OPENCLAW_JUDGE_TEMPERATURE`

The middleware now defaults `OPENCLAW_JUDGE_MAX_TOKENS` to `2048` so reasoning
judge models have enough budget to emit a final `\boxed{...}` score.

If `OPENCLAW_STATE_DB` is unset, the middleware stores state in
`~/.cache/openforge/examples/openclaw.sqlite3`.

## OpenClaw Configuration

Point the OpenClaw model provider at the middleware:

```json
{
  "models": {
    "providers": {
      "openforge-openclaw": {
        "baseUrl": "http://127.0.0.1:8012/v1",
        "apiKey": "unused",
        "api": "openai-completions",
        "models": [
          {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "name": "OpenForge Qwen 2.5 3B",
            "reasoning": true,
            "input": ["text"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "openforge-openclaw/Qwen/Qwen2.5-3B-Instruct"
      }
    }
  }
}
```

You also need to enable the bundled local extension from this example folder. The included [openclaw.json.example](/home/guo/kzhao/github/openforge/examples/openclaw/openclaw.json.example) shows both:

- the provider pointing at `http://127.0.0.1:8012/v1`
- the plugin entry for `rl-training-headers`

Recommended setup:

1. Run [install_extension.sh](/home/guo/kzhao/github/openforge/examples/openclaw/install_extension.sh) once to link the bundled extension into `~/.openclaw/extensions/rl-training-headers`.
2. Copy the example config into your OpenClaw config and keep the `plugins.entries.rl-training-headers` block enabled.

The example plugin config matches both middleware URLs:

- `127.0.0.1:8012/v1/chat/completions`
- `localhost:8012/v1/chat/completions`

For learning across turns, OpenClaw must send a stable session id. The bundled
OpenClaw RL header plugin is the intended way to do that:

- `X-Session-Id`
- `X-Turn-Type`

Reference:
- https://raw.githubusercontent.com/Gen-Verse/OpenClaw-RL/main/extensions/rl-training-headers/README.md

Requests missing `X-Session-Id` are rejected.

## Judge Logic

The example scores the most recent pending `main` turn by:

- extracting the next-state follow-up text for the session
- building the same boxed-score PRM prompt style used by OpenClaw-RL from the prior assistant reply plus that follow-up
- treating question-only or otherwise ambiguous follow-ups as neutral by default
- calling the active OpenForge session model in a validation-purpose trajectory
- controlling repeated judge samples with `OPENCLAW_JUDGE_M`, `OPENCLAW_JUDGE_TEMPERATURE`, and `OPENCLAW_JUDGE_MAX_TOKENS`
- majority-voting parsed `\boxed{...}` rewards across judge samples

Prompt construction and parsing live in [reward.py](/home/guo/kzhao/github/openforge/examples/openclaw/reward.py).

## Why This Is Middleware Instead Of `ninja`

`ninja` is a Python training API for local scripts that return rewards. OpenClaw still needs a provider-facing HTTP service, so the middleware stays in front. Inside the middleware, though, this example reuses Ninja's internal session client to hide the OpenForge `_openforge` trajectory metadata.
