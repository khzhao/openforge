# OpenClaw Example

This example runs a small middleware service between OpenClaw and an active OpenForge gateway:

`OpenClaw -> examples/openclaw -> OpenForge`

The middleware accepts normal OpenAI-style `POST /v1/chat/completions` requests from OpenClaw, creates one OpenForge trajectory per trainable turn, and closes the previous pending trajectory when the next request arrives for the same OpenClaw session.

Internally it talks to the OpenForge gateway through Ninja's session client layer from one small `app.py`.

## What This Example Does

- Accepts OpenClaw provider traffic on `/v1/chat/completions`
- Reads `X-Session-Id` and `X-Turn-Type` when available
- Maps one OpenClaw session to one pending OpenForge train trajectory
- Uses the next request as delayed feedback for the previous assistant turn
- Uses a validation-purpose judge call to score the most recent pending `main` turn
- Forwards the actual generation request to the active OpenForge gateway

## What This Example Does Not Do

- No streaming support
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

The middleware discovers the active OpenForge gateway automatically when it runs on the same machine. You can also set an explicit upstream base URL:

```bash
export OPENCLAW_OPENFORGE_BASE_URL="http://127.0.0.1:8000"
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
python -m examples.openclaw.app
```

Link the bundled OpenClaw extension into your local OpenClaw install:

```bash
bash examples/openclaw/install_extension.sh
```

## Run

```bash
python -m examples.openclaw.app
```

By default the middleware serves on `127.0.0.1:8011`.

Optional environment variables:

- `OPENCLAW_HOST`
- `OPENCLAW_PORT`
- `OPENCLAW_OPENFORGE_BASE_URL`
- `OPENCLAW_STATE_DB`

## OpenClaw Configuration

Point the OpenClaw model provider at the middleware:

```json
{
  "models": {
    "providers": {
      "openclaw-openforge": {
        "baseUrl": "http://127.0.0.1:8011/v1",
        "apiKey": "unused",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-8b",
            "name": "Qwen3 8B via OpenForge",
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
  }
}
```

You also need to enable the bundled local extension from this example folder. The included [openclaw.json.example](/home/guo/kzhao/github/openforge/examples/openclaw/openclaw.json.example) shows both:

- the provider pointing at `http://127.0.0.1:8011/v1`
- the plugin load path for `extensions/rl-training-headers`

For learning across turns, you should configure OpenClaw to send a stable session id. The OpenClaw RL header plugin is the cleanest way to do that:

- `X-Session-Id`
- `X-Turn-Type`

Reference:
- https://raw.githubusercontent.com/Gen-Verse/OpenClaw-RL/main/extensions/rl-training-headers/README.md

If `X-Session-Id` is not available, this example falls back to the OpenAI `user` field when present.

## Judge Logic

The example scores the most recent pending `main` turn by:

- extracting the next-state follow-up text for the session
- building a compact judge prompt from the prior assistant reply plus that follow-up
- running the judge through OpenForge with `purpose="validation"` so the judge output is not stored for training
- parsing the judge JSON into a scalar reward for the pending train trajectory

Prompt construction and parsing live in [reward.py](/home/guo/kzhao/github/openforge/examples/openclaw/reward.py).

## Why This Is Middleware Instead Of `ninja`

`ninja` is a Python training API for local scripts that return rewards. OpenClaw still needs a provider-facing HTTP service, so the middleware stays in front. Inside the middleware, though, this example reuses Ninja's internal session client to hide the OpenForge `_openforge` trajectory metadata.
