# RL Training Headers Extension

This OpenClaw plugin injects the headers expected by the OpenForge OpenClaw example:

- `X-Session-Id`
- `X-Turn-Type`

It follows the OpenClaw plugin model:

- plugin manifest: `openclaw.plugin.json`
- runtime entrypoint: `index.ts`
- typed lifecycle hooks via `api.on(...)`

Docs used for this shape:

- https://docs.openclaw.ai/plugins
- https://docs.openclaw.ai/tools/plugin

## Behavior

- On `before_prompt_build`, the plugin captures the current session id.
- It classifies the turn as:
  - `main` for normal user turns
  - `side` for `heartbeat`, `memory`, and `cron`
- It monkey-patches `globalThis.fetch` and injects those headers into matching POST requests.
- On `agent_end`, it clears the active header state.

## Config

Optional plugin config:

```json
{
  "plugins": {
    "entries": {
      "rl-training-headers": {
        "enabled": true,
        "config": {
          "urlIncludes": [
            "127.0.0.1:8012/v1/chat/completions",
            "localhost:8012/v1/chat/completions"
          ]
        }
      }
    }
  }
}
```

`urlIncludes` limits header injection to matching request URLs.
