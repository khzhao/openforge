# OpenClaw Replay Experiments

This folder is for scripted training/evaluation runs that simulate prompting the
OpenClaw bot without going through Telegram.

Available scripts:

- `gsm8k_replay.py`
- `gsm8k_runtime.yaml`

Shared helpers:

- `common.py`

`gsm8k_replay.py` is the simplest way to test whether the middleware plus
OpenForge training stack can improve on isolated one-turn homework-style
episodes. In `train` mode it injects deterministic `0/1` rewards directly into
OpenForge after each middleware-generated reply.

`gsm8k_runtime.yaml` is a small replay-oriented runtime config that keeps the
same model family but reduces rollout replicas and batch sizes enough for quick
end-to-end verification runs.

It uses the real `openai/gsm8k` dataset via the existing loader in
`examples/gsm8k/common.py`:

- `train` mode defaults to the `train` split
- `eval` mode defaults to the `test` split

There is no local toy dataset by default anymore.
