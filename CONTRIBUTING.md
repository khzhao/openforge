# Contributing to openforge

Contributions are welcome.

If you have ideas for new algorithms, runtime improvements, better developer ergonomics, or documentation fixes, feel free to open an issue or send a pull request.

## Good First Contribution Areas

Some areas where contributions would be especially useful:

- rollout/engine recovery and restart
- gateway back pressure and load balancing
- LoRA training support
- higher-throughput transport for gateway/runtime communication
- additional algorithms and rollout backends
- tests, docs, and observability

## Development Setup

Create a local development environment:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --dev
```

Install and enable pre-commit:

```bash
pre-commit install
```

## Code Quality

Before sending a PR, run:

```bash
pre-commit run --all-files --show-diff-on-failure --color=always
ruff format src tests
ruff check src tests
pyrefly check
pytest
```

If you are only touching a small area, targeted tests are fine while iterating. For example:

```bash
pytest tests/test_cli.py tests/test_ninja.py tests/test_gateway_service.py
```

## Pull Requests

Please keep PRs focused and easy to review.

- explain the motivation and behavior change clearly
- avoid bundling unrelated refactors with functional changes
- add or update tests when behavior changes
- update docs when public APIs, CLI behavior, or configuration changes
- call out runtime assumptions if the change depends on specific GPU, Ray, or networking behavior

For larger changes, opening an issue or discussion first is preferred.

## Style Notes

- follow the existing code style and project structure
- use pre-commit to keep formatting and linting consistent
- prefer small, incremental changes over large rewrites
- keep public APIs and config changes explicit and documented

## License

By contributing to openforge, you agree that your contributions will be licensed under the Apache 2.0 license used by this repository.
