# Copyright 2026 openforge

from __future__ import annotations

from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Static


def run_textual_watch(
    *,
    fetch_snapshot: Callable[[], tuple[bool, dict[str, object] | str]],
    interval: float,
) -> int:
    """Run the interactive watch dashboard."""
    app = OpenForgeWatchApp(fetch_snapshot=fetch_snapshot, interval=interval)
    app.run()
    return 0


class OpenForgeWatchApp(App[None]):
    """Interactive Textual dashboard for `openforge watch`."""

    CSS = """
    Screen {
        background: #06141a;
        color: #e8f1f5;
    }

    #body {
        layout: vertical;
        padding: 1 2;
    }

    #hero {
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        border: round #29a3c6;
        background: #0b232c;
    }

    .metrics-row {
        layout: horizontal;
        height: auto;
        margin: 0 0 1 0;
    }

    .card {
        width: 1fr;
        min-height: 10;
        padding: 1 2;
        border: round #1b6f8c;
        background: #0a1c24;
    }

    #gateway {
        margin: 0 1 0 0;
    }

    #rollout {
        margin: 0 1 0 0;
    }

    #workers {
        height: 1fr;
        padding: 1 2;
        border: round #1b6f8c;
        background: #08171d;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
    ]

    def __init__(
        self,
        *,
        fetch_snapshot: Callable[[], tuple[bool, dict[str, object] | str]],
        interval: float,
    ) -> None:
        super().__init__()
        self._fetch_snapshot = fetch_snapshot
        self._interval = interval

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            yield Static(id="hero")
            with Horizontal(classes="metrics-row"):
                yield Static(id="gateway", classes="card")
                yield Static(id="train", classes="card")
            with Horizontal(classes="metrics-row"):
                yield Static(id="rollout", classes="card")
                yield Static(id="cluster", classes="card")
            yield Static(id="workers")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OpenForge Watch"
        self.sub_title = "Live session dashboard"
        self._refresh_view()
        self.set_interval(self._interval, self._refresh_view)

    def action_refresh_now(self) -> None:
        self._refresh_view()

    def _refresh_view(self) -> None:
        ok, payload = self._fetch_snapshot()
        if ok:
            assert isinstance(payload, dict)
            self._render_status(payload)
            return
        self._render_error(str(payload))

    def _render_status(self, payload: dict[str, object]) -> None:
        session_id = str(payload.get("session_id") or "-")
        wall_time_s = float(payload.get("wall_time_s", 0.0))
        gateway = _as_dict(payload.get("gateway"))
        train = _as_dict(payload.get("train"))
        rollout = _as_dict(payload.get("rollout"))
        cluster = _as_dict(payload.get("cluster"))
        latest_update = _as_dict(train.get("latest_update"))

        train_active = bool(train.get("active", False))
        self.sub_title = f"{session_id} | wall {wall_time_s:.1f}s"
        hero_lines = [
            "[b cyan]OpenForge Watch[/b cyan]",
            (
                f"[bold]session[/bold] {_escape(session_id)}    "
                f"[bold]wall[/bold] {wall_time_s:.1f}s    "
                f"{_badge('GATEWAY LIVE', 'black on #56d364')}    "
                f"{_badge('TRAIN ACTIVE', 'black on #3fb950') if train_active else _badge('TRAIN IDLE', 'black on #f2cc60')}"
            ),
        ]
        stale_workers = rollout.get("stale_worker_count")
        if stale_workers is not None and int(stale_workers) > 0:
            hero_lines.append(
                _badge(f"STALE WORKERS {int(stale_workers)}", "black on #ff6b6b")
            )
        self._widget("hero").update("\n".join(hero_lines))

        self._widget("gateway").update(
            _card_markup(
                "Gateway",
                [
                    ("Status", "live", "bold #56d364"),
                    (
                        "Last activity",
                        _age_text(gateway.get("heartbeat_age_s")),
                        _age_markup(gateway.get("heartbeat_age_s")),
                    ),
                    (
                        "Pending generates",
                        _format_int(gateway.get("pending_generate_count")),
                        "bold white",
                    ),
                ],
            )
        )
        self._widget("train").update(
            _card_markup(
                "Train",
                [
                    (
                        "Active",
                        "yes" if train_active else "no",
                        _bool_markup(train_active),
                    ),
                    (
                        "Heartbeat",
                        _age_text(train.get("heartbeat_age_s")),
                        _age_markup(train.get("heartbeat_age_s")),
                    ),
                    (
                        "Last update",
                        _age_text(train.get("last_update_age_s")),
                        _age_markup(train.get("last_update_age_s")),
                    ),
                    (
                        "Global step",
                        _format_int(train.get("global_step")),
                        "bold white",
                    ),
                    (
                        "Policy version",
                        _format_int(train.get("policy_version")),
                        "bold white",
                    ),
                    (
                        "Reward mean",
                        _format_float(latest_update.get("reward_mean")),
                        _reward_markup(latest_update.get("reward_mean")),
                    ),
                    (
                        "Grad norm",
                        _format_float(latest_update.get("grad_norm")),
                        "bold white",
                    ),
                    ("LR", _format_float(latest_update.get("lr")), "bold white"),
                ],
            )
        )
        self._widget("rollout").update(
            _card_markup(
                "Rollout",
                [
                    (
                        "Heartbeat",
                        _age_text(rollout.get("heartbeat_age_s")),
                        _age_markup(rollout.get("heartbeat_age_s")),
                    ),
                    (
                        "Published version",
                        _format_int(rollout.get("latest_published_train_version")),
                        "bold white",
                    ),
                    (
                        "Min weight version",
                        _format_int(rollout.get("min_weight_version")),
                        "bold white",
                    ),
                    (
                        "Max weight version",
                        _format_int(rollout.get("max_weight_version")),
                        "bold white",
                    ),
                    (
                        "Version skew",
                        _format_int(rollout.get("max_version_skew")),
                        _skew_markup(rollout.get("max_version_skew")),
                    ),
                    (
                        "Stale workers",
                        _format_int(rollout.get("stale_worker_count")),
                        _stale_markup(rollout.get("stale_worker_count")),
                    ),
                ],
            )
        )

        total_resources = _as_dict(cluster.get("total_resources"))
        available_resources = _as_dict(cluster.get("available_resources"))
        cluster_rows = [
            ("Alive nodes", _format_int(cluster.get("alive_nodes")), "bold white")
        ]
        if total_resources and available_resources:
            cluster_rows.extend(
                [
                    (
                        "CPU free / total",
                        _resource_pair(
                            available_resources.get("CPU"), total_resources.get("CPU")
                        ),
                        "bold white",
                    ),
                    (
                        "GPU free / total",
                        _resource_pair(
                            available_resources.get("GPU"), total_resources.get("GPU")
                        ),
                        "bold white",
                    ),
                ]
            )
        else:
            cluster_rows.append(
                ("Status", "no cluster telemetry yet", "italic #8aa1ad")
            )
        self._widget("cluster").update(_card_markup("Cluster", cluster_rows))

        workers = _as_dict(rollout.get("workers"))
        worker_lines = ["[b cyan]Workers[/b cyan]"]
        if workers:
            for worker_name, worker in sorted(workers.items()):
                worker_dict = _as_dict(worker)
                worker_lines.append(
                    (
                        f"[bold white]{_escape(str(worker_name))}[/bold white]  "
                        f"state={_escape(str(worker_dict.get('state', '-')))}  "
                        f"healthy=[{_bool_markup(bool(worker_dict.get('healthy', False)))}]"
                        f"{'yes' if bool(worker_dict.get('healthy', False)) else 'no'}[/]  "
                        f"weight={_format_int(worker_dict.get('weight_version'))}  "
                        f"active_traj={_format_int(worker_dict.get('active_trajectory_count'))}"
                    )
                )
        else:
            worker_lines.append(
                "[italic #8aa1ad]No rollout workers reported yet.[/italic #8aa1ad]"
            )
        self._widget("workers").update("\n".join(worker_lines))

    def _render_error(self, message: str) -> None:
        self.sub_title = "waiting for next refresh"
        self._widget("hero").update(
            "\n".join(
                [
                    "[b red]Watch Error[/b red]",
                    "[bold]state[/bold] errored",
                    f"[bold]error[/bold] {_escape(message)}",
                    "[italic #8aa1ad]waiting for next refresh attempt...[/italic #8aa1ad]",
                ]
            )
        )

    def _widget(self, widget_id: str) -> Static:
        return self.query_one(f"#{widget_id}", Static)


def _card_markup(title: str, rows: list[tuple[str, str, str]]) -> str:
    lines = [f"[b cyan]{_escape(title)}[/b cyan]"]
    for label, value, style in rows:
        lines.append(f"[dim]{_escape(label)}[/dim] [{style}]{_escape(value)}[/]")
    return "\n".join(lines)


def _badge(text: str, style: str) -> str:
    return f"[{style}] {_escape(text)} [/]"


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _resource_pair(available: object, total: object) -> str:
    return f"{_format_float(available)}/{_format_float(total)}"


def _age_text(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}s"


def _age_markup(value: object) -> str:
    if value is None:
        return "bold white"
    age_s = float(value)
    if age_s <= 1.0:
        return "bold #3fb950"
    if age_s <= 5.0:
        return "bold #f2cc60"
    return "bold #ff6b6b"


def _bool_markup(value: bool) -> str:
    return "bold #3fb950" if value else "bold #f2cc60"


def _reward_markup(value: object) -> str:
    if value is None:
        return "bold white"
    reward = float(value)
    if reward > 0:
        return "bold #3fb950"
    if reward < 0:
        return "bold #ff6b6b"
    return "bold white"


def _stale_markup(value: object) -> str:
    if value is None:
        return "bold white"
    return "bold #ff6b6b" if int(value) > 0 else "bold #3fb950"


def _skew_markup(value: object) -> str:
    if value is None:
        return "bold white"
    skew = int(value)
    if skew <= 1:
        return "bold #3fb950"
    if skew <= 3:
        return "bold #f2cc60"
    return "bold #ff6b6b"


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _escape(value: str) -> str:
    return value.replace("[", "[[").replace("]", "]]")
