# Copyright 2026 openforge

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Static

from openforge import active_state

from .session_state import SessionStateStore, TurnRecord

__all__ = ["main"]

_DEFAULT_GATEWAY_YAML = Path(__file__).with_name("gateway.yaml")


@dataclass(slots=True)
class _DashboardData:
    status: dict[str, Any] | None
    pending: list[TurnRecord]
    recent: list[TurnRecord]
    traj_summary: dict[str, Any]
    gateway_base_url: str
    gateway_db: Path
    state_db: Path
    rendered_at: float


class OpenClawDemoApp(App[None]):
    CSS = """
    Screen {
      background: #08111d;
      color: #f4f3ea;
    }

    #root {
      layout: vertical;
      padding: 1;
    }

    .row {
      height: auto;
      margin-bottom: 1;
    }

    .panel {
      width: 1fr;
      min-height: 12;
      margin-right: 1;
      border: round #2b5f75;
      background: #0f1b2b;
      padding: 1;
    }

    .panel:last-of-type {
      margin-right: 0;
    }

    #records {
      height: 1fr;
      border: round #8c5e34;
      background: #121f31;
      padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("x", "clear_databases", "Clear DBs"),
    ]

    def __init__(
        self,
        *,
        store: SessionStateStore,
        gateway_db: Path,
        gateway_base_url: str,
        refresh_seconds: float,
        limit: int,
    ) -> None:
        super().__init__()
        self._store = store
        self._gateway_db = gateway_db
        self._gateway_base_url = gateway_base_url
        self._refresh_seconds = refresh_seconds
        self._limit = limit
        self._last_fingerprint: str | None = None
        self._pending_clear_confirmation = False
        self._busy = False
        self._busy_message = ""
        self._busy_frame = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="root"):
            with Horizontal(classes="row"):
                yield Static(id="status", classes="panel")
                yield Static(id="trajectory", classes="panel")
            with Horizontal(classes="row"):
                yield Static(id="pending", classes="panel")
                yield Static(id="conversation", classes="panel")
            yield Static(id="records")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OpenClaw Demo Dashboard"
        self.sub_title = "OpenForge + Middleware + Rewards"
        self.set_interval(self._refresh_seconds, self._refresh)
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh(force=True)

    def action_clear_databases(self) -> None:
        self._pending_clear_confirmation = True
        self.notify("Press y to confirm clearing both DBs, or n to cancel.")

    def on_key(self, event) -> None:
        if not self._pending_clear_confirmation:
            return
        key = str(event.key).lower()
        if key == "y":
            self._pending_clear_confirmation = False
            self._busy = True
            self._busy_message = "Clearing middleware state DB and gateway DB..."
            self._busy_frame = 0
            self.query_one("#records", Static).update(
                _working_panel(self._busy_message, self._busy_frame)
            )
            self.call_after_refresh(self._clear_databases_now)
            event.stop()
        elif key in {"n", "escape"}:
            self._pending_clear_confirmation = False
            self.notify("Clear cancelled.")
            event.stop()

    def _clear_databases_now(self) -> None:
        self._store.clear_all()
        _clear_gateway_db(self._gateway_db)
        self._busy = False
        self._busy_message = ""
        self.notify("Cleared middleware state DB and gateway DB rows.")
        self._refresh(force=True)

    def _refresh(self, force: bool = False) -> None:
        data = _collect_dashboard_data(
            store=self._store,
            gateway_db=self._gateway_db,
            gateway_base_url=self._gateway_base_url,
            limit=self._limit,
        )
        fingerprint = _fingerprint(data)
        if not force and fingerprint == self._last_fingerprint:
            return
        self._last_fingerprint = fingerprint
        self.query_one("#status", Static).update(_status_panel(data))
        self.query_one("#trajectory", Static).update(_trajectory_panel(data))
        self.query_one("#pending", Static).update(_pending_panel(data))
        self.query_one("#conversation", Static).update(_conversation_panel(data))
        if self._busy:
            self._busy_frame += 1
            self.query_one("#records", Static).update(
                _working_panel(self._busy_message, self._busy_frame)
            )
        else:
            self.query_one("#records", Static).update(_records_panel(data))


def main() -> int:
    """Run the OpenClaw / OpenForge dashboard."""
    parser = argparse.ArgumentParser(description="OpenClaw / OpenForge visibility TUI")
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--state-db", default=None)
    parser.add_argument("--gateway-yaml", default=None)
    parser.add_argument("--gateway-db", default=None)
    parser.add_argument("--gateway-base-url", default=None)
    args = parser.parse_args()

    if args.state_db is None:
        raise SystemExit("--state-db is required")
    if args.gateway_db is None and args.gateway_yaml is None:
        raise SystemExit("either --gateway-db or --gateway-yaml is required")

    state_db = Path(args.state_db).expanduser()
    gateway_db = (
        Path(args.gateway_db).expanduser()
        if args.gateway_db is not None
        else _gateway_db_path(Path(str(args.gateway_yaml)).expanduser())
    )
    gateway_base_url = _resolve_gateway_base_url(args.gateway_base_url)
    store = SessionStateStore(state_db)
    app = OpenClawDemoApp(
        store=store,
        gateway_db=gateway_db,
        gateway_base_url=gateway_base_url,
        refresh_seconds=float(args.refresh_seconds),
        limit=int(args.limit),
    )
    app.run()
    return 0


def _collect_dashboard_data(
    *,
    store: SessionStateStore,
    gateway_db: Path,
    gateway_base_url: str,
    limit: int,
) -> _DashboardData:
    return _DashboardData(
        status=_fetch_gateway_status(gateway_base_url),
        pending=store.list_pending_turn_records(limit=limit),
        recent=store.list_turn_records(limit=limit),
        traj_summary=_trajectory_summary(gateway_db),
        gateway_base_url=gateway_base_url,
        gateway_db=gateway_db,
        state_db=store._path,
        rendered_at=time.time(),
    )


def _resolve_gateway_base_url(explicit: str | None) -> str:
    if explicit:
        return str(explicit).rstrip("/")
    host, port = active_state.load_active_gateway_target()
    return f"http://{host}:{port}"


def _status_panel(data: _DashboardData):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold #8bd3ff")
    table.add_column(style="#f4f3ea")
    status = data.status
    if status is None:
        table.add_row("Status", "unreachable")
        table.add_row("Hint", "Start the OpenForge gateway/session first")
    else:
        train = status.get("train", {})
        rollout = status.get("rollout", {})
        latest_update = train.get("latest_update") or {}
        table.add_row("Gateway", data.gateway_base_url)
        table.add_row("Train active", str(train.get("active")))
        table.add_row("Policy version", str(train.get("policy_version")))
        table.add_row("Global step", str(train.get("global_step")))
        table.add_row("Reward mean", str(latest_update.get("reward_mean")))
        table.add_row("Rollout weight", f"{rollout.get('min_weight_version')}..{rollout.get('max_weight_version')}")
    return Panel(table, title="OpenForge Status", border_style="#2b8a99")


def _trajectory_panel(data: _DashboardData):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold #f7c873")
    table.add_column(style="#f4f3ea")
    counts = data.traj_summary.get("counts", {})
    total = sum(counts.values())
    table.add_row("Gateway DB", str(data.gateway_db))
    table.add_row("State DB", str(data.state_db))
    table.add_row("Total trajectories", str(total))
    for status, count in sorted(counts.items()):
        table.add_row(status, str(count))
    return Panel(table, title="Trajectory Summary", border_style="#c0843d")


def _pending_panel(data: _DashboardData):
    lines = Text()
    if not data.pending:
        lines.append("No pending train turns.\n", style="italic #8aa0b8")
    else:
        now = time.time()
        for record in data.pending[:4]:
            age = now - record.created_at
            request_role, request_text = _latest_visible_request_message(
                record.normalized_messages
            )
            lines.append(f"{record.external_session_id}\n", style="bold #8bd3ff")
            lines.append(
                f"  pending {record.trajectory_id[:12]}  age={age:4.1f}s\n",
                style="#f7c873",
            )
            lines.append(
                "  awaiting next user/tool message to score this turn\n",
                style="italic #8aa0b8",
            )
            if request_text:
                lines.append(
                    f"  {request_role}: {_trim(request_text, 82)}\n",
                    style="#8bd3ff" if request_role == "user" else "#f7c873",
                )
            lines.append(
                f"  assistant: {_trim(record.assistant_text, 68)}\n",
                style="#f4f3ea",
            )
    rewarded = [record for record in data.recent if record.reward is not None][:4]
    lines.append("\nRecent Rewards\n", style="bold #7be0a2")
    if not rewarded:
        lines.append("  none\n", style="italic #8aa0b8")
    for record in rewarded:
        lines.append(
            f"  {record.reward:+.2f}  {record.external_session_id}  {_trim(record.reason or '-', 34)}\n",
            style="#f4f3ea",
        )
        if record.feedback_text:
            lines.append(
                f"    judged by next state: {_trim(record.feedback_text, 68)}\n",
                style="#8aa0b8",
            )
    return Panel(lines, title="Pending / Rewards", border_style="#5ba36b")


def _conversation_panel(data: _DashboardData):
    record = _focus_session_record(data)
    lines = Text()
    if record is None:
        lines.append("No conversation recorded yet.", style="italic #8aa0b8")
    else:
        session_records = [
            item
            for item in reversed(data.recent)
            if item.external_session_id == record.external_session_id
            and item.purpose == "train"
        ]
        max_turns = 4
        truncated = len(session_records) > max_turns
        visible_records = session_records[-max_turns:]
        lines.append(
            f"session={record.external_session_id}  status={record.status}\n\n",
            style="bold #f4f3ea",
        )
        if truncated:
            lines.append("… older turns omitted …\n\n", style="italic #8aa0b8")
        start_index = len(session_records) - len(visible_records) + 1
        for turn_index, item in enumerate(visible_records, start=start_index):
            request_role, request_text = _latest_visible_request_message(
                item.normalized_messages
            )
            lines.append(
                f"turn {turn_index}  {item.trajectory_id[:12]}  {item.status}\n",
                style="bold #f7c873",
            )
            if request_text:
                lines.append(
                    f"{request_role}: ",
                    style="bold #8bd3ff" if request_role == "user" else "bold #f7c873",
                )
                lines.append(f"{_trim(request_text, 110)}\n", style="#f4f3ea")
            lines.append("assistant: ", style="bold #7be0a2")
            lines.append(f"{_trim(item.assistant_text, 110)}\n", style="#f4f3ea")
            if item.reward is not None:
                if item.feedback_text:
                    lines.append("next state: ", style="bold #f7c873")
                    lines.append(
                        f"{_trim(item.feedback_text, 110)}\n",
                        style="#f4f3ea",
                    )
                lines.append("reward: ", style="bold #ff8f8f")
                lines.append(
                    f"{item.reward:+.2f} ({item.reason or '-'})\n",
                    style="#f4f3ea",
                )
            elif item.status == "pending":
                lines.append(
                    "waiting for next user/tool message to score this turn\n",
                    style="italic #8aa0b8",
                )
            lines.append("\n", style="#f4f3ea")
        if record.feedback_text:
            lines.append("feedback: ", style="bold #f7c873")
            lines.append(f"{_trim(record.feedback_text, 110)}\n", style="#f4f3ea")
    return Panel(lines, title="Latest Conversation", border_style="#8a4fff")


def _records_panel(data: _DashboardData):
    table = Table(expand=True, box=None, show_header=True, header_style="bold #8bd3ff")
    table.add_column("Status", width=10)
    table.add_column("Type", width=6)
    table.add_column("Input", width=6)
    table.add_column("Session", width=18)
    table.add_column("Reward", width=8, justify="right")
    table.add_column("Reason", width=28)
    table.add_column("Trajectory", width=14)
    for record in data.recent[:10]:
        reward = "-" if record.reward is None else f"{record.reward:+.2f}"
        request_role, _request_text = _latest_visible_request_message(
            record.normalized_messages
        )
        style = "#f4f3ea"
        if record.status == "pending":
            style = "#f7c873"
        elif record.status == "rewarded":
            style = "#7be0a2"
        elif record.status in {"failed", "discarded"}:
            style = "#ff8f8f"
        table.add_row(
            record.status,
            record.turn_type,
            request_role,
            record.external_session_id[:18],
            reward,
            _trim(record.reason or "-", 28),
            record.trajectory_id[:14],
            style=style,
        )
    footer = Text()
    footer.append("Press ", style="#8aa0b8")
    footer.append("r", style="bold #8bd3ff")
    footer.append(" to refresh, ", style="#8aa0b8")
    footer.append("x", style="bold #8bd3ff")
    footer.append(" then y to clear DBs, ", style="#8aa0b8")
    footer.append("q", style="bold #8bd3ff")
    footer.append(" to quit.", style="#8aa0b8")
    return Panel(table, title="Records", subtitle=footer, border_style="#8c5e34")


def _working_panel(message: str, frame: int):
    glyphs = ["◐", "◓", "◑", "◒"]
    glyph = glyphs[frame % len(glyphs)]
    return Panel(
        Text(f"{glyph}  {message}", style="bold #f7c873"),
        title="Working",
        border_style="#f7c873",
    )


def _fetch_gateway_status(base_url: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/status")
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _trajectory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with sqlite3.connect(path) as conn:
        counts_rows = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM trajectories
            GROUP BY status
            """
        ).fetchall()
        recent_rows = conn.execute(
            """
            SELECT trajectory_id, status, group_id, final_reward
            FROM trajectories
            ORDER BY rowid DESC
            LIMIT 12
            """
        ).fetchall()
    return {
        "counts": {str(status): int(count) for status, count in counts_rows},
        "recent": [
            {
                "trajectory_id": str(row[0]),
                "status": str(row[1]),
                "group_id": None if row[2] is None else str(row[2]),
                "final_reward": None if row[3] is None else float(row[3]),
            }
            for row in recent_rows
        ],
    }


def _clear_gateway_db(path: Path) -> None:
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            if not str(row[0]).startswith("sqlite_")
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()


def _gateway_db_path(gateway_yaml: Path) -> Path:
    payload = yaml.safe_load(gateway_yaml.read_text(encoding="utf-8")) or {}
    data = payload.get("data", {})
    path = data.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"gateway yaml {gateway_yaml} does not define data.path")
    return Path(path).expanduser()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _latest_visible_request_message(messages: list[dict[str, Any]]) -> tuple[str, str]:
    for message in reversed(messages):
        role = str(message.get("role"))
        if role not in {"user", "tool"}:
            continue
        text = (
            _display_user_message(_message_text(message)).strip()
            if role == "user"
            else _message_text(message).strip()
        )
        if text:
            return role, text
    return "?", ""


def _focus_session_record(data: _DashboardData) -> TurnRecord | None:
    if data.pending:
        return data.pending[0]
    train_recent = [item for item in data.recent if item.purpose == "train"]
    if train_recent:
        return train_recent[0]
    return data.recent[0] if data.recent else None


def _display_user_message(text: str) -> str:
    """Strip OpenClaw-injected metadata and show the actual user utterance."""
    cleaned = re.sub(
        r"Conversation info \(untrusted metadata\):\s*```json.*?```",
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"Sender \(untrusted metadata\):\s*```json.*?```",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
    if not paragraphs:
        return text
    return paragraphs[-1]


def _fingerprint(data: _DashboardData) -> str:
    payload = {
        "status": data.status,
        "pending": [(item.trajectory_id, item.status) for item in data.pending],
        "recent": [
            (item.trajectory_id, item.status, item.reward, item.reason)
            for item in data.recent
        ],
        "counts": data.traj_summary.get("counts", {}),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _trim(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
