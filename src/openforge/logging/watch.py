# Copyright 2026 openforge

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any

__all__ = ["render_status", "render_watch_error"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def render_status(
    payload: dict[str, Any], *, use_color: bool = False, width: int = 100
) -> str:
    """Render the status of a session."""
    session_id = payload.get("session_id") or "-"
    wall_time_s = float(payload.get("wall_time_s", 0.0))
    gateway = payload.get("gateway", {})
    train = payload.get("train", {})
    rollout = payload.get("rollout", {})
    cluster = payload.get("cluster", {})
    latest_update = train.get("latest_update") or {}
    usable_width = max(76, width)
    title = _style("OPENFORGE WATCH", "1;36", use_color)
    train_active = bool(train.get("active", False))
    stale_workers = _coerce_int(rollout.get("stale_worker_count"))
    badges = [
        _badge(f"session {session_id}", "1;97", use_color),
        _badge(f"wall {wall_time_s:.1f}s", "1;94", use_color),
        _badge(
            "train active" if train_active else "train idle",
            "1;32" if train_active else "1;33",
            use_color,
        ),
    ]
    if stale_workers is not None and stale_workers > 0:
        badges.append(_badge(f"stale workers {stale_workers}", "1;31", use_color))
    lines = [title, "  ".join(badges), ""]

    total_resources = cluster.get("total_resources", {})
    available_resources = cluster.get("available_resources", {})
    sections = [
        _render_panel(
            "Gateway",
            [
                _kv("status", _style("live", "1;32", use_color), use_color),
                _kv(
                    "last activity",
                    _age_text(gateway.get("heartbeat_age_s")),
                    use_color,
                ),
                _kv(
                    "pending generates",
                    _format_int(gateway.get("pending_generate_count")),
                    use_color,
                ),
            ],
            width=_panel_width(usable_width),
            use_color=use_color,
        ),
        _render_panel(
            "Train",
            [
                _kv("active", _bool_text(train_active, use_color), use_color),
                _kv("heartbeat", _age_text(train.get("heartbeat_age_s")), use_color),
                _kv(
                    "last update", _age_text(train.get("last_update_age_s")), use_color
                ),
                _kv("global step", _format_int(train.get("global_step")), use_color),
                _kv(
                    "policy version",
                    _format_int(train.get("policy_version")),
                    use_color,
                ),
                _kv(
                    "reward mean",
                    _format_float(latest_update.get("reward_mean")),
                    use_color,
                ),
                _kv(
                    "grad norm",
                    _format_float(latest_update.get("grad_norm")),
                    use_color,
                ),
                _kv("lr", _format_float(latest_update.get("lr")), use_color),
            ],
            width=_panel_width(usable_width),
            use_color=use_color,
        ),
        _render_panel(
            "Rollout",
            [
                _kv("heartbeat", _age_text(rollout.get("heartbeat_age_s")), use_color),
                _kv(
                    "published version",
                    _format_int(rollout.get("latest_published_train_version")),
                    use_color,
                ),
                _kv(
                    "min weight version",
                    _format_int(rollout.get("min_weight_version")),
                    use_color,
                ),
                _kv(
                    "max weight version",
                    _format_int(rollout.get("max_weight_version")),
                    use_color,
                ),
                _kv(
                    "max version skew",
                    _format_int(rollout.get("max_version_skew")),
                    use_color,
                ),
                _kv(
                    "stale workers",
                    _format_int(rollout.get("stale_worker_count")),
                    use_color,
                ),
            ],
            width=_panel_width(usable_width),
            use_color=use_color,
        ),
    ]
    if isinstance(total_resources, dict) and isinstance(available_resources, dict):
        sections.append(
            _render_panel(
                "Cluster",
                [
                    _kv(
                        "alive nodes",
                        _format_int(cluster.get("alive_nodes")),
                        use_color,
                    ),
                    _kv(
                        "CPU free/total",
                        _resource_pair(
                            available_resources.get("CPU"), total_resources.get("CPU")
                        ),
                        use_color,
                    ),
                    _kv(
                        "GPU free/total",
                        _resource_pair(
                            available_resources.get("GPU"), total_resources.get("GPU")
                        ),
                        use_color,
                    ),
                ],
                width=_panel_width(usable_width),
                use_color=use_color,
            )
        )

    lines.extend(_join_panels(sections[:2], usable_width))
    if len(sections) > 2:
        lines.append("")
        lines.extend(_join_panels(sections[2:], usable_width))

    workers = rollout.get("workers", {})
    if isinstance(workers, dict) and workers:
        lines.append("")
        worker_rows = [
            _worker_row(name, worker, width=usable_width - 4, use_color=use_color)
            for name, worker in sorted(workers.items())
        ]
        lines.extend(
            _render_panel(
                "Workers",
                worker_rows,
                width=usable_width,
                use_color=use_color,
            )
        )

    return "\n".join(lines)


def render_watch_error(
    message: str, *, use_color: bool = False, width: int = 100
) -> str:
    """Render an errored watch state."""
    return "\n".join(
        _render_panel(
            "Watch Error",
            [
                _kv("state", _style("errored", "1;31", use_color), use_color),
                _kv("error", message, use_color),
                "waiting for next refresh attempt...",
            ],
            width=max(76, width),
            use_color=use_color,
        )
    )


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _age_text(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}s"


def _resource_pair(available: object, total: object) -> str:
    return f"{_format_float(available)}/{_format_float(total)}"


def _bool_text(value: bool, use_color: bool) -> str:
    if value:
        return _style("yes", "1;32", use_color)
    return _style("no", "1;33", use_color)


def _style(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"\033[{code}m{text}\033[0m"


def _badge(text: str, code: str, use_color: bool) -> str:
    return _style(f"[ {text} ]", code, use_color)


def _panel_width(total_width: int) -> int:
    return max(38, (total_width - 3) // 2)


def _kv(label: str, value: str, use_color: bool) -> str:
    padded_label = _style(label.upper().ljust(18), "2", use_color)
    return f"{padded_label} {value}"


def _render_panel(
    title: str, rows: list[str], *, width: int, use_color: bool
) -> list[str]:
    inner_width = max(20, width - 4)
    panel_title = _style(title.upper(), "1;36", use_color)
    border = "+" + "-" * (width - 2) + "+"
    lines = [border, f"| {_pad_visible(panel_title, inner_width)} |"]
    lines.append("|" + "-" * (width - 2) + "|")
    for row in rows:
        lines.append(
            f"| {_pad_visible(_truncate_visible(row, inner_width), inner_width)} |"
        )
    lines.append(border)
    return lines


def _join_panels(panels: list[list[str]], total_width: int) -> list[str]:
    if not panels:
        return []
    if len(panels) == 1 or total_width < 100:
        lines: list[str] = []
        for index, panel in enumerate(panels):
            if index > 0:
                lines.append("")
            lines.extend(panel)
        return lines
    left = panels[0]
    right = panels[1]
    combined = [
        f"{left_line}   {right_line}"
        for left_line, right_line in zip_longest(left, right, fillvalue="")
    ]
    if len(panels) > 2:
        combined.append("")
        combined.extend(_join_panels(panels[2:], total_width))
    return combined


def _worker_row(
    worker_name: str, worker: dict[str, Any], *, width: int, use_color: bool
) -> str:
    healthy = bool(worker.get("healthy", False))
    state = worker.get("state", "-")
    parts = [
        _style(worker_name, "1;97", use_color),
        f"state={state}",
        f"healthy={_bool_text(healthy, use_color)}",
        f"weight={_format_int(worker.get('weight_version'))}",
        f"active_traj={_format_int(worker.get('active_trajectory_count'))}",
    ]
    return _truncate_visible("  ".join(parts), width)


def _truncate_visible(value: str, width: int) -> str:
    visible = _visible_len(value)
    if visible <= width:
        return value
    if width <= 3:
        return _slice_visible(value, width)
    return f"{_slice_visible(value, width - 3)}..."


def _pad_visible(value: str, width: int) -> str:
    visible = _visible_len(value)
    if visible >= width:
        return value
    return value + (" " * (width - visible))


def _visible_len(value: str) -> int:
    return len(_ANSI_RE.sub("", value))


def _slice_visible(value: str, width: int) -> str:
    if width <= 0:
        return ""
    parts: list[str] = []
    visible = 0
    index = 0
    while index < len(value) and visible < width:
        match = _ANSI_RE.match(value, index)
        if match is not None:
            parts.append(match.group(0))
            index = match.end()
            continue
        parts.append(value[index])
        visible += 1
        index += 1
    return "".join(parts)
