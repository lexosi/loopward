"""audit.py — structured, per-run audit trail.

Every run gets its own directory ``runs/<iso-timestamp>/`` containing:
  - ``audit.json``  machine-readable envelope (events + summary)
  - ``audit.md``    human-readable log

The audit log is append-only during a run via :meth:`AuditLog.record`, then
finalized once with :meth:`AuditLog.finalize`, which writes both files.

Design notes
------------
The envelope shape and the token/cost accounting are distilled from a
production DeepSeek wrapper, rewritten here with no project-specific paths.
Files are written UTF-8 without BOM (``Path.write_text(..., encoding="utf-8")``)
so they parse cleanly on every platform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def iso_now() -> str:
    """UTC timestamp, second precision, e.g. ``2026-05-29T16:34:01Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_for_dir() -> str:
    """Filesystem-safe timestamp for the run directory name."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class AuditEvent:
    """One recorded event in a run."""

    ts: str
    kind: str  # e.g. "phase", "attempt", "gate", "llm_call", "result"
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "kind": self.kind, "message": self.message, "data": self.data}


class AuditLog:
    """Collects events for a single run and writes the trail on finalize.

    Parameters
    ----------
    run_id:
        Logical name of the run (e.g. ``"code-review"``). Used in the summary.
    base_dir:
        Parent directory under which ``<timestamp>/`` is created. Defaults to
        ``./runs``. The directory is created lazily on :meth:`finalize`.
    """

    def __init__(self, run_id: str, base_dir: str | Path = "runs") -> None:
        self.run_id = run_id
        self.started_at = iso_now()
        self._base_dir = Path(base_dir)
        self._run_dir = self._base_dir / _ts_for_dir()
        self._events: list[AuditEvent] = []
        self._tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._cost_usd = 0.0

    @property
    def run_dir(self) -> Path:
        """Directory this run's files will be written to (created on finalize)."""
        return self._run_dir

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def record(self, kind: str, message: str, **data: Any) -> None:
        """Append an event. Keyword args are stored under ``data``."""
        self._events.append(AuditEvent(ts=iso_now(), kind=kind, message=message, data=dict(data)))

    def record_usage(self, prompt: int, completion: int, cost_usd: float) -> None:
        """Accumulate token usage and cost across LLM calls."""
        self._tokens["prompt"] += prompt
        self._tokens["completion"] += completion
        self._tokens["total"] += prompt + completion
        self._cost_usd += cost_usd

    def summary(self, status: str, result: Any) -> dict[str, Any]:
        """Build the run summary envelope (without writing it)."""
        return {
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "ended_at": iso_now(),
            "tokens": dict(self._tokens),
            "cost_usd": round(self._cost_usd, 6),
            "event_count": len(self._events),
            "result": result,
        }

    def finalize(self, status: str, result: Any) -> Path:
        """Write ``audit.json`` and ``audit.md`` to the run directory.

        Returns the run directory path.
        """
        self._run_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary(status, result)
        envelope = {"summary": summary, "events": [e.to_dict() for e in self._events]}

        (self._run_dir / "audit.json").write_text(
            json.dumps(envelope, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        (self._run_dir / "audit.md").write_text(self._render_markdown(summary), encoding="utf-8")
        return self._run_dir

    def _render_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            f"# Audit — {summary['run_id']}",
            "",
            f"- **status**: {summary['status']}",
            f"- **started**: {summary['started_at']}",
            f"- **ended**: {summary['ended_at']}",
            f"- **tokens**: {summary['tokens']['total']} "
            f"(prompt {summary['tokens']['prompt']}, completion {summary['tokens']['completion']})",
            f"- **cost_usd**: {summary['cost_usd']:.6f}",
            "",
            "## Events",
            "",
            "| time | kind | message |",
            "|------|------|---------|",
        ]
        for e in self._events:
            msg = e.message.replace("|", "\\|")
            lines.append(f"| {e.ts} | {e.kind} | {msg} |")
        lines.append("")
        return "\n".join(lines)
