"""audit.py — structured, per-run audit trail.

Every run gets its own directory ``runs/<iso-timestamp>/`` containing:
  - ``audit.json``  machine-readable envelope (events + summary)
  - ``audit.md``    human-readable log

The audit log is append-only during a run via :meth:`AuditLog.record`, then
finalized once with :meth:`AuditLog.finalize`, which writes both files.
``Orchestrator.run`` is the only caller of ``finalize``, on every exit path
including a crash, and calling it twice is refused. A write can still fail
part-way, leaving the directory reserved or only ``audit.json`` written, so
:attr:`AuditLog.run_dir_on_disk` reports whether this run created a directory
at all, rather than the name it wanted. It does not say which files landed
inside it.

The summary also carries ``provider_stop_reasons``: the raw reason each
provider gave for stopping, one per call, in order. It is written and never
read back by this package — recorded so a trail reader can tell a truncated or
refused answer from a complete one, which nothing upstream could do while the
value was being dropped inside ``llm_wrapper``. Acting on it is separate work.

Design notes
------------
The envelope shape and the token/cost accounting are distilled from a
production DeepSeek wrapper, rewritten here with no project-specific paths.
Files are written UTF-8 without BOM (``Path.write_text(..., encoding="utf-8")``)
so they parse cleanly on every platform.
"""

from __future__ import annotations

import json
import os
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


def _render_stop_reasons(reasons: list[str | None]) -> str:
    """Render the raw stop reasons for ``audit.md``. Not a summary of them.

    A missing reason renders as ``null`` — the same token ``audit.json`` uses --
    so the two files name the same thing the same way and neither invents a
    word for it. The order and the repeats are kept: collapsing them into
    counts would be this renderer deciding what the reader cares about.
    """
    if not reasons:
        return "(none recorded)"
    return ", ".join("null" if r is None else str(r) for r in reasons)


def _render_cost(cost: float | None) -> str:
    """Render ``cost_usd`` for ``audit.md``. ``null`` — the token ``audit.json``
    uses and the same one ``_render_stop_reasons`` uses for a missing value — when
    the run could not be priced, so the markdown never shows a misleading
    ``0.000000`` for an uncalculable cost."""
    return "null" if cost is None else f"{cost:.6f}"


def _render_cost_basis(basis: dict[str, Any] | None) -> str:
    """Render the cost basis line for ``audit.md``.

    Carries the same facts the JSON ``cost_basis`` block holds — that the figures
    are a ``len//4`` estimate, whether they were priced, and the date the rates
    were verified — so the markdown, read on its own, distinguishes an estimate
    from a bill and shows the figure's age.
    """
    if not basis:
        return "(not recorded)"
    priced = basis.get("priced")
    verified = basis.get("rates_verified")
    priced_phrase = "priced" if priced else "unpriced -> cost_usd null"
    verified_phrase = f"; rates verified {verified}" if verified else ""
    return f"estimated (len//4 tokens x published rate), {priced_phrase}{verified_phrase}"


#: How many ``<ts>-N`` fallbacks to try before giving up on a free run directory.
MAX_DIR_COLLISIONS = 100


class AuditDirectoryError(RuntimeError):
    """Every candidate run-directory name was already taken.

    Scoped narrowly on purpose. This is the name-collision exhaustion, which
    earns its own type because refusing to overwrite an existing trail is a
    decision worth naming. The other reasons a directory cannot be created —
    permissions, a full disk, a path under a file — are not caught here and
    surface as the ``OSError`` they already are.
    """


class AuditAlreadyFinalizedError(RuntimeError):
    """This log already wrote its trail; a second run tried to reuse it.

    A ``RuntimeError`` subclass so existing handlers keep working, and its own
    type so a caller can tell it apart from a write that failed. The difference
    decides what to tell the user: a failed write may have left something of
    *this* run on disk to go and look at; this one means the run wrote nothing
    and whatever is in the directory belongs to an earlier one.
    """


@dataclass
class AuditEvent:
    """One recorded event in a run."""

    ts: str
    # Every kind the engine emits. It is a comment, not a constraint: nothing
    # validates it and no test asserts the set, so it is only as true as the
    # last person to add an event. `review_parse` was emitted for several
    # commits before it was written down here.
    # phase | attempt | class_jump | review_parse | chunk_review | anti_loop |
    # gate | result
    kind: str
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

    The directory name has second precision, so two runs starting within the
    same second would collide. :meth:`finalize` claims the name atomically and
    falls back to ``<timestamp>-1``, ``<timestamp>-2``, ... rather than writing
    over an existing trail.
    """

    def __init__(self, run_id: str, base_dir: str | Path = "runs") -> None:
        self.run_id = run_id
        self.started_at = iso_now()
        self._base_dir = Path(base_dir)
        self._run_dir = self._base_dir / _ts_for_dir()
        self._dir_reserved = False
        self._finalized = False
        self._events: list[AuditEvent] = []
        self._tokens = {"prompt": 0, "completion": 0, "total": 0}
        #: ``None`` once any folded cost was unpriced — an uncalculable total is
        #: kept uncalculable, never rounded to ``0.0``.
        self._cost_usd: float | None = 0.0
        #: How the usage figures were produced, supplied by the usage source and
        #: echoed into the summary so the trail is honest on its own.
        self._cost_basis: dict[str, Any] | None = None
        self._stop_reasons: list[str | None] = []

    @property
    def run_dir(self) -> Path:
        """Directory this run's files are written to (claimed on finalize).

        Before :meth:`finalize` this is the preferred name; if another run has
        already taken it, finalize resolves to a suffixed one.
        """
        return self._run_dir

    @property
    def finalized(self) -> bool:
        """True once :meth:`finalize` has been entered, successfully or not.

        A failed finalize still counts: it may have left a reserved directory
        or a half-written pair of files, and re-running it would overwrite
        whatever did land. One run, one finalize.
        """
        return self._finalized

    @property
    def run_dir_on_disk(self) -> str:
        """The run directory **if this log created something**, else ``""``.

        Deliberately not the same thing as :attr:`run_dir`, which is the name
        this run *wants*. This one answers the only question a caller reporting
        a failure may ask: is there anything on disk to go and look at? Pointing
        a user at a path that was never created is the same class of untruth
        this package exists to prevent, so an empty string means empty-handed.

        Scope: it answers for *this* log. A caller that reuses one ``AuditLog``
        across two runs gets a path here for the second run as well, and it is
        the first run's trail — which is why ``Orchestrator`` does not ask this
        when the finalize it refused was :class:`AuditAlreadyFinalizedError`.
        """
        return str(self._run_dir) if self._dir_reserved else ""

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def record(self, kind: str, message: str, **data: Any) -> None:
        """Append an event. Keyword args are stored under ``data``."""
        self._events.append(AuditEvent(ts=iso_now(), kind=kind, message=message, data=dict(data)))

    def record_usage(
        self,
        prompt: int,
        completion: int,
        cost_usd: float | None,
        *,
        basis: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate token usage and cost across LLM calls.

        ``cost_usd`` is ``None`` when the run could not be priced (an unpriced
        (provider, model)); the accumulated cost then stays ``None`` — an
        uncalculable total is not silently turned into ``0.0``, the same
        distinction the summary and markdown then preserve as ``null``.

        ``basis`` describes how the figures were produced — a ``len//4`` estimate
        and a derived, unbilled cost — and is echoed into the summary so the
        trail is self-describing. The usage source supplies it; a caller that
        does not still gets a summary that reports ``priced`` from the cost.
        """
        self._tokens["prompt"] += prompt
        self._tokens["completion"] += completion
        self._tokens["total"] += prompt + completion
        # None is sticky, mirroring LLMClient.totals: one unpriced fold makes the
        # whole run's cost uncalculable.
        if cost_usd is None or self._cost_usd is None:
            self._cost_usd = None
        else:
            self._cost_usd += cost_usd
        if basis is not None:
            self._cost_basis = basis

    def record_provider_stop_reasons(self, reasons: list[str | None]) -> None:
        """Take the run's raw provider stop reasons, in call order.

        Named for the provider throughout — method, envelope key, and rendered
        label — because the trail already had a `stop_reason`: the `anti_loop`
        event's, which says why the *review loop* cut itself short. The two are
        unrelated and one of them is not going to be renamed, since tests and
        readers depend on it. Two different things under one word in one file is
        the ambiguity this package keeps closing elsewhere.

        A sibling of :meth:`record_usage`, and deliberately as dumb as it is:
        the strings are stored exactly as the provider sent them, nothing is
        counted, mapped, or de-duplicated here. The envelope reports them and
        no code in this package reads them back.

        Landing in the summary rather than as events is the point. An event per
        call would add one line per LLM call to every run's trail — and any rule
        for emitting *some* of them ("only the unusual ones") would first have to
        decide which values are unusual, which is the mapping decision this
        change exists to defer.
        """
        self._stop_reasons.extend(reasons)

    def summary(self, status: str, result: Any) -> dict[str, Any]:
        """Build the run summary envelope (without writing it)."""
        return {
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "ended_at": iso_now(),
            "tokens": dict(self._tokens),
            # `null`, not 0.0, when the run could not be priced — distinguishable
            # from a run that genuinely cost nothing. See `cost_basis`.
            "cost_usd": round(self._cost_usd, 6) if self._cost_usd is not None else None,
            "cost_basis": self._cost_basis_summary(),
            # Raw, ordered, unnormalised; `null` where the provider reported
            # nothing (always so for `fake`). Can be longer than the call count
            # — see `LLMClient.stop_reasons`.
            "provider_stop_reasons": list(self._stop_reasons),
            "event_count": len(self._events),
            "result": result,
        }

    def _cost_basis_summary(self) -> dict[str, Any]:
        """The cost basis as written to the summary.

        ``priced`` is authoritative here and reconciled to the emitted cost: it
        is true iff ``cost_usd`` is a number, so the two can never disagree inside
        one trail. The rest — the estimate/derivation flags, the note, the
        verification date — is echoed from whatever the usage source supplied;
        with no source, ``priced`` alone still lands, so the field is never empty.
        """
        basis = dict(self._cost_basis or {})
        basis["priced"] = self._cost_usd is not None
        return basis

    def finalize(self, status: str, result: Any) -> Path:
        """Write ``audit.json`` and ``audit.md`` to the run directory.

        Returns the run directory path.

        Raises :class:`AuditAlreadyFinalizedError` if called twice. Only
        ``Orchestrator.run`` finalizes a trail, exactly once; a second call
        would mean that rule had been broken somewhere, and silently honouring
        it could overwrite a good trail with a worse one.
        """
        if self._finalized:
            raise AuditAlreadyFinalizedError(
                f"audit trail for run_id={self.run_id!r} was already finalized; "
                f"a run writes its trail exactly once"
            )
        self._finalized = True
        if not self._dir_reserved:
            self._run_dir = self._reserve_run_dir()
            self._dir_reserved = True
        summary = self.summary(status, result)
        envelope = {"summary": summary, "events": [e.to_dict() for e in self._events]}

        (self._run_dir / "audit.json").write_text(
            json.dumps(envelope, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        (self._run_dir / "audit.md").write_text(self._render_markdown(summary), encoding="utf-8")
        return self._run_dir

    def _reserve_run_dir(self) -> Path:
        """Claim a run directory that no other run owns, and return it.

        ``exist_ok=False`` is what makes the claim atomic: the OS refuses the
        second creation of the same name, so the loser retries with a suffix.
        An ``os.path.exists()`` probe would leave a TOCTOU window between the
        check and the create, which is exactly how a trail gets overwritten.
        """
        preferred = self._run_dir
        for suffix in range(MAX_DIR_COLLISIONS + 1):
            candidate = preferred if suffix == 0 else preferred.with_name(
                f"{preferred.name}-{suffix}"
            )
            try:
                os.makedirs(candidate, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise AuditDirectoryError(
            f"could not reserve a run directory: {preferred} and "
            f"{MAX_DIR_COLLISIONS} suffixed variants are all taken. "
            f"Refusing to overwrite an existing audit trail."
        )

    def _render_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            f"# Audit — {summary['run_id']}",
            "",
            f"- **status**: {summary['status']}",
            f"- **started**: {summary['started_at']}",
            f"- **ended**: {summary['ended_at']}",
            f"- **tokens**: {summary['tokens']['total']} "
            f"(prompt {summary['tokens']['prompt']}, completion {summary['tokens']['completion']})",
            f"- **cost_usd**: {_render_cost(summary['cost_usd'])}",
            f"- **cost basis**: {_render_cost_basis(summary.get('cost_basis'))}",
            f"- **provider_stop_reasons**: "
            f"{_render_stop_reasons(summary['provider_stop_reasons'])}",
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
