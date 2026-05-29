"""stop_gate.py — human approval between critical phases.

A :class:`StopGate` pauses the orchestrator and asks a human to approve before
the next phase runs. Three modes cover the real situations:

- ``interactive``  prompt a human on the console and read their answer.
- ``auto``         approve automatically (CI, demos, batch runs) — logged, so
                   the audit trail still shows the gate was passed and how.
- ``deny``         reject everything (dry-run / safety drills).

Every decision is recorded through the optional audit sink, so "who approved
what, when" is always in the trail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

GATE_INTERACTIVE = "interactive"
GATE_AUTO = "auto"
GATE_DENY = "deny"
GateMode = str  # one of the GATE_* constants

APPROVE = "approve"
DENY = "deny"

AuditSink = Callable[[str, str], None]
# Prompt function: (phase, summary) -> "y"/"n" answer. Injectable for tests.
Prompter = Callable[[str, str], str]


@dataclass(frozen=True)
class Decision:
    """Immutable gate decision."""

    phase: str
    verdict: str  # APPROVE | DENY
    mode: GateMode
    reason: str

    @property
    def approved(self) -> bool:
        return self.verdict == APPROVE


def _default_prompter(phase: str, summary: str) -> str:
    """Console prompt used in interactive mode."""
    print(f"\n[stop-gate] phase '{phase}' awaiting approval:")
    print(f"  {summary}")
    return input("  approve? [y/N] ").strip().lower()


class StopGate:
    """Gate the transition into a phase, requiring approval per the mode.

    Parameters
    ----------
    mode:
        ``interactive`` | ``auto`` | ``deny``. Defaults to ``interactive``.
    audit:
        Optional ``(kind, message)`` callback; called for every decision.
    prompter:
        Override the console prompt (used by tests). Only used in interactive mode.
    """

    def __init__(
        self,
        mode: GateMode = GATE_INTERACTIVE,
        audit: AuditSink | None = None,
        prompter: Prompter | None = None,
    ) -> None:
        if mode not in (GATE_INTERACTIVE, GATE_AUTO, GATE_DENY):
            raise ValueError(f"unknown gate mode {mode!r} (use interactive|auto|deny)")
        self.mode = mode
        self._audit = audit
        self._prompter = prompter or _default_prompter

    def request(self, phase: str, summary: str) -> Decision:
        """Ask for approval to proceed into ``phase``."""
        if self.mode == GATE_AUTO:
            decision = Decision(phase, APPROVE, self.mode, "auto-approved")
        elif self.mode == GATE_DENY:
            decision = Decision(phase, DENY, self.mode, "deny mode")
        else:
            answer = self._prompter(phase, summary)
            if answer in ("y", "yes"):
                decision = Decision(phase, APPROVE, self.mode, "human approved")
            else:
                decision = Decision(phase, DENY, self.mode, f"human declined ({answer!r})")

        if self._audit is not None:
            self._audit(
                "gate",
                f"phase '{phase}' [{self.mode}] -> {decision.verdict} ({decision.reason})",
            )
        return decision
