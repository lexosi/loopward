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
from weakref import WeakSet

GATE_INTERACTIVE = "interactive"
GATE_AUTO = "auto"
GATE_DENY = "deny"
GateMode = str  # one of the GATE_* constants

APPROVE = "approve"
DENY = "deny"

_MINT = object()  # module-private token; only this module can mint an Approval
_MINTED: WeakSet[Approval] = WeakSet()  # identity registry of genuine Approvals

AuditSink = Callable[[str, str], None]
# Prompt function: (phase, summary) -> "y"/"n" answer. Injectable for tests.
Prompter = Callable[[str, str], str]


class Approval:
    """Unforgeable capability token proving a stop-gate APPROVED a phase.

    Only :meth:`StopGate.request` mints one (via :func:`_mint_approval`, which
    holds the module-private ``_MINT`` sentinel and records the instance in the
    ``_MINTED`` identity registry). Genuineness is checked by *identity
    membership* (:func:`is_genuine_approval`), not ``isinstance`` — so a subclass
    or an ``object.__new__(Approval)`` forgery is rejected: it is not in the
    registry. Forging one requires reaching into module-private state, which is
    out of scope (as with any capability).

    Note: genuineness is checked by *identity*, not field value. ``__setattr__``
    blocks casual mutation, but a caller with direct ``object.__setattr__`` access
    could still mutate ``phase`` in place — that is outside the guarantee and
    harmless, since ``phase`` feeds no security decision (it is a label used only
    in ``__repr__``); the identity-registry check is unaffected.
    """

    __slots__ = ("phase", "__weakref__")

    def __init__(self, phase: str, *, _mint: object) -> None:
        if _mint is not _MINT:
            raise PermissionError(
                "Approval cannot be constructed directly; obtain one from "
                "StopGate.request() (it is a capability token, not a plain value)."
            )
        object.__setattr__(self, "phase", phase)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("Approval cannot be subclassed (it is a capability token).")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Approval is immutable")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Approval(phase={self.phase!r})"


def _mint_approval(phase: str) -> Approval:
    """Mint a genuine Approval and record it in the identity registry."""
    approval = Approval(phase, _mint=_MINT)
    _MINTED.add(approval)
    return approval


def is_genuine_approval(approval: object) -> bool:
    """True only for an Approval instance minted by StopGate.request().

    Identity membership in ``_MINTED`` — not ``isinstance`` — so forgeries
    (subclass instances, ``object.__new__`` instances) are rejected. Non-
    weakref-able / unhashable inputs (e.g. ``None``) return False cleanly.
    """
    try:
        return approval in _MINTED
    except TypeError:
        return False


@dataclass(frozen=True)
class Decision:
    """Immutable gate decision."""

    phase: str
    verdict: str  # APPROVE | DENY
    mode: GateMode
    reason: str
    approval: Approval | None = None

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
            decision = Decision(
                phase, APPROVE, self.mode, "auto-approved",
                approval=_mint_approval(phase),
            )
        elif self.mode == GATE_DENY:
            decision = Decision(phase, DENY, self.mode, "deny mode")
        else:
            answer = self._prompter(phase, summary)
            if answer in ("y", "yes"):
                decision = Decision(
                    phase, APPROVE, self.mode, "human approved",
                    approval=_mint_approval(phase),
                )
            else:
                decision = Decision(phase, DENY, self.mode, f"human declined ({answer!r})")

        if self._audit is not None:
            self._audit(
                "gate",
                f"phase '{phase}' [{self.mode}] -> {decision.verdict} ({decision.reason})",
            )
        return decision
