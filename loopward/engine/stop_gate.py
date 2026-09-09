"""stop_gate.py — human approval between critical phases.

A :class:`StopGate` pauses the orchestrator and asks a human to approve before
the next phase runs. Three modes cover the real situations:

- ``interactive``  prompt a human on the console and read their answer.
- ``auto``         approve automatically (CI, demos, batch runs) — logged, so
                   the audit trail still shows the gate was passed and how.
- ``deny``         reject everything (dry-run / safety drills).

Every decision is emitted through the audit sink — as a ``gate`` event carrying
mode, verdict, reason and approver — so "who approved what, when" is in the
trail. The sink is optional on this class and :class:`Orchestrator` attaches it
for any gate that arrives without one; a gate you drive yourself records
nothing until you give it a sink.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol
from weakref import WeakSet

#: The three gate modes, as a type rather than a promise. It used to be
#: ``GateMode = str``, which made ``def f(m: GateMode)`` accept ``"potato"`` —
#: a name that looked like a constraint and was an alias for ``str``.
GateMode = Literal["interactive", "auto", "deny"]

# ``Final`` is load-bearing here, not decoration: without it a type checker
# infers plain ``str`` for these and then rejects ``StopGate(GATE_AUTO)``
# against the Literal above.
GATE_INTERACTIVE: Final[GateMode] = "interactive"
GATE_AUTO: Final[GateMode] = "auto"
GATE_DENY: Final[GateMode] = "deny"

APPROVE = "approve"
DENY = "deny"

_MINT = object()  # module-private token; only this module can mint an Approval
_MINTED: WeakSet[Approval] = WeakSet()  # identity registry of genuine Approvals
#: Genuine Approvals that have been spent. A sibling of ``_MINTED``, and
#: deliberately not the same set: consuming a token does not un-mint it, so
#: ``is_genuine_approval`` keeps meaning "minted by us" and its "did not mint"
#: rejection never lands on a token we did mint. The "spent" bit lives here, at
#: module level, for the same reason "genuine" does — never on the object, whose
#: immutability (``__setattr__`` blocked) the no-evasion tests pin.
_CONSUMED: WeakSet[Approval] = WeakSet()

#: Environment variable a caller can set to declare who is deciding, e.g.
#: ``LOOPWARD_APPROVER=ci:github-actions``.
APPROVER_ENV = "LOOPWARD_APPROVER"

SOURCE_ENV = "env"  # declared explicitly by the caller
SOURCE_OS_USER = "os_user"  # inferred from the OS account
SOURCE_UNKNOWN = "unknown"  # no identity available


class AuditSink(Protocol):
    """Structured event sink: ``(kind, message, **data)``.

    ``AuditLog.record`` satisfies it. The gate emits its decision through this
    sink and nowhere else, so a trail with no ``gate`` event means the sink was
    never attached — not that no decision was taken.
    """

    def __call__(self, kind: str, message: str, **data: Any) -> None: ...


# Prompt function: (phase, summary) -> "y"/"n" answer. Injectable for tests.
Prompter = Callable[[str, str], str]


def _resolve_approver() -> tuple[str | None, str]:
    """Best-effort identity to attribute this gate decision to, plus its source.

    An explicit declaration always wins. Without it, an unattended CI run would
    record the runner's OS account as though a person had approved something —
    in a trail whose whole purpose is "who approved what, when", an identity
    that merely looks true is worse than none. Hence ``approver_source``: it
    travels with the value so a reader can tell a declaration from an inference.

    Never raises. ``getpass.getuser()`` fails on hosts with no passwd entry and
    no user env vars, and a gate decision must not die for want of a name.
    """
    declared = os.environ.get(APPROVER_ENV)
    if declared:
        return declared, SOURCE_ENV
    try:
        return getpass.getuser(), SOURCE_OS_USER
    except Exception:
        return None, SOURCE_UNKNOWN


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
    could still mutate ``phase`` in place. ``phase`` now feeds one decision — the
    phase-binding check in ``Verifier.verify`` — so that direct-mutation route can
    defeat the binding on a genuine token; it is the same out-of-scope hole as a
    caller minting its own approval. The identity-registry check and single-use
    (whose spent bit lives in ``_CONSUMED``, off the object) are both unaffected.
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


def consume_approval(approval: object) -> None:
    """Spend a genuine Approval, so no later use of it is honoured.

    Idempotent, and a no-op on anything that is not a genuine Approval: the
    genuineness gate is :func:`is_genuine_approval` and its callers already run
    it, so spending a forged or already-dead token need not raise here. Called
    once per gated phase, when the phase concludes — whatever the outcome — by
    the code that drove the phase, never by the token's consumer, because the
    consumer may legitimately be retried within the one phase (see the
    orchestrator's verify loop).

    Consuming does not remove the token from ``_MINTED``: spent and never-minted
    are different facts with different rejections. This records only the first.
    """
    if is_genuine_approval(approval):
        _CONSUMED.add(approval)  # type: ignore[arg-type]  # narrowed by is_genuine_approval


def is_consumed_approval(approval: object) -> bool:
    """True for a genuine Approval that :func:`consume_approval` has spent.

    Independent of :func:`is_genuine_approval`: a spent token is still genuine.
    Non-weakref-able / unhashable inputs (e.g. ``None``) return False cleanly.
    """
    try:
        return approval in _CONSUMED
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
    #: Who this decision is attributed to; ``None`` when unidentifiable. Named
    #: for the common case — on a DENY, read it together with ``mode`` and
    #: ``reason`` (a ``deny``-mode refusal is policy, not a person's call).
    approver: str | None = None
    approver_source: str = SOURCE_UNKNOWN  # env | os_user | unknown

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
        # Kept even though `mode` is typed. `cli.py` narrows its own input with
        # argparse `choices`, but what it hands over is still a plain `str`, and
        # StopGate is public: anyone can construct one with "potato". The
        # Literal helps a reader and a checker; this is what holds at runtime.
        if mode not in (GATE_INTERACTIVE, GATE_AUTO, GATE_DENY):
            raise ValueError(f"unknown gate mode {mode!r} (use interactive|auto|deny)")
        self.mode = mode
        self._audit = audit
        self._prompter = prompter or _default_prompter

    def attach_audit(self, sink: AuditSink) -> None:
        """Adopt ``sink`` only if this gate was built without one.

        A caller's own sink always wins — the orchestrator uses this to wire up
        gates that arrived unwired, not to hijack one that came with a sink.
        """
        if self._audit is None:
            self._audit = sink

    def request(self, phase: str, summary: str) -> Decision:
        """Ask for approval to proceed into ``phase``."""
        approver, source = _resolve_approver()
        who = {"approver": approver, "approver_source": source}

        if self.mode == GATE_AUTO:
            decision = Decision(
                phase, APPROVE, self.mode, "auto-approved",
                approval=_mint_approval(phase), **who,
            )
        elif self.mode == GATE_DENY:
            decision = Decision(phase, DENY, self.mode, "deny mode", **who)
        else:
            answer = self._prompter(phase, summary)
            if answer in ("y", "yes"):
                decision = Decision(
                    phase, APPROVE, self.mode, "human approved",
                    approval=_mint_approval(phase), **who,
                )
            else:
                decision = Decision(
                    phase, DENY, self.mode, f"human declined ({answer!r})", **who,
                )

        if self._audit is not None:
            # The message stays byte-identical to what it always was; the
            # decision's parts also go out as structured `data` so a trail
            # reader can filter on them without parsing prose.
            self._audit(
                "gate",
                f"phase '{phase}' [{self.mode}] -> {decision.verdict} ({decision.reason})",
                mode=self.mode,
                verdict=decision.verdict,
                reason=decision.reason,
                approver=decision.approver,
                approver_source=decision.approver_source,
            )
        return decision
