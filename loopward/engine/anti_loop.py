"""anti_loop.py — stop retrying a losing approach.

The rule: a single subtask may be attempted at most ``MAX_ATTEMPTS`` times with
the *same class of approach*. On the next failure the tracker returns a
**class-jump** verdict — the instruction to switch to a materially different
strategy.

The tracker issues verdicts; it does not enforce them. ``record_failure`` can be
called any number of times and always returns successfully. What actually stops
an agent from burning tokens hammering the same wall is
``Orchestrator._review_with_anti_loop``, which owns the loop and computes its
budget before entering it.

Usage
-----
>>> t = AttemptTracker()
>>> t.record_failure("parse-findings", strategy="regex").action
'retry'
>>> t.record_failure("parse-findings", strategy="regex").action
'retry'
>>> t.record_failure("parse-findings", strategy="regex").action
'class_jump'

The decision itself is a pure function (:func:`decide`) so it is trivial to test
and reason about; the tracker only owns the per-subtask counters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from weakref import WeakSet

MAX_ATTEMPTS = 3

RETRY = "retry"
CLASS_JUMP = "class_jump"

_MINT = object()  # module-private token; only this module can mint an AttemptOutcome
_MINTED: WeakSet[AttemptOutcome] = WeakSet()  # identity registry of genuine outcomes

# Optional sink for structured events. This is the *minimum* the tracker needs —
# two positionals — not the full sink shape: it records ``(kind, message)`` and
# no structured data. ``AuditLog.record(kind, message, **data)`` satisfies it,
# and so does the wider Protocol of the same name in ``stop_gate``, which the
# gate needs because it does emit data.
AuditSink = Callable[[str, str], None]


@dataclass(frozen=True, eq=False)
class AttemptOutcome:
    """Verdict for one recorded failure. Only the tracker mints one.

    A hand-built ``AttemptOutcome(...)`` raises and the class cannot be
    subclassed. Copying a genuine one is not closed off, though: both
    ``dataclasses.replace`` and ``copy.copy`` carry ``_mint`` over from the
    original and construct without raising, and ``replace`` can change
    ``action`` on the way through. Neither copy is registered, so
    :func:`is_genuine_outcome` rejects both and the orchestrator acts on
    neither — but constructing one is not the same as minting one, and only
    the second is closed.

    **Not immutable**, and the caveat weighs more here than the matching one on
    ``Approval``. ``frozen=True`` blocks casual assignment, but
    ``object.__setattr__(outcome, "action", CLASS_JUMP)`` mutates a genuine,
    registered outcome in place: ``must_class_jump`` flips and
    :func:`is_genuine_outcome` still returns True. ``Approval`` can call its own
    mutable field harmless because ``phase`` feeds no decision; here ``action``
    *is* the decision, and nothing detects the edit.

    What minting closes is also narrower than it looks. Forging a ``class_jump``
    makes the loop stop *earlier* — the harmless direction. The costly verdict is
    a perpetual ``retry``, and that one never needed forging: ``record_failure``
    hands it out. So this token is not the bound on retries. The bound is the
    budget in ``Orchestrator._review_with_anti_loop``, computed before the loop
    and enforced there whatever the injected tracker returns.
    """

    subtask_id: str
    attempt: int
    action: str  # RETRY | CLASS_JUMP
    strategy: str
    message: str
    _mint: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise PermissionError(
                "AttemptOutcome cannot be constructed directly; obtain one from "
                "AttemptTracker.record_failure() (it is a capability token)."
            )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AttemptOutcome cannot be subclassed (it is a capability token).")

    @property
    def must_class_jump(self) -> bool:
        return self.action == CLASS_JUMP


def decide(attempt: int, max_attempts: int = MAX_ATTEMPTS) -> str:
    """Pure decision: given the attempt number (1-based), retry or class-jump.

    Attempts ``1 .. max_attempts-1`` may retry; attempt ``>= max_attempts``
    forces a class-jump.
    """
    return RETRY if attempt < max_attempts else CLASS_JUMP


def _mint_outcome(
    subtask_id: str, attempt: int, action: str, strategy: str, message: str
) -> AttemptOutcome:
    """Mint a genuine AttemptOutcome and record it in the identity registry."""
    outcome = AttemptOutcome(
        subtask_id=subtask_id,
        attempt=attempt,
        action=action,
        strategy=strategy,
        message=message,
        _mint=_MINT,
    )
    _MINTED.add(outcome)
    return outcome


def is_genuine_outcome(outcome: object) -> bool:
    """True only for an AttemptOutcome minted by AttemptTracker.record_failure().

    Identity membership in ``_MINTED`` (not ``isinstance``) — subclass /
    ``object.__new__`` forgeries are rejected. Non-weakref-able / unhashable
    inputs return False cleanly.
    """
    try:
        return outcome in _MINTED
    except TypeError:
        return False


class AttemptTracker:
    """Tracks failures per subtask and issues the class-jump verdict.

    It issues; it does not enforce — this line used to claim it did, directly
    contradicting the module docstring above.
    ``record_failure`` can be called any number of times, always returns, and
    goes on returning ``class_jump`` forever once the count is past
    ``max_attempts`` — it refuses nothing. What stops the retrying is
    ``Orchestrator._review_with_anti_loop``, which owns the loop and its budget.

    Parameters
    ----------
    max_attempts:
        Retries allowed before a class-jump is forced. Defaults to
        :data:`MAX_ATTEMPTS`.
    audit:
        Optional callback ``(kind, message)`` invoked on every recorded failure.
    """

    def __init__(self, max_attempts: int = MAX_ATTEMPTS, audit: AuditSink | None = None) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self._audit = audit
        self._counts: dict[str, int] = {}

    def attach_audit(self, sink: AuditSink) -> None:
        """Adopt ``sink`` only if this tracker was built without one.

        Counterpart to :meth:`StopGate.attach_audit`; see the wiring loop in
        ``Orchestrator.__init__`` for why this is a method and not a constructor
        argument the orchestrator fills in.
        """
        if self._audit is None:
            self._audit = sink

    def attempts(self, subtask_id: str) -> int:
        """How many failures have been recorded for this subtask."""
        return self._counts.get(subtask_id, 0)

    def record_failure(self, subtask_id: str, strategy: str) -> AttemptOutcome:
        """Record a failed attempt and return the verdict for what to do next."""
        attempt = self._counts.get(subtask_id, 0) + 1
        self._counts[subtask_id] = attempt
        action = decide(attempt, self.max_attempts)

        if action == CLASS_JUMP:
            message = (
                f"subtask '{subtask_id}' failed {attempt}x with strategy '{strategy}' — "
                f"class-jump required (no further retries with the same approach)"
            )
        else:
            remaining = self.max_attempts - attempt
            message = (
                f"subtask '{subtask_id}' failed (attempt {attempt}/{self.max_attempts}, "
                f"strategy '{strategy}') — retry allowed ({remaining} left)"
            )

        if self._audit is not None:
            self._audit("attempt", message)

        return _mint_outcome(subtask_id, attempt, action, strategy, message)

    def reset(self, subtask_id: str) -> None:
        """Clear the counter for a subtask (e.g. after a successful class-jump)."""
        self._counts.pop(subtask_id, None)
