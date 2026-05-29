"""anti_loop.py — stop retrying a losing approach.

The rule: a single subtask may be attempted at most ``MAX_ATTEMPTS`` times with
the *same class of approach*. On the next failure the tracker refuses another
naive retry and demands a **class-jump** — a switch to a materially different
strategy. This is what stops an agent from burning tokens hammering the same
wall.

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
from dataclasses import dataclass

MAX_ATTEMPTS = 3

RETRY = "retry"
CLASS_JUMP = "class_jump"

# Optional sink for structured events, e.g. AuditLog.record.
AuditSink = Callable[[str, str], None]


@dataclass(frozen=True)
class AttemptOutcome:
    """Immutable verdict for one recorded failure."""

    subtask_id: str
    attempt: int
    action: str  # RETRY | CLASS_JUMP
    strategy: str
    message: str

    @property
    def must_class_jump(self) -> bool:
        return self.action == CLASS_JUMP


def decide(attempt: int, max_attempts: int = MAX_ATTEMPTS) -> str:
    """Pure decision: given the attempt number (1-based), retry or class-jump.

    Attempts ``1 .. max_attempts-1`` may retry; attempt ``>= max_attempts``
    forces a class-jump.
    """
    return RETRY if attempt < max_attempts else CLASS_JUMP


class AttemptTracker:
    """Tracks failures per subtask and enforces the class-jump rule.

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

        return AttemptOutcome(
            subtask_id=subtask_id,
            attempt=attempt,
            action=action,
            strategy=strategy,
            message=message,
        )

    def reset(self, subtask_id: str) -> None:
        """Clear the counter for a subtask (e.g. after a successful class-jump)."""
        self._counts.pop(subtask_id, None)
