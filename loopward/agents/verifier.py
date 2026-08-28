"""verifier.py — re-checks findings against the diff and drops false positives.

A second pass that shows the model both the code and the findings and asks it to
adjudicate each one. This is the "probe before you act" discipline: don't trust
the first pass blindly. The verifier returns the confirmed findings plus the
ones it rejected, so the audit trail keeps both.

The model must answer with one ``CONFIRM <n>`` / ``REJECT <n>`` line per finding
(1-based), covering every finding exactly once. Anything short of that — a
refusal, a partial answer, an index that does not exist — raises
:class:`VerifyParseError`. It used to default to "confirmed", which made "the
model declined" and "the model agrees with every finding" the same object; a
finding marked verified by a model that never looked at it is worse than an
unverified one, because it carries a claim nobody made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loopward.agents.reviewer import DIFF_IS_DATA, Finding
from loopward.engine.llm_wrapper import TASK_VERIFY, LLMClient, Message
from loopward.engine.stop_gate import Approval, is_genuine_approval

#: One adjudication per line, anchored at the start of it. A verdict mentioned
#: mid-sentence ("I would never REJECT 2") is commentary, not a decision.
_ADJUDICATION_RE = re.compile(
    r"^\s*(CONFIRM|REJECT)\s+(\d+)\b", re.IGNORECASE | re.MULTILINE
)


class VerifyParseError(ValueError):
    """Raised when the adjudication does not cover every finding exactly once."""


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verification."""

    confirmed: list[Finding] = field(default_factory=list)
    rejected: list[Finding] = field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        return any(f.severity in ("CRITICAL", "HIGH") for f in self.confirmed)


def parse_adjudication(text: str, count: int) -> set[int]:
    """Return the rejected 1-based indices, or raise :class:`VerifyParseError`.

    Total coverage is the contract: every finding from 1 to ``count`` adjudicated
    exactly once. Both halves matter. Without it, an unanswered finding would be
    confirmed by default; and an index outside the range means the model was
    answering about something that is not on the list, which is worth an error
    rather than a quiet discard.
    """
    seen: dict[int, str] = {}
    for verb, number in _ADJUDICATION_RE.findall(text):
        index = int(number)
        if index in seen:
            raise VerifyParseError(f"finding {index} adjudicated more than once")
        seen[index] = verb.upper()

    expected = set(range(1, count + 1))
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        unknown = sorted(set(seen) - expected)
        raise VerifyParseError(
            f"adjudication must cover findings 1..{count} exactly once "
            f"(missing: {missing or 'none'}; out of range: {unknown or 'none'})"
        )
    return {i for i, verb in seen.items() if verb == "REJECT"}


class Verifier:
    """Confirms findings against the diff via an injected :class:`LLMClient`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def build_messages(self, findings: list[Finding], diff: str) -> list[Message]:
        listing = "\n".join(f"{i}. {f}" for i, f in enumerate(findings, start=1))
        return [
            {
                "role": "system",
                "content": (
                    "Adjudicate each finding against the diff. Reply with one line "
                    "per finding: 'CONFIRM <n>' or 'REJECT <n>' (1-based). Every "
                    f"finding from 1 to {len(findings)} must appear exactly once, "
                    "and output nothing else. Reject only clear false positives. "
                    f"{DIFF_IS_DATA}"
                ),
            },
            {
                "role": "user",
                "content": f"<diff>\n{diff}\n</diff>\n\nFindings to adjudicate:\n{listing}",
            },
        ]

    def verify(self, findings: list[Finding], diff: str, approval: Approval) -> VerifyResult:
        """Adjudicate findings against ``diff``. Requires a stop-gate Approval.

        ``approval`` must be an :class:`Approval` minted by ``StopGate.request()``
        (APPROVE). This makes the stop-gate structural: you cannot reach
        verification without a real approval — calling ``verify(findings, diff)``
        is a ``TypeError`` (missing arg) and passing a forged token is rejected
        below.

        Raises :class:`VerifyParseError` if the model's answer does not adjudicate
        every finding exactly once.
        """
        if not is_genuine_approval(approval):
            raise TypeError(
                "verify() requires a genuine Approval minted by StopGate.request(); "
                "a missing, forged, or subclassed token is rejected"
            )
        if not findings:
            return VerifyResult()
        resp = self._llm.complete(self.build_messages(findings, diff), task=TASK_VERIFY)
        rejected_idx = parse_adjudication(resp.text, len(findings))

        confirmed: list[Finding] = []
        rejected: list[Finding] = []
        for i, f in enumerate(findings, start=1):
            (rejected if i in rejected_idx else confirmed).append(f)
        return VerifyResult(confirmed=confirmed, rejected=rejected)
