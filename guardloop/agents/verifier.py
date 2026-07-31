"""verifier.py — re-checks findings and drops likely false positives.

A second pass that asks the model to confirm or reject each finding. This is the
"probe before you act" discipline: don't trust the first pass blindly. The
verifier returns the confirmed findings plus the ones it rejected, so the audit
trail keeps both.

The model is asked to reply with ``CONFIRM <n>`` / ``REJECT <n>`` lines (1-based).
Parsing is forgiving: anything not explicitly rejected stays confirmed, so a
vague verifier never silently deletes real findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from guardloop.agents.reviewer import Finding
from guardloop.engine.llm_wrapper import LLMClient, Message
from guardloop.engine.stop_gate import Approval, is_genuine_approval

# Marker the fake provider can recognise to behave like a verifier offline.
VERIFY_MARKER = "Confirm or reject each finding"
_REJECT_RE = re.compile(r"\bREJECT\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verification."""

    confirmed: list[Finding] = field(default_factory=list)
    rejected: list[Finding] = field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        return any(f.severity in ("CRITICAL", "HIGH") for f in self.confirmed)


class Verifier:
    """Confirms findings via an injected :class:`LLMClient`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def build_messages(self, findings: list[Finding]) -> list[Message]:
        listing = "\n".join(f"{i}. {f}" for i, f in enumerate(findings, start=1))
        return [
            {
                "role": "system",
                "content": (
                    f"{VERIFY_MARKER}. Reply with lines 'CONFIRM <n>' or "
                    "'REJECT <n>' (1-based). Reject only clear false positives."
                ),
            },
            {"role": "user", "content": listing},
        ]

    def verify(self, findings: list[Finding], approval: Approval) -> VerifyResult:
        """Confirm/reject findings. Requires a valid stop-gate Approval.

        ``approval`` must be an :class:`Approval` minted by
        ``StopGate.request()`` (APPROVE). This makes the stop-gate structural:
        you cannot reach verification without a real approval — calling
        ``verify(findings)`` is a ``TypeError`` (missing arg) and passing a
        forged token is rejected below.
        """
        if not is_genuine_approval(approval):
            raise TypeError(
                "verify() requires a genuine Approval minted by StopGate.request(); "
                "a missing, forged, or subclassed token is rejected"
            )
        if not findings:
            return VerifyResult()
        resp = self._llm.complete(self.build_messages(findings))
        rejected_idx = {int(n) for n in _REJECT_RE.findall(resp.text)}

        confirmed: list[Finding] = []
        rejected: list[Finding] = []
        for i, f in enumerate(findings, start=1):
            (rejected if i in rejected_idx else confirmed).append(f)
        return VerifyResult(confirmed=confirmed, rejected=rejected)
