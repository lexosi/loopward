"""reviewer.py — turns a diff into severity-tagged findings.

The reviewer asks an LLM to review a diff and emit one finding per line in the
form ``SEVERITY: message``. Parsing is strict: if the model returns nothing that
looks like a finding, :func:`parse_findings` raises :class:`ReviewParseError`.

That strictness is deliberate — it gives the orchestrator a real failure signal
to feed the anti-loop tracker. A flaky prompt that never parses will, after a
few attempts, force a *class-jump* to a different review ``strategy`` instead of
looping forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from guardloop.engine.llm_wrapper import LLMClient, Message

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
_FINDING_RE = re.compile(rf"^\s*({'|'.join(SEVERITIES)})\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Review strategies. The orchestrator switches to the next one on a class-jump.
STRATEGIES = ("concise", "structured")

_STRATEGY_INSTRUCTIONS = {
    "concise": "Review the diff. List issues, most severe first.",
    "structured": (
        "You are a strict code reviewer. Output ONE finding per line and nothing "
        "else. Each line MUST start with a severity tag from "
        f"{', '.join(SEVERITIES)} followed by a colon. Example:\n"
        "HIGH: off-by-one in loop bound."
    ),
}


class ReviewParseError(ValueError):
    """Raised when the model output contains no parseable findings."""


@dataclass(frozen=True)
class Finding:
    """One review finding."""

    severity: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def parse_findings(text: str) -> list[Finding]:
    """Parse ``SEVERITY: message`` lines. Raises if none are found."""
    findings: list[Finding] = []
    for line in text.splitlines():
        m = _FINDING_RE.match(line)
        if m:
            findings.append(Finding(severity=m.group(1).upper(), message=m.group(2)))
    if not findings:
        raise ReviewParseError("no severity-tagged findings in model output")
    return findings


class Reviewer:
    """Reviews a diff via an injected :class:`LLMClient`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def build_messages(self, diff: str, strategy: str) -> list[Message]:
        instruction = _STRATEGY_INSTRUCTIONS.get(strategy, _STRATEGY_INSTRUCTIONS["concise"])
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Review this diff:\n\n{diff}"},
        ]

    def review(self, diff: str, strategy: str = "concise") -> list[Finding]:
        """Return findings, or raise :class:`ReviewParseError` on unusable output."""
        resp = self._llm.complete(self.build_messages(diff, strategy))
        return parse_findings(resp.text)
