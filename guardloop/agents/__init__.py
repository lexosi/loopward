"""guardloop agents: focused workers driven by the orchestrator.

- Reviewer: reads a diff, returns severity-tagged findings.
- Verifier: re-checks findings and drops likely false positives.

Agents never call each other; the orchestrator coordinates them.
"""

from guardloop.agents.reviewer import Finding, Reviewer, ReviewParseError, parse_findings
from guardloop.agents.verifier import Verifier, VerifyResult

__all__ = [
    "Finding",
    "ReviewParseError",
    "Reviewer",
    "parse_findings",
    "VerifyResult",
    "Verifier",
]
