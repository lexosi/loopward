"""loopward agents: focused workers driven by the orchestrator.

- Reviewer: reads a diff, returns severity-tagged findings.
- Verifier: re-checks findings and drops likely false positives.

Agents never call each other; the orchestrator coordinates them.
"""

from loopward.agents.reviewer import Finding, ReviewParseError, parse_findings
from loopward.agents.verifier import VerifyResult

# NOTE: Reviewer and Verifier are intentionally NOT re-exported here. They are
# internal workers driven only by Orchestrator.run(); calling them directly
# bypasses the anti-loop/stop-gate coordination. The structural guards still
# hold if someone imports them from their submodules anyway (Verifier.verify
# requires a real Approval token minted by StopGate), but the supported public
# API is loopward.Orchestrator.
__all__ = [
    "Finding",
    "ReviewParseError",
    "parse_findings",
    "VerifyResult",
]
