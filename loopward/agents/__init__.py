"""loopward agents: focused workers driven by the orchestrator.

- Reviewer: reads a diff, returns severity-tagged findings.
- Verifier: re-checks findings and drops likely false positives.

Agents never call each other; the orchestrator coordinates them.
"""

from loopward.agents.reviewer import Finding, ReviewParseError, parse_findings
from loopward.agents.verifier import VerifyParseError, VerifyResult

# NOTE: Reviewer and Verifier are intentionally NOT re-exported here. They are
# internal workers driven only by Orchestrator.run(); calling them directly
# bypasses the anti-loop/stop-gate coordination.
#
# One structural guard survives that bypass, not all of them: Verifier.verify
# requires a real Approval minted by StopGate, so the gate cannot be skipped by
# importing the submodule. The Reviewer has no such guard and needs none — it
# holds no capability. What it loses is the coordination: Reviewer(llm).review()
# runs with no anti-loop budget, no attempt tracking and no audit trail. The
# supported public API is loopward.Orchestrator.
__all__ = [
    "Finding",
    "ReviewParseError",
    "parse_findings",
    "VerifyParseError",
    "VerifyResult",
]
