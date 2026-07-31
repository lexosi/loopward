"""orchestrator.py — the single coordinator.

Runs a code-review as two phases with reliability baked in:

1. **review** — ask the Reviewer for findings. If the output won't parse, the
   failure is fed to the :class:`AttemptTracker`. After ``MAX_ATTEMPTS`` the
   tracker forces a *class-jump*: the orchestrator switches to a different review
   strategy instead of retrying the same one forever. If strategies run out, the
   run fails cleanly (no infinite loop).
2. **verify** — gated by the :class:`StopGate`. A human (or ``auto`` mode in CI)
   approves before the Verifier re-checks the findings.

Everything — every attempt, gate decision, and token cost — lands in the
:class:`AuditLog`. The orchestrator is the only component that calls agents;
agents never call each other.

All collaborators are injected, so the whole flow runs offline and is trivial to
test with the ``fake`` provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from guardloop.agents.reviewer import STRATEGIES, Finding, Reviewer, ReviewParseError
from guardloop.agents.verifier import Verifier, VerifyResult
from guardloop.engine.anti_loop import AttemptTracker
from guardloop.engine.audit import AuditLog
from guardloop.engine.llm_wrapper import LLMClient
from guardloop.engine.stop_gate import StopGate

REVIEW_SUBTASK = "review-parse"

STATUS_OK = "ok"
STATUS_GATE_DENIED = "gate_denied"
STATUS_EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class RunResult:
    """Final outcome of a run."""

    status: str
    findings: list[Finding]
    verify: VerifyResult | None
    run_dir: str
    summary: str


class Orchestrator:
    """Coordinates reviewer + verifier under anti-loop and a stop-gate."""

    def __init__(
        self,
        llm: LLMClient,
        gate: StopGate,
        audit: AuditLog | None = None,
        tracker: AttemptTracker | None = None,
        strategies: tuple[str, ...] = STRATEGIES,
    ) -> None:
        self._llm = llm
        self._gate = gate
        self._audit = audit or AuditLog(run_id="code-review")
        self._tracker = tracker or AttemptTracker(audit=self._audit.record)
        self._strategies = strategies
        self._reviewer = Reviewer(llm)
        self._verifier = Verifier(llm)

    def run(self, diff: str) -> RunResult:
        """Execute the review→verify flow and finalize the audit trail."""
        self._audit.record("phase", "review: start")
        findings, strategy = self._review_with_anti_loop(diff)

        if findings is None:
            summary = "review exhausted all strategies without parseable findings"
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_EXHAUSTED, summary)
            return RunResult(STATUS_EXHAUSTED, [], None, str(run_dir), summary)

        self._audit.record(
            "phase", f"review: ok via strategy '{strategy}' ({len(findings)} findings)"
        )

        gate_summary = f"{len(findings)} findings ready to verify"
        decision = self._gate.request("verify", gate_summary)
        if not decision.approved:
            summary = f"stop-gate denied verification ({decision.reason})"
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_GATE_DENIED, summary)
            return RunResult(STATUS_GATE_DENIED, findings, None, str(run_dir), summary)

        self._audit.record("phase", "verify: start")
        verify = self._verifier.verify(findings, decision.approval)
        self._audit.record(
            "phase",
            f"verify: confirmed {len(verify.confirmed)}, rejected {len(verify.rejected)}",
        )

        summary = (
            f"{len(verify.confirmed)} confirmed finding(s)"
            + (", blocking" if verify.has_blocking else ", none blocking")
        )
        self._audit.record("result", summary)
        run_dir = self._record_usage_and_finalize(STATUS_OK, summary)
        return RunResult(STATUS_OK, findings, verify, str(run_dir), summary)

    # ---- internals --------------------------------------------------------

    def _review_with_anti_loop(self, diff: str) -> tuple[list[Finding] | None, str]:
        """Try strategies in order, advancing on a class-jump. Never loops forever."""
        strat_idx = 0
        while strat_idx < len(self._strategies):
            strategy = self._strategies[strat_idx]
            try:
                findings = self._reviewer.review(diff, strategy=strategy)
                return findings, strategy
            except ReviewParseError:
                outcome = self._tracker.record_failure(REVIEW_SUBTASK, strategy=strategy)
                if outcome.must_class_jump:
                    self._audit.record(
                        "class_jump",
                        f"switching review strategy after {outcome.attempt} failures "
                        f"on '{strategy}'",
                    )
                    self._tracker.reset(REVIEW_SUBTASK)
                    strat_idx += 1
        return None, ""

    def _record_usage_and_finalize(self, status: str, result: str):
        # Pull accumulated usage from the client and fold it into the audit trail
        # once, just before writing it out.
        t = self._llm.totals
        self._audit.record_usage(
            prompt=int(t["prompt"]), completion=int(t["completion"]), cost_usd=t["cost_usd"]
        )
        return self._audit.finalize(status, result)
