"""C3 — an empty completion consumes an attempt instead of crossing the loop.

``EmptyCompletionError`` is raised by ``LLMClient.complete`` on blank content.
Until C3 it was neither a ``ReviewParseError`` nor a ``VerifyParseError``, so the
anti-loop's ``except`` clauses did not catch it: it crossed the loop without ever
touching the attempt counter and crashed the run. That is a confirmed defect — a
blank reply is a failed attempt, the same as an unparseable one.

C3 catches it in BOTH loops (review and verify), symmetric with the parse
errors: it consumes an attempt, is recorded, and drives the same class-jump /
exhaustion machinery. Catching it in review only would leave the verify sibling
alive.
"""

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import TASK_VERIFY, LLMClient
from loopward.engine.orchestrator import STATUS_EXHAUSTED, Orchestrator
from loopward.engine.stop_gate import StopGate

DIFF = "return now <= self.expires_at"


def _audit(tmp):
    return AuditLog(run_id="test", base_dir=tmp)


@pytest.mark.integration
def test_empty_review_completion_consumes_attempts_and_exhausts(tmp_path):
    """A blank review reply is a failed attempt, not a crash.

    Today the EmptyCompletionError propagates and the run raises. After C3 it is
    consumed as an attempt across both strategies and the run terminates
    EXHAUSTED, having recorded the attempts.
    """
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "")
    result = Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)

    assert result.status == STATUS_EXHAUSTED
    # the counter was touched: each blank reply spent an attempt, bounded exactly
    # like the parse-error path at MAX_ATTEMPTS x len(STRATEGIES) = 6. (totals
    # ["calls"] stays 0: an empty completion raises before it is counted, which is
    # why the attempt events are the honest measure of what was spent.)
    assert len([e for e in audit.events if e.kind == "attempt"]) == 6


@pytest.mark.integration
def test_empty_verify_completion_consumes_attempts_and_exhausts(tmp_path):
    """Same defect on the verify loop: a blank adjudication is a failed attempt.

    Review parses; verify returns blank. Today that crashes; after C3 the verify
    loop consumes attempts and returns no usable adjudication -> EXHAUSTED.
    """

    def reply(messages, task):
        return "" if task == TASK_VERIFY else "HIGH: real finding"

    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=reply)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)

    assert result.status == STATUS_EXHAUSTED
    assert "adjudication" in result.summary
    assert [e for e in audit.events if e.kind == "attempt"]
