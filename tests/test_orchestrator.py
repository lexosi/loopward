"""Integration tests for the orchestrator (fake provider, offline)."""


import pytest

from loopward.engine import orchestrator as orch_mod
from loopward.engine.anti_loop import AttemptTracker
from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import LLMClient, Message
from loopward.engine.orchestrator import (
    STATUS_EXHAUSTED,
    STATUS_GATE_DENIED,
    STATUS_OK,
    Orchestrator,
)
from loopward.engine.stop_gate import StopGate

DIFF = "return now <= self.expires_at"


def _audit(tmp):
    return AuditLog(run_id="test", base_dir=tmp)


@pytest.mark.integration
def test_happy_path_ok(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_OK
    assert result.verify is not None
    assert result.verify.confirmed


@pytest.mark.integration
def test_gate_denied_stops_before_verify(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    orch = Orchestrator(llm, StopGate(mode="deny"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_GATE_DENIED
    assert result.verify is None
    assert result.findings  # findings were produced before the gate


@pytest.mark.integration
def test_anti_loop_class_jump_then_success(tmp_path):
    """Concise strategy never parses; structured one does -> class-jump rescues the run."""
    llm = LLMClient(provider="fake", fake_script=_force_class_jump)
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_OK
    kinds = [e.kind for e in orch._audit.events]
    assert "class_jump" in kinds


@pytest.mark.integration
def test_exhausted_when_nothing_parses(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m: "no tags here, just prose")
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert result.findings == []


@pytest.mark.integration
def test_audit_files_written(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    from pathlib import Path

    run_dir = Path(result.run_dir)
    assert (run_dir / "audit.json").exists()
    assert (run_dir / "audit.md").exists()


# ---- fake reply helpers ---------------------------------------------------


def _reply(messages: list[Message]) -> str:
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    if "confirm or reject each finding" in system.lower():
        return "CONFIRM all"
    return "HIGH: token expiry uses <=, use <"


def _force_class_jump(messages: list[Message]) -> str:
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    if "confirm or reject each finding" in system.lower():
        return "CONFIRM all"
    if "strict code reviewer" in system.lower():  # structured strategy
        return "HIGH: token expiry uses <=, use <"
    return "vague prose with no severity tags"  # concise strategy: unparseable


# ---- the gate must leave a trace in the trail -------------------------------
# loopward's third headline is the audit trail, and the stop-gate is the
# primitive the README leads with. A gate decision that happens but is not
# recorded is indistinguishable, after the fact, from a gate that never ran.


@pytest.mark.integration
def test_gate_decision_lands_in_the_trail(tmp_path):
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)
    assert "gate" in {e.kind for e in audit.events}


@pytest.mark.integration
def test_gate_denied_is_recorded_too(tmp_path):
    # A refusal is the decision most worth having in the trail.
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    Orchestrator(llm, StopGate(mode="deny"), audit=audit).run(DIFF)
    gate_events = [e for e in audit.events if e.kind == "gate"]
    assert len(gate_events) == 1
    assert gate_events[0].data["verdict"] == "deny"


@pytest.mark.integration
def test_gate_event_carries_structured_data(tmp_path):
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)
    data = next(e for e in audit.events if e.kind == "gate").data
    assert data["mode"] == "auto"
    assert data["verdict"] == "approve"
    assert data["reason"]
    # who decided, and whether that identity was declared or inferred
    assert data["approver_source"] in ("env", "os_user", "unknown")
    assert "approver" in data


@pytest.mark.integration
def test_injected_tracker_still_records_attempts(tmp_path):
    """Same defect class as the gate: a collaborator passed in ready-made.

    The wiring used to live inside ``tracker or AttemptTracker(audit=...)``, so
    it only ever reached the branch the orchestrator built itself.
    """
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=_force_class_jump)
    Orchestrator(llm, StopGate(mode="auto"), audit=audit,
                 tracker=AttemptTracker()).run(DIFF)
    assert [e for e in audit.events if e.kind == "attempt"]


@pytest.mark.integration
def test_orchestrator_does_not_override_an_explicit_sink(tmp_path):
    """Adopting unwired collaborators must not hijack a caller's own sink."""
    mine: list[tuple[str, str]] = []
    audit = _audit(tmp_path)
    gate = StopGate(mode="auto", audit=lambda kind, msg, **data: mine.append((kind, msg)))
    llm = LLMClient(provider="fake", fake_script=lambda m: _reply(m))
    Orchestrator(llm, gate, audit=audit).run(DIFF)
    assert mine and mine[0][0] == "gate"
    assert "gate" not in {e.kind for e in audit.events}


# ---- the bound belongs to the loop, not to the collaborator -----------------
# `Orchestrator(..., tracker=..., strategies=...)` are public parameters. Both
# multiply into the anti-loop's budget, so if the loop trusts them for its
# termination, "never spins" is a promise the caller gets to break.

SPIN_GUARD = 600  # far above MAX_TOTAL_ATTEMPTS; trips only if the loop is unbounded


class LoopUnbounded(RuntimeError):
    """Raised by the test double instead of letting the suite hang."""


def _never_parses():
    """Fake LLM whose output never parses, and that screams rather than spin."""
    calls = {"n": 0}

    def reply(_messages: list[Message]) -> str:
        calls["n"] += 1
        if calls["n"] > SPIN_GUARD:
            raise LoopUnbounded(f"{calls['n']} LLM calls and still going")
        return "vague prose with no severity tags"

    return reply, calls


@pytest.mark.integration
def test_injected_tracker_cannot_spin(tmp_path):
    """A tracker that never grants a class-jump must not buy an unbounded loop."""
    reply, calls = _never_parses()
    llm = LLMClient(provider="fake", fake_script=reply)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                          tracker=AttemptTracker(max_attempts=10**9)).run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert calls["n"] <= orch_mod.MAX_TOTAL_ATTEMPTS


@pytest.mark.integration
def test_injected_strategies_cannot_spin(tmp_path):
    """Same defect, the other multiplier: the default tracker is untouched here."""
    reply, calls = _never_parses()
    llm = LLMClient(provider="fake", fake_script=reply)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                          strategies=tuple(f"s{i}" for i in range(10**6))).run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert calls["n"] <= orch_mod.MAX_TOTAL_ATTEMPTS
