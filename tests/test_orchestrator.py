"""Integration tests for the orchestrator (fake provider, offline)."""


import pytest

from guardloop.engine.audit import AuditLog
from guardloop.engine.llm_wrapper import LLMClient, Message
from guardloop.engine.orchestrator import (
    STATUS_EXHAUSTED,
    STATUS_GATE_DENIED,
    STATUS_OK,
    Orchestrator,
)
from guardloop.engine.stop_gate import StopGate

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
