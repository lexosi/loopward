"""Tests for the stop-gate."""

import pytest

from guardloop.engine.stop_gate import StopGate


@pytest.mark.unit
def test_auto_mode_approves():
    gate = StopGate(mode="auto")
    d = gate.request("verify", "summary")
    assert d.approved is True
    assert d.verdict == "approve"


@pytest.mark.unit
def test_deny_mode_rejects():
    gate = StopGate(mode="deny")
    d = gate.request("verify", "summary")
    assert d.approved is False


@pytest.mark.unit
def test_interactive_yes_approves():
    gate = StopGate(mode="interactive", prompter=lambda phase, summary: "y")
    assert gate.request("verify", "s").approved is True


@pytest.mark.unit
def test_interactive_no_denies():
    gate = StopGate(mode="interactive", prompter=lambda phase, summary: "n")
    assert gate.request("verify", "s").approved is False


@pytest.mark.unit
def test_decision_is_audited():
    events = []
    gate = StopGate(mode="auto", audit=lambda kind, msg: events.append((kind, msg)))
    gate.request("verify", "s")
    assert events and events[0][0] == "gate"


@pytest.mark.unit
def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        StopGate(mode="whatever")
