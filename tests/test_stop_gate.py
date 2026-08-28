"""Tests for the stop-gate."""

import pytest

from loopward.engine.stop_gate import StopGate


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
    gate = StopGate(mode="auto", audit=lambda kind, msg, **data: events.append((kind, msg)))
    gate.request("verify", "s")
    assert events and events[0][0] == "gate"


@pytest.mark.unit
def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        StopGate(mode="whatever")


# ---- who decided ------------------------------------------------------------
# The trail's purpose is "who approved what, when". An identity that merely
# looks true is worse than none, so the source travels with the value.


@pytest.mark.unit
def test_approver_declared_via_env_wins(monkeypatch):
    monkeypatch.setenv("LOOPWARD_APPROVER", "ci:github-actions")
    d = StopGate(mode="auto").request("verify", "s")
    assert d.approver == "ci:github-actions"
    assert d.approver_source == "env"


@pytest.mark.unit
def test_approver_falls_back_to_os_user(monkeypatch):
    monkeypatch.delenv("LOOPWARD_APPROVER", raising=False)
    monkeypatch.setattr("loopward.engine.stop_gate.getpass.getuser", lambda: "ada")
    d = StopGate(mode="auto").request("verify", "s")
    assert d.approver == "ada"
    assert d.approver_source == "os_user"


@pytest.mark.unit
def test_unidentifiable_approver_never_breaks_the_gate(monkeypatch):
    """A gate decision must not die because the OS has no name for the caller."""
    monkeypatch.delenv("LOOPWARD_APPROVER", raising=False)

    def _boom():
        raise OSError("no username available")

    monkeypatch.setattr("loopward.engine.stop_gate.getpass.getuser", _boom)
    d = StopGate(mode="auto").request("verify", "s")
    assert d.approved is True
    assert d.approver is None
    assert d.approver_source == "unknown"
