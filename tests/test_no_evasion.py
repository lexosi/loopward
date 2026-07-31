"""No-evasion tests — the stop-gate and anti-loop are STRUCTURAL, not advisory.

guardloop's headline is "you cannot skip the gate". These tests are the proof:
a gated phase is unreachable without a real capability token, and the tokens
(Approval, AttemptOutcome) can only be minted by the gate / the tracker.
"""

import pytest

from guardloop.agents.reviewer import Finding
from guardloop.agents.verifier import Verifier
from guardloop.engine.anti_loop import (
    CLASS_JUMP,
    AttemptOutcome,
    AttemptTracker,
    is_genuine_outcome,
)
from guardloop.engine.llm_wrapper import LLMClient
from guardloop.engine.stop_gate import Approval, StopGate, is_genuine_approval


def _verifier() -> Verifier:
    return Verifier(LLMClient(provider="fake"))


def _findings() -> list[Finding]:
    return [Finding(severity="HIGH", message="token expiry uses <=")]


# ---- F1: stop-gate Approval token -------------------------------------------

@pytest.mark.unit
def test_verify_without_approval_is_typeerror():
    # NEGATIVE CONTROL: no approval at all -> missing-arg TypeError, cannot run.
    with pytest.raises(TypeError):
        _verifier().verify(_findings())


@pytest.mark.unit
def test_verify_with_forged_nonapproval_is_rejected():
    with pytest.raises(TypeError):
        _verifier().verify(_findings(), object())


@pytest.mark.unit
def test_approval_missing_mint_is_typeerror():
    with pytest.raises(TypeError):
        Approval("verify")


@pytest.mark.unit
def test_approval_forged_mint_is_permissionerror():
    with pytest.raises(PermissionError):
        Approval("verify", _mint=object())


@pytest.mark.unit
def test_denied_gate_has_no_approval_and_cannot_verify():
    decision = StopGate(mode="deny").request("verify", "s")
    assert decision.approved is False
    assert decision.approval is None
    with pytest.raises(TypeError):
        _verifier().verify(_findings(), decision.approval)


@pytest.mark.unit
def test_only_gate_minted_approval_proceeds():
    decision = StopGate(mode="auto").request("verify", "s")
    assert isinstance(decision.approval, Approval)
    result = _verifier().verify(_findings(), decision.approval)
    assert result.confirmed  # proceeds only with the real token


# ---- F2: anti-loop AttemptOutcome token -------------------------------------

@pytest.mark.unit
def test_attemptoutcome_hand_constructed_is_rejected():
    with pytest.raises(PermissionError):
        AttemptOutcome("sub", 3, CLASS_JUMP, "regex", "forged")


@pytest.mark.unit
def test_attemptoutcome_forged_mint_is_rejected():
    with pytest.raises(PermissionError):
        AttemptOutcome("sub", 3, CLASS_JUMP, "regex", "forged", _mint=object())


@pytest.mark.unit
def test_only_tracker_mints_class_jump_grant():
    tracker = AttemptTracker(max_attempts=3)
    tracker.record_failure("sub", strategy="regex")
    tracker.record_failure("sub", strategy="regex")
    outcome = tracker.record_failure("sub", strategy="regex")
    assert outcome.must_class_jump is True


# ---- F2: agents are not the public surface ----------------------------------

@pytest.mark.unit
def test_public_entry_is_orchestrator_not_agents():
    import guardloop
    import guardloop.agents as agents

    assert hasattr(guardloop, "Orchestrator")
    assert "Reviewer" not in agents.__all__
    assert "Verifier" not in agents.__all__


# ---- Adversarial forgery: the bypass a prior review found -------------------
# These prove the tokens are object-capabilities (identity-registry membership),
# not isinstance checks: a subclass or an object.__new__ instance is rejected.

@pytest.mark.unit
def test_approval_cannot_be_subclassed():
    with pytest.raises(TypeError):
        type("FakeApproval", (Approval,), {})


@pytest.mark.unit
def test_verify_rejects_object_new_forged_approval():
    forged = object.__new__(Approval)  # skips __init__, never registered
    with pytest.raises(TypeError):
        _verifier().verify(_findings(), forged)


@pytest.mark.unit
def test_object_new_approval_is_not_genuine():
    forged = object.__new__(Approval)
    assert is_genuine_approval(forged) is False


@pytest.mark.unit
def test_genuine_approval_is_genuine():
    decision = StopGate(mode="auto").request("verify", "s")
    assert is_genuine_approval(decision.approval) is True


@pytest.mark.unit
def test_approval_is_immutable():
    decision = StopGate(mode="auto").request("verify", "s")
    with pytest.raises(AttributeError):
        decision.approval.phase = "hacked"


@pytest.mark.unit
def test_attemptoutcome_cannot_be_subclassed():
    with pytest.raises(TypeError):
        type("FakeOutcome", (AttemptOutcome,), {})


@pytest.mark.unit
def test_object_new_attemptoutcome_is_not_genuine():
    forged = object.__new__(AttemptOutcome)
    assert is_genuine_outcome(forged) is False


@pytest.mark.unit
def test_genuine_outcome_is_genuine():
    tracker = AttemptTracker(max_attempts=3)
    outcome = None
    for _ in range(3):
        outcome = tracker.record_failure("sub", strategy="regex")
    assert is_genuine_outcome(outcome) is True
