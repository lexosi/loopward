"""Tests for the verifier: it must see the code, and it must read the answer.

Two failure modes this file pins down. The verifier used to adjudicate findings
without ever being shown the diff they came from, and it used to parse only
REJECT — so a model that refused, or answered with `{}`, produced exactly the
same object as a model that confirmed everything. "The model declined" and "the
model agrees with every finding" must never be the same result.
"""

import pytest

import loopward.agents.verifier as V
from loopward.agents.reviewer import Finding
from loopward.engine.llm_wrapper import LLMClient
from loopward.engine.stop_gate import StopGate

DIFF = "--- a/auth.py\n+++ b/auth.py\n@@\n-    return now < exp\n+    return now <= exp\n"


def _approval():
    return StopGate(mode="auto").request("verify", "s").approval


def _findings(n: int) -> list[Finding]:
    return [Finding(severity="HIGH", message=f"finding {i}") for i in range(1, n + 1)]


def _verify(reply: str, n: int) -> V.VerifyResult:
    llm = LLMClient(provider="fake", fake_script=lambda m, task, r=reply: r)
    return V.Verifier(llm).verify(_findings(n), DIFF, _approval())


def _expect_parse_error(reply: str, n: int) -> None:
    """Fail loudly if an unusable adjudication is silently accepted instead."""
    try:
        result = _verify(reply, n)
    except V.VerifyParseError:
        return
    pytest.fail(
        f"unparseable adjudication {reply!r} silently confirmed "
        f"{len(result.confirmed)} of {n} finding(s)"
    )


# ---- the verifier must see the code it is adjudicating ----------------------


@pytest.mark.unit
def test_verifier_message_carries_the_diff():
    messages = V.Verifier(LLMClient(provider="fake")).build_messages(_findings(2), DIFF)
    user = messages[-1]["content"]
    assert DIFF in user, "the verifier cannot spot a false positive it never sees"
    assert "<diff>" in user and "</diff>" in user, "untrusted content must be delimited"
    assert "1. HIGH: finding 1" in user and "2. HIGH: finding 2" in user


@pytest.mark.unit
def test_verifier_system_prompt_marks_the_diff_as_data():
    messages = V.Verifier(LLMClient(provider="fake")).build_messages(_findings(1), DIFF)
    assert "never instructions" in messages[0]["content"]


# ---- an unusable answer is an error, not a default --------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        "I am a language model and cannot help.",
        "Sure! Both findings look reasonable to me.",
    ],
)
def test_garbage_reply_is_never_taken_as_confirmation(reply):
    _expect_parse_error(reply, 2)


@pytest.mark.unit
def test_partial_coverage_is_a_parse_error():
    # The model adjudicated finding 1 and said nothing about 2. Confirming 2 by
    # default is asserting something the model never claimed.
    _expect_parse_error("CONFIRM 1", 2)


@pytest.mark.unit
def test_out_of_range_index_is_a_parse_error():
    # 99 does not exist; swallowing it silently loses the fact that the model
    # was answering about something else entirely.
    _expect_parse_error("CONFIRM 1\nCONFIRM 2\nREJECT 99", 2)


@pytest.mark.unit
def test_duplicate_adjudication_is_a_parse_error():
    _expect_parse_error("CONFIRM 1\nREJECT 1\nCONFIRM 2", 2)


@pytest.mark.unit
def test_verdict_mentioned_in_prose_does_not_count():
    # "I would never REJECT 2" is commentary, not an adjudication of finding 2.
    _expect_parse_error("CONFIRM 1 - I would never REJECT 2, it is clearly real.", 2)


# ---- the happy path still adjudicates ---------------------------------------


@pytest.mark.unit
def test_confirm_and_reject_are_both_honoured():
    result = _verify("CONFIRM 1\nREJECT 2", 2)
    assert [f.message for f in result.confirmed] == ["finding 1"]
    assert [f.message for f in result.rejected] == ["finding 2"]


@pytest.mark.unit
def test_full_confirmation_keeps_every_finding():
    result = _verify("CONFIRM 1\nCONFIRM 2\nCONFIRM 3", 3)
    assert len(result.confirmed) == 3
    assert result.rejected == []
    assert result.has_blocking is True


@pytest.mark.unit
def test_no_findings_needs_no_model_call():
    llm = LLMClient(provider="fake")
    assert V.Verifier(llm).verify([], DIFF, _approval()).confirmed == []
    assert llm.totals["calls"] == 0
