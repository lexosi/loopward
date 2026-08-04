"""Tests for the LLM wrapper (fake provider + mocked real providers, no network)."""

import pytest

from loopward.engine.llm_wrapper import (
    EmptyCompletionError,
    LLMClient,
    _calc_cost,
)


@pytest.mark.unit
def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        LLMClient(provider="gpt")


@pytest.mark.unit
def test_fake_script_list_consumed_in_order_then_clamped():
    llm = LLMClient(provider="fake", fake_script=["HIGH: a", "LOW: b"])
    assert llm.complete([{"role": "user", "content": "x"}]).text == "HIGH: a"
    assert llm.complete([{"role": "user", "content": "x"}]).text == "LOW: b"
    # exhausted -> clamps to last
    assert llm.complete([{"role": "user", "content": "x"}]).text == "LOW: b"


@pytest.mark.unit
def test_fake_callable_script():
    llm = LLMClient(provider="fake", fake_script=lambda msgs: "HIGH: from callable")
    assert "from callable" in llm.complete([{"role": "user", "content": "x"}]).text


@pytest.mark.unit
def test_fake_heuristic_detects_planted_smell():
    llm = LLMClient(provider="fake")
    msg = [{"role": "user", "content": "return now <= self.expires_at  # expiry"}]
    assert "HIGH" in llm.complete(msg).text


@pytest.mark.unit
def test_empty_completion_raises():
    llm = LLMClient(provider="fake", fake_script=["   "])
    with pytest.raises(EmptyCompletionError):
        llm.complete([{"role": "user", "content": "x"}])


@pytest.mark.unit
def test_totals_accumulate():
    llm = LLMClient(provider="fake", fake_script=lambda m: "LOW: ok")
    llm.complete([{"role": "user", "content": "hello world"}])
    llm.complete([{"role": "user", "content": "again"}])
    t = llm.totals
    assert t["calls"] == 2
    assert t["prompt"] > 0 and t["completion"] > 0


@pytest.mark.unit
def test_cost_zero_for_unknown_model():
    assert _calc_cost("fake", "fake-1", 1000, 1000) == 0.0


@pytest.mark.unit
def test_cost_nonzero_for_known_model():
    cost = _calc_cost("deepseek", "deepseek-v4-flash", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.14 + 0.28)


@pytest.mark.unit
def test_deepseek_provider_mocked(monkeypatch):
    """No real network: inject a fake client and key."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")

    class _Msg:
        content = "HIGH: mocked finding"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(model, messages):
                    return _Resp()

    llm._client = _FakeClient()  # bypass lazy SDK construction
    out = llm.complete([{"role": "user", "content": "x"}])
    assert out.text == "HIGH: mocked finding"
    assert out.provider == "deepseek"
