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
    llm = LLMClient(provider="fake", fake_script=lambda msgs, task: "HIGH: from callable")
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
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "LOW: ok")
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


# ---- the provider's stop reason crosses the wrapper -------------------------
# Both real providers report why they stopped; both values used to be dropped
# here, which made a truncated reply indistinguishable from a complete one
# everywhere upstream. Carried raw, recorded, and consulted by nothing.


class _AnthropicBlock:
    """One content block, shaped like anthropic's TextBlock."""

    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _AnthropicResp:
    def __init__(self, blocks, stop_reason=None, with_stop_reason=True):
        self.content = blocks
        if with_stop_reason:
            self.stop_reason = stop_reason


def _claude_client(resp, seen=None):
    """A stand-in for anthropic.Anthropic that records the kwargs it was given."""

    class _Messages:
        @staticmethod
        def create(**kwargs):
            if seen is not None:
                seen.update(kwargs)
            return resp

    class _FakeClient:
        messages = _Messages()

    return _FakeClient()


@pytest.mark.unit
def test_claude_provider_mocked(monkeypatch):
    """The claude route had no mocked test at all; it has one now.

    Asserts the whole shape this wrapper promises for that provider: text is
    joined from the text blocks only, the system message is lifted out of
    `messages` into `system`, and `stop_reason` comes back on the response.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = LLMClient(provider="claude", model="claude-haiku-4-5")
    seen = {}
    llm._client = _claude_client(  # bypass lazy SDK construction
        _AnthropicResp(
            [
                _AnthropicBlock("ignored", type="thinking"),
                _AnthropicBlock("HIGH: mocked finding"),
            ],
            stop_reason="end_turn",
        ),
        seen,
    )

    out = llm.complete(
        [
            {"role": "system", "content": "be strict"},
            {"role": "user", "content": "review this"},
        ]
    )

    assert out.text == "HIGH: mocked finding"  # the non-text block is dropped
    assert out.provider == "claude"
    assert out.model == "claude-haiku-4-5"
    assert out.stop_reason == "end_turn"
    assert seen["system"] == "be strict"
    assert seen["messages"] == [{"role": "user", "content": "review this"}]


@pytest.mark.unit
def test_claude_truncation_is_reported_not_swallowed(monkeypatch):
    """The case this whole change exists for: a truncated reply that still parses.

    `max_tokens` arrives on a perfectly ordinary HTTP 200 with usable-looking
    text. Nothing about the text says it was cut off; the stop reason is the
    only thing that does.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = LLMClient(provider="claude")
    llm._client = _claude_client(
        _AnthropicResp([_AnthropicBlock("HIGH: off-by-one in the loop bo")], "max_tokens")
    )

    assert llm.complete([{"role": "user", "content": "x"}]).stop_reason == "max_tokens"


@pytest.mark.unit
def test_claude_stop_reason_read_defensively(monkeypatch):
    """`pyproject` floors the extra at anthropic>=0.39; the field may not exist.

    A response object without `stop_reason` must yield None, not AttributeError.
    The SDK the repo runs against (1.1.0) grew `model_context_window_exceeded`
    since that floor, so "the field is there and its values are known" is not
    something this wrapper may assume.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = LLMClient(provider="claude")
    llm._client = _claude_client(
        _AnthropicResp([_AnthropicBlock("HIGH: x")], with_stop_reason=False)
    )

    assert llm.complete([{"role": "user", "content": "x"}]).stop_reason is None


def _deepseek_client(content, finish_reason=None, with_finish_reason=True):
    class _Msg:
        pass

    class _Choice:
        pass

    msg, choice = _Msg(), _Choice()
    msg.content = content
    choice.message = msg
    if with_finish_reason:
        choice.finish_reason = finish_reason

    class _Resp:
        choices = [choice]

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(model, messages):
                    return _Resp()

    return _FakeClient()


@pytest.mark.unit
def test_deepseek_finish_reason_is_carried_raw():
    """`length`, not `max_tokens`. The OpenAI-compatible word, untranslated.

    Normalising the two vocabularies would mean deciding they are the same
    thing, which is a mapping decision this layer does not make. The two value
    sets are disjoint, so the raw string identifies its own provider.
    """
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    llm._client = _deepseek_client("HIGH: truncated her", "length")

    out = llm.complete([{"role": "user", "content": "x"}])
    assert out.stop_reason == "length"


@pytest.mark.unit
def test_deepseek_finish_reason_read_defensively():
    """Two reasons at once, and either alone would justify the `getattr`.

    `openai>=1.40` is a floor, so an older client object may not carry the
    field; and the value on the wire comes from DeepSeek's server, not from
    openai's type, which describes what OpenAI sends. The pre-existing
    `test_deepseek_provider_mocked` stub has no `finish_reason` either — if it
    ever goes red on an AttributeError, the read stopped being defensive.
    """
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    llm._client = _deepseek_client("HIGH: x", with_finish_reason=False)

    assert llm.complete([{"role": "user", "content": "x"}]).stop_reason is None


@pytest.mark.unit
def test_fake_reports_no_stop_reason():
    """None, never `end_turn`.

    The offline provider does not know why it stopped, and claiming a clean
    finish would be it asserting a fact only a real provider can establish --
    the same class of untruth the rest of this package keeps closing.
    """
    llm = LLMClient(provider="fake", fake_script=["HIGH: a"])
    assert llm.complete([{"role": "user", "content": "x"}]).stop_reason is None
    assert llm.stop_reasons == [None]


@pytest.mark.unit
def test_stop_reasons_accumulate_in_call_order():
    """Ordered, not counted and not de-duplicated.

    A count is already a summary and a set loses how many; the order is what
    lets a trail reader line a truncation up against the attempt that caused it.
    """
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    llm._client = _deepseek_client("HIGH: a", "stop")
    llm.complete([{"role": "user", "content": "x"}])
    llm._client = _deepseek_client("HIGH: b", "length")
    llm.complete([{"role": "user", "content": "x"}])
    llm._client = _deepseek_client("HIGH: c", "stop")
    llm.complete([{"role": "user", "content": "x"}])

    assert llm.stop_reasons == ["stop", "length", "stop"]


@pytest.mark.unit
def test_stop_reasons_is_a_copy():
    llm = LLMClient(provider="fake", fake_script=["HIGH: a"])
    llm.complete([{"role": "user", "content": "x"}])
    llm.stop_reasons.append("forged")
    assert llm.stop_reasons == [None]


@pytest.mark.unit
def test_a_reason_survives_the_empty_completion_guard():
    """The asymmetry is the point, not an accounting slip.

    A provider that truncates at zero tokens reports `length` and returns "".
    That pair is the most informative thing the wrapper ever sees and it is also
    what ends the run, so the reason is recorded *before* the guard raises.
    `calls` is still incremented after it, so the two counts differ by exactly
    the number of blank completions — and the reason is not lost.
    """
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    llm._client = _deepseek_client("", "length")

    with pytest.raises(EmptyCompletionError):
        llm.complete([{"role": "user", "content": "x"}])

    assert llm.stop_reasons == ["length"]
    assert llm.totals["calls"] == 0
