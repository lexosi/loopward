"""The provider's stop reason reaches the trail — and decides nothing.

Both real providers say why they stopped generating: Anthropic as `stop_reason`
(`end_turn`, `max_tokens`, `refusal`, `model_context_window_exceeded`, ...),
OpenAI-compatible ones as `finish_reason` (`stop`, `length`, `content_filter`,
...). Both used to be discarded inside `llm_wrapper`, which made a truncated
reply indistinguishable from a complete one everywhere upstream — a silent
failure of the same class as the ones this package has been closing.

The signal now crosses the wrapper, accumulates on the client, and is folded
into the run summary. What it does **not** do is decide anything. That is the
deliberate half, and the tests below are written to fail if it ever changes:
mapping a stop reason to an action is the next piece of work, not this one, and
a branch added here quietly would move the benchmark and the demo with it.
"""

import json
import pathlib

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import TASK_VERIFY, LLMClient
from loopward.engine.orchestrator import STATUS_CRASHED, STATUS_OK, Orchestrator
from loopward.engine.stop_gate import StopGate

DIFF = "return now <= self.expires_at"


def _trail(run_dir):
    return json.loads((pathlib.Path(run_dir) / "audit.json").read_text(encoding="utf-8"))


def _scripted_deepseek(replies):
    """A stubbed OpenAI-compatible client: `(content, finish_reason)` per call.

    The fake provider cannot express a stop reason — `FakeScript` is still
    `-> str` and the fake reports None — so anything about a *real* reason has
    to go through a stubbed real provider. Still offline, still no key.
    """
    state = {"i": 0}

    class _Resp:
        def __init__(self, content, finish_reason):
            msg = type("_Msg", (), {"content": content})()
            self.choices = [type("_Choice", (), {"message": msg, "finish_reason": finish_reason})()]

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(model, messages):
                    content, finish = replies[min(state["i"], len(replies) - 1)]
                    state["i"] += 1
                    return _Resp(content, finish)

    return _FakeClient()


def _deepseek_orch(audit, replies, gate=None):
    llm = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    llm._client = _scripted_deepseek(replies)  # bypass lazy SDK construction
    return Orchestrator(llm, gate or StopGate(mode="auto"), audit=audit)


# --- the signal reaches the trail -------------------------------------------


@pytest.mark.integration
def test_the_envelope_carries_one_reason_per_call(tmp_path):
    """Ordered and one-per-call, so a reader can tell which call was odd.

    The fake reports None for every call, and `null` is the honest rendering of
    "the provider said nothing" — not a stand-in for a clean finish.
    """
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    llm = LLMClient(
        provider="fake",
        fake_script=lambda m, task: "CONFIRM 1" if task == TASK_VERIFY else "HIGH: x",
    )
    result = Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)

    reasons = _trail(result.run_dir)["summary"]["provider_stop_reasons"]
    assert reasons == [None, None]  # one review call, one verify call
    assert len(reasons) == llm.totals["calls"]


@pytest.mark.integration
def test_a_truncation_is_visible_in_the_envelope(tmp_path):
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = _deepseek_orch(
        audit, [("HIGH: off-by-one in the loop bo", "length"), ("CONFIRM 1", "stop")]
    )
    result = orch.run(DIFF)

    assert _trail(result.run_dir)["summary"]["provider_stop_reasons"] == ["length", "stop"]


@pytest.mark.integration
def test_the_crashed_path_folds_the_reasons_too(tmp_path):
    """The path where the signal matters most, which is why it is folded twice.

    A run that ends because the provider truncated or refused is exactly the run
    whose trail would otherwise say nothing about it. `_fold_provider_stop_reasons`
    runs in `_finalize_crashed` as well as on the normal exit.
    """

    def closed_stdin(phase, summary):
        raise EOFError("EOF when reading a line")

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = _deepseek_orch(
        audit,
        [("HIGH: a finding", "length")],
        gate=StopGate(mode="interactive", prompter=closed_stdin),
    )

    with pytest.raises(EOFError):
        orch.run(DIFF)

    envelope = _trail(audit.run_dir_on_disk)
    assert envelope["summary"]["status"] == STATUS_CRASHED
    assert envelope["summary"]["provider_stop_reasons"] == ["length"]


@pytest.mark.integration
def test_audit_md_shows_them_too(tmp_path):
    """`audit.md` is the file a human opens; the signal has to be in it.

    It renders only ts/kind/message for events, so a summary-level value is
    the only way this reaches a markdown reader at all.
    """
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    result = _deepseek_orch(
        audit, [("HIGH: cut off her", "length"), ("CONFIRM 1", "stop")]
    ).run(DIFF)

    md = (pathlib.Path(result.run_dir) / "audit.md").read_text(encoding="utf-8")
    assert "- **provider_stop_reasons**: length, stop" in md


@pytest.mark.unit
def test_the_two_stop_reasons_in_the_trail_are_named_apart():
    """The trail already had a `stop_reason`: the anti-loop's, in event data.

    Different thing entirely — why the review *loop* cut itself short. The new
    one is keyed `provider_stop_reasons` precisely so a reader is never asked to
    work out which of the two a bare `stop_reason` meant.
    """
    audit = AuditLog(run_id="test")
    audit.record_provider_stop_reasons(["length"])
    audit.record("anti_loop", "stopped", stop_reason="strategies_exhausted")

    summary = audit.summary("exhausted", "r")
    assert "provider_stop_reasons" in summary
    assert "stop_reason" not in summary
    assert audit.events[0].data["stop_reason"] == "strategies_exhausted"


# --- and decides nothing ----------------------------------------------------


@pytest.mark.integration
def test_a_truncated_reply_that_still_parses_still_succeeds(tmp_path):
    """THE acceptance test. Signal available, decision untouched.

    The reply was cut off mid-line: one line still matches the finding format,
    the partial one does not and is dropped. Today that is a `ok` run with one
    finding, and after this change it is *still* an `ok` run with one finding —
    the only difference is that the trail now says `length` beside it.

    Making that outcome better is the mapping work. Making it *different* here
    would move the benchmark and the demo, which is why this test exists.
    """
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = _deepseek_orch(
        audit,
        [("HIGH: off-by-one in the loop bound.\nMEDIU", "length"), ("CONFIRM 1", "stop")],
    )

    result = orch.run(DIFF)

    assert result.status == STATUS_OK
    assert [str(f) for f in result.findings] == ["HIGH: off-by-one in the loop bound."]
    assert result.verify is not None and len(result.verify.confirmed) == 1

    envelope = _trail(result.run_dir)
    assert envelope["summary"]["provider_stop_reasons"] == ["length", "stop"]
    # The dropped partial line is counted, as it already was — and still by the
    # parser, not by the stop reason.
    parses = [e for e in envelope["events"] if e["kind"] == "review_parse"]
    assert parses and parses[0]["data"]["dropped"] == 1


@pytest.mark.integration
def test_the_run_is_byte_identical_whatever_the_reason_says(tmp_path):
    """The strongest form of "nothing branches on it".

    Two runs, same replies, different stop reasons — one ordinary, one a
    truncation. Every event and the status must come out the same. If a branch
    is ever added that reads the reason, this goes red before the demo or the
    benchmark notice.
    """

    def events_for(reason, base):
        audit = AuditLog(run_id="test", base_dir=base)
        result = _deepseek_orch(
            audit, [("HIGH: a finding here", reason), ("CONFIRM 1", reason)]
        ).run(DIFF)
        envelope = _trail(result.run_dir)
        return result.status, [(e["kind"], e["message"]) for e in envelope["events"]]

    ordinary = events_for("stop", tmp_path / "a")
    truncated = events_for("length", tmp_path / "b")

    assert ordinary == truncated


@pytest.mark.integration
def test_reasons_that_cannot_be_folded_do_not_cost_the_trail(tmp_path, capsys):
    """`llm` is a public parameter, so reading `stop_reasons` is a call out.

    Same invariant as `_fold_usage`, and a separate guard for it on purpose:
    one of the two folds failing must not cost the other. Here the reasons are
    lost and the token totals still land, inside a trail that exists.
    """

    class _ReasonsBackendDown(LLMClient):
        @property
        def stop_reasons(self):
            raise RuntimeError("reason backend down")

    llm = _ReasonsBackendDown(
        provider="fake",
        fake_script=lambda m, task: "CONFIRM 1" if task == TASK_VERIFY else "HIGH: x",
    )
    audit = AuditLog(run_id="test", base_dir=tmp_path)

    result = Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)

    assert result.status == STATUS_OK
    envelope = _trail(result.run_dir)
    assert envelope["summary"]["provider_stop_reasons"] == []
    assert envelope["summary"]["tokens"]["total"] > 0  # the other fold was untouched
    assert "stop reasons" in capsys.readouterr().err
