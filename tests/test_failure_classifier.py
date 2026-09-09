"""A wire signal becomes a failure class, and the failure class steers the run.

The crash path classifies a failed run and, when it cannot, records the raw wire
signal instead of swallowing it. The one measured class
(``context_length_exceeded``) is now consumed by the review loop's upstream
capture point: it reviews the diff a file at a time and the run ends
``partial_review`` (see ``test_the_measured_failure_now_routes_to_a_per_file_review``
below, and ``tests/test_chunk_route.py`` for the mechanism). Everything the table
does not place stays UNKNOWN: it re-raises, finalizes ``crashed``, and leaves its
raw signal on the trail.

Detection is by WIRE SIGNAL, never by exception class: the anthropic 1.4.0
``ValueError("Streaming is required...")`` proves the class is not stable, so the
classifier reads status/body/message off the exception and matches a data table.
"""

import json
import pathlib
import re

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.failure_classifier import (
    CHUNK_DIFF_STRATEGY,
    CONTEXT_LENGTH_EXCEEDED,
    NEXT_STRATEGY,
    UNKNOWN,
    classify,
    extract_signal,
)
from loopward.engine.llm_wrapper import LLMClient
from loopward.engine.orchestrator import STATUS_CRASHED, Orchestrator
from loopward.engine.stop_gate import StopGate

#: The overflow message shape anthropic returns on a context-window 400,
#: measured 2026-09-09. The literal number is illustrative; the phrase is the
#: signal.
OVERFLOW_MSG = "prompt is too long: 250000 tokens > 200000 maximum"


class _StubAnthropic400(Exception):
    """Duck-types the fields ``extract_signal`` reads off a real BadRequestError.

    The real one is ``anthropic.BadRequestError``; constructing it needs an
    ``httpx`` response, so only the wire-relevant attributes are stubbed here.
    ``.status_code``, ``.type`` and ``.message`` are exactly what the SDK sets —
    see ``anthropic._exceptions.APIStatusError.__init__``: ``.type`` is the
    parsed ``body["error"]["type"]``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400
        self.type = "invalid_request_error"
        self.message = message


def _trail(run_dir: str) -> dict:
    return json.loads((pathlib.Path(run_dir) / "audit.json").read_text(encoding="utf-8"))


def _resp(text: str, stop_reason: str | None = None):
    """A minimal Anthropic-shaped response: ``.content`` blocks + ``.stop_reason``."""
    block = type("_Block", (), {"type": "text", "text": text})()
    return type("_Resp", (), {"content": [block], "stop_reason": stop_reason})()


# --- the table and the map: one measured entry each -------------------------


@pytest.mark.unit
def test_the_table_has_exactly_one_entry_and_it_is_the_measured_class():
    from loopward.engine import failure_classifier as fc

    assert len(fc._TABLE) == 1
    assert fc._TABLE[0].failure_class == CONTEXT_LENGTH_EXCEEDED


@pytest.mark.unit
def test_the_map_has_one_entry_pointing_at_the_declared_destination():
    # Shape only: exactly one class -> one destination. Whether that destination
    # is a built strategy yet off the ordered list is the tripwire's job, below.
    assert NEXT_STRATEGY == {CONTEXT_LENGTH_EXCEEDED: CHUNK_DIFF_STRATEGY}


@pytest.mark.unit
def test_chunk_diff_is_built_but_stays_off_the_ordered_strategy_list():
    """GUARD: the destination is built as an agent, yet is NOT an ordered strategy.

    Two mechanisms, deliberately kept apart. The ordered list (``STRATEGIES`` +
    ``_STRATEGY_INSTRUCTIONS``) is traversed on exhaustion and its length sets
    the anti-loop's hard bound. This map reaches its destination by a MEASURED
    failure class instead, so its destination must NOT be a member of that list:
    if it were, ``len(STRATEGIES)`` would grow and the benchmark's bound of 6
    would move.

    So this holds two things at once:

    - The destination RESOLVES TO A BUILT STRATEGY — a real prompt
      (``CHUNK_DIFF_INSTRUCTION``, whose own example the reviewer's parser reads)
      and a real splitter (``split_diff``) exist in ``loopward.agents.chunk_diff``.
    - The destination is STILL NOT a member of the ordered list, nor a key in the
      ordered-list prompt table.

    It goes RED the day someone folds chunk-diff into ``STRATEGIES`` (raising the
    bound) — the earlier version of this test asserted chunk-diff was not built
    at all, a premise the real design makes false.
    """
    from loopward.agents.chunk_diff import CHUNK_DIFF_INSTRUCTION, split_diff
    from loopward.agents.reviewer import _STRATEGY_INSTRUCTIONS, STRATEGIES, parse_findings

    destination = NEXT_STRATEGY[CONTEXT_LENGTH_EXCEEDED]

    # (1) resolves to a built strategy: a prompt that honours the format contract
    #     and a splitter that exists.
    assert CHUNK_DIFF_INSTRUCTION.strip()
    assert len(parse_findings(CHUNK_DIFF_INSTRUCTION.splitlines()[-1])) == 1
    assert callable(split_diff)

    # (2) and still off the ordered list, and out of its prompt table.
    assert destination not in STRATEGIES
    assert destination not in _STRATEGY_INSTRUCTIONS


@pytest.mark.unit
def test_the_destination_is_built_elsewhere_and_the_reviewer_still_rejects_it():
    """chunk-diff is built as its own agent, not as a reviewer strategy — so the
    reviewer must still refuse it by name.

    ``reviewer.build_messages`` once resolved an unknown strategy to 'concise'
    via ``_STRATEGY_INSTRUCTIONS.get(strategy, default)`` with NO signal. That
    was the danger this map poses: point the review loop at this destination and
    it would silently run as 'concise' while the trail recorded a class-jump that
    did not happen — a certificate over nothing.

    That is closed upstream of any wiring. chunk-diff now exists (in
    ``loopward.agents.chunk_diff``), but it is NOT one of the reviewer's ordered
    strategies, so the reviewer has no prompt for it and raises
    ``UnknownStrategyError`` with the strategy named. Built as an agent, unknown
    to the reviewer: it cannot slip through as concise behind the trail's back.
    """
    from loopward.agents.reviewer import Reviewer, UnknownStrategyError

    llm = LLMClient(provider="fake", fake_script=lambda m, task: "HIGH: x")
    reviewer = Reviewer(llm)
    destination = NEXT_STRATEGY[CONTEXT_LENGTH_EXCEEDED]

    with pytest.raises(UnknownStrategyError) as exc_info:
        reviewer.build_messages("diff", destination)
    # The cause names the strategy that has no prompt, not a silent concise.
    assert destination in str(exc_info.value)


# --- the one measured signal classifies; everything else is UNKNOWN ---------


@pytest.mark.unit
def test_the_measured_anthropic_overflow_is_context_length_exceeded():
    assert classify(_StubAnthropic400(OVERFLOW_MSG), provider="claude") == CONTEXT_LENGTH_EXCEEDED


@pytest.mark.unit
def test_same_shape_from_a_different_provider_is_not_the_measured_class():
    # Only anthropic was measured. The same 400 shape from deepseek is UNKNOWN —
    # not a class this repo has not measured for that provider.
    assert classify(_StubAnthropic400(OVERFLOW_MSG), provider="deepseek") == UNKNOWN


@pytest.mark.unit
def test_the_anthropic_1_4_0_streaming_valueerror_is_unknown():
    # The measured proof that the exception class is not stable: a plain
    # ValueError with no status -> UNKNOWN, and its literal message is preserved.
    exc = ValueError("Streaming is required for operations that may take longer than 10 minutes.")
    assert classify(exc, provider="claude") == UNKNOWN
    sig = extract_signal(exc, provider="claude")
    assert sig.http_status is None
    assert sig.body_type is None
    assert sig.message == (
        "Streaming is required for operations that may take longer than 10 minutes."
    )


@pytest.mark.unit
def test_a_400_without_the_overflow_message_is_unknown():
    # 400 + invalid_request_error but a different message (a bad param, say) is
    # not an overflow. No unmeasured class is invented for it.
    exc = _StubAnthropic400("messages: at least one message is required")
    assert classify(exc, provider="claude") == UNKNOWN


@pytest.mark.unit
def test_classify_never_raises_on_a_hostile_exception():
    # classify runs inside _finalize_crashed, which must never raise. A field
    # whose read blows up must degrade to UNKNOWN, not propagate.
    class _Nasty(Exception):
        @property
        def status_code(self):  # noqa: D401
            raise RuntimeError("boom")

    assert classify(_Nasty("x"), provider="claude") == UNKNOWN


# --- "never raises" must not mean "swallows in silence" ----------------------
# A classifier that returns UNKNOWN for a malformed exception AND leaves no
# trace of what it could not read is an `except: pass` with good manners — the
# exact defect this repo exists to catch. Every malformed shape below must do
# BOTH: return UNKNOWN, and record on the signal which wire fields it could not
# resolve.


def _exc_with(**attrs):
    exc = Exception("malformed")
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


MALFORMED = [
    ("no .type and no body", Exception("plain exception, nothing to read")),
    ("body is None", _exc_with(body=None)),
    ("body is not a dict", _exc_with(body="not a dict")),
    ("body dict without 'error'", _exc_with(body={"note": "no error key here"})),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,exc", MALFORMED, ids=[c[0] for c in MALFORMED])
def test_extract_signal_records_what_it_could_not_read(label, exc):
    # (a) it classifies as UNKNOWN
    assert classify(exc, provider="claude") == UNKNOWN
    # (b) it leaves a trace of WHICH fields it could not resolve
    sig = extract_signal(exc, provider="claude")
    assert sig.unresolved, "a malformed exception left no trace of what was unreadable"
    joined = " | ".join(sig.unresolved)
    assert "http_status" in joined
    assert "body_type" in joined


@pytest.mark.unit
def test_a_read_that_raises_is_marked_unreadable_not_merely_absent():
    # The anti-swallow proof: a field whose read RAISED must not be collapsed
    # into the same silent None as a field that was simply absent. The trace
    # must say the read failed, and with what.
    class _StatusRaises(Exception):
        @property
        def status_code(self):  # noqa: D401
            raise RuntimeError("backend hiccup")

    exc = _StatusRaises("x")
    assert classify(exc, provider="claude") == UNKNOWN
    notes = " | ".join(extract_signal(exc, provider="claude").unresolved)
    assert "http_status" in notes
    assert "unreadable" in notes and "RuntimeError" in notes


# --- the crash path records the class, and UNKNOWN records the raw signal ----


@pytest.mark.integration
def test_an_unknown_failure_leaves_its_raw_signal_in_the_trail(tmp_path):
    """An unclassifiable failure is never swallowed and never lands in a strategy.

    The crash path classifies it UNKNOWN, records the raw wire signal
    (http_status, body_type, message), and still does exactly what it did before:
    finalize crashed and re-raise the original.
    """

    def boom(messages, task):
        raise ValueError(
            "Streaming is required for operations that may take longer than 10 minutes."
        )

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = Orchestrator(
        LLMClient(provider="fake", fake_script=boom), StopGate(mode="auto"), audit=audit
    )

    with pytest.raises(ValueError, match="Streaming is required"):
        orch.run("some diff")

    env = _trail(audit.run_dir_on_disk)
    assert env["summary"]["status"] == STATUS_CRASHED
    # Never landed in a strategy: no attempt, no class-jump happened.
    assert not any(e["kind"] == "class_jump" for e in env["events"])

    result = [e for e in env["events"] if e["kind"] == "result"][-1]
    assert result["data"]["failure_class"] == UNKNOWN
    sig = result["data"]["wire_signal"]
    assert sig["http_status"] is None
    assert sig["body_type"] is None
    assert "Streaming is required" in sig["message"]
    # The trace reaches the trail: what could not be read is named, not silent.
    assert sig["unresolved"], "the trail's wire_signal swallowed what it could not read"


@pytest.mark.integration
def test_the_measured_failure_now_routes_to_a_per_file_review(tmp_path):
    """The measured class is CONSUMED here: it reviews the diff a file at a time.

    The review loop's capture point is where the map's one destination is finally
    acted on. A context-length
    400 on the whole-diff review no longer crashes; it splits the diff and
    reviews each file separately, ending in ``partial_review`` — not ``ok`` (a
    per-file review cannot see a cross-file defect), not ``crashed``.

    A REAL two-file diff is used on purpose. A headerless one would fail to split
    and crash for an unrelated reason (see
    ``tests/test_chunk_route.py::...headerless...``), and a green for the wrong
    reason is worse than a red.
    """

    def handler(system, messages):
        if "adjudicate" in system.lower():
            listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
            n = max(1, len(re.findall(r"^\s*(\d+)\.\s", listing, re.MULTILINE)))
            return _resp("\n".join(f"CONFIRM {i}" for i in range(1, n + 1)))
        if "one file from a larger diff" in system.lower():
            return _resp("HIGH: issue in this file.")
        raise _StubAnthropic400(OVERFLOW_MSG)  # the whole-diff review overflows

    class _Messages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            return handler(system or "", messages)

    two_files = (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-p\n+q\n"
    )

    llm = LLMClient(provider="claude", model="claude-haiku-4-5")
    llm._client = type("_Client", (), {"messages": _Messages()})()
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=audit)

    result = orch.run(two_files)  # no longer raises

    env = _trail(result.run_dir)
    assert env["summary"]["status"] == "partial_review"
    assert result.status != STATUS_CRASHED
    # the run reviewed each file: two confirmed findings from the per-piece pass
    assert len(result.verify.confirmed) == 2
