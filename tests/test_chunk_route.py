"""A context-length overflow reviews the diff a file at a time.

The map in ``engine.failure_classifier`` declares ``context_length_exceeded ->
chunk-diff``. This is where that map is finally *consumed*: an upstream capture
point in the orchestrator catches the wire failure, classifies it by SIGNAL
(never by exception class), and — for the one mapped class — reviews the diff
one file per piece instead of letting the 400 kill the run.

The run does NOT come back ``ok``. A per-file review cannot see a defect that
spans two files, so returning ``ok`` would certify coverage that never happened.
It ends in its own status, ``partial_review``, and the trail records the
mechanism (how many pieces, which were unreadable, and what the split cannot
see) while the status carries only the consequence.

Only review is replaced. The stop-gate and the verifier still run — skipping the
gate would mean loopward doing work without asking, and skipping verify would
carve a second path around a core primitive. The per-piece findings are merged
into one flat list, which the verifier renumbers 1..n and adjudicates whole.

Detection is by wire signal: the stub 400 duck-types exactly the fields
``extract_signal`` reads off a real ``anthropic.BadRequestError``.
"""

import json
import pathlib
import re

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import LLMClient
from loopward.engine.orchestrator import Orchestrator
from loopward.engine.stop_gate import StopGate

OVERFLOW_MSG = "prompt is too long: 250000 tokens > 200000 maximum"


class _StubAnthropic400(Exception):
    """Duck-types the fields ``extract_signal`` reads off a real BadRequestError."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400
        self.type = "invalid_request_error"
        self.message = message


class _StubAnthropic500(Exception):
    """A server error: status present, but not the measured overflow -> UNKNOWN."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 500
        self.type = "api_error"
        self.message = message


def _resp(text: str, stop_reason: str | None = None):
    block = type("_Block", (), {"type": "text", "text": text})()
    return type("_Resp", (), {"content": [block], "stop_reason": stop_reason})()


def _claude_with(handler):
    """A fake Anthropic client whose ``messages.create`` delegates to ``handler``.

    ``handler(system, messages) -> resp`` (or raises). Bypasses the lazy SDK
    construction, so no key and no network — same technique as
    test_provider_stop_reason.
    """

    class _Messages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            return handler(system or "", messages)

    return type("_Client", (), {"messages": _Messages()})()


def _confirm_all(messages) -> str:
    listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
    n = max(1, len(re.findall(r"^\s*(\d+)\.\s", listing, re.MULTILINE)))
    return "\n".join(f"CONFIRM {i}" for i in range(1, n + 1))


def _is_verify(system: str) -> bool:
    return "adjudicate" in system.lower()


def _is_piece(system: str) -> bool:
    return "one file from a larger diff" in system.lower()


def _bare(name: str, old: str, new: str) -> str:
    return f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


TWO_FILES = _bare("a.py", "x", "y") + _bare("b.py", "p", "q")
THREE_FILES = TWO_FILES + _bare("c.py", "m", "n")


def _trail(run_dir: str) -> dict:
    return json.loads((pathlib.Path(run_dir) / "audit.json").read_text(encoding="utf-8"))


def _claude_orch(audit, handler, gate=None):
    llm = LLMClient(provider="claude", model="claude-haiku-4-5")
    llm._client = _claude_with(handler)
    return Orchestrator(llm, gate or StopGate(mode="auto"), audit=audit)


# --- the happy path: a 400 chunks instead of dying, and ends partial_review ---


@pytest.mark.integration
def test_a_context_length_400_chunks_instead_of_crashing(tmp_path):
    """The whole-diff review overflows; the per-file review succeeds.

    Ends in ``partial_review`` (NOT ``ok``, NOT ``crashed``), with a finding per
    file and a completed verify. The run reached a verdict; it just did so a
    piece at a time, so it says so.
    """

    def handler(system, messages):
        if _is_verify(system):
            return _resp(_confirm_all(messages))
        if _is_piece(system):
            # which file this piece is, so we can assert order downstream
            body = "\n".join(m["content"] for m in messages)
            tag = "a.py" if "a.py" in body else "b.py"
            return _resp(f"HIGH: issue in {tag}.")
        raise _StubAnthropic400(OVERFLOW_MSG)  # the whole-diff review

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    result = _claude_orch(audit, handler).run(TWO_FILES)

    assert result.status == "partial_review"
    assert result.status != "ok"
    assert result.verify is not None
    assert [str(f) for f in result.verify.confirmed] == [
        "HIGH: issue in a.py.",
        "HIGH: issue in b.py.",
    ]
    assert _trail(result.run_dir)["summary"]["status"] == "partial_review"


@pytest.mark.integration
def test_the_trail_names_the_pieces_and_what_it_could_not_see(tmp_path):
    """The status carries the consequence; the trail carries the mechanism.

    How many pieces, which were unreadable, and the cross-file blind spot — the
    thing a per-file review structurally cannot find.
    """

    def handler(system, messages):
        if _is_verify(system):
            return _resp(_confirm_all(messages))
        if _is_piece(system):
            return _resp("HIGH: something.")
        raise _StubAnthropic400(OVERFLOW_MSG)

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    _claude_orch(audit, handler).run(TWO_FILES)

    chunk_events = [e for e in audit.events if e.kind == "chunk_review"]
    assert len(chunk_events) == 1, "the chunked review left no trace of what it did"
    data = chunk_events[0].data
    assert data["chunks"] == 2
    # the blind spot is named, not implied
    blob = json.dumps(data).lower()
    assert "cross-file" in blob or "spanning" in blob or "two files" in blob


@pytest.mark.integration
def test_merged_findings_are_renumbered_and_verify_covers_them_all(tmp_path):
    """Three pieces -> three findings -> verify adjudicates 1..3 exactly.

    The verifier demands total coverage, so the per-piece findings must merge
    into one flat list renumbered 1..n. If the renumbering were wrong the verify
    would raise and the run would end EXHAUSTED, not partial_review.
    """
    seen_listings = []

    def handler(system, messages):
        if _is_verify(system):
            listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
            seen_listings.append(listing)
            return _resp(_confirm_all(messages))
        if _is_piece(system):
            body = "\n".join(m["content"] for m in messages)
            for tag in ("a.py", "b.py", "c.py"):
                if tag in body:
                    return _resp(f"HIGH: bug in {tag}.")
            return _resp("HIGH: bug.")
        raise _StubAnthropic400(OVERFLOW_MSG)

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    result = _claude_orch(audit, handler).run(THREE_FILES)

    assert result.status == "partial_review"
    assert len(result.verify.confirmed) == 3
    # the verifier saw a single 1..3 listing, not three 1..1 ones
    assert seen_listings, "verify never ran"
    listing = seen_listings[0]
    assert re.search(r"^\s*1\.\s", listing, re.MULTILINE)
    assert re.search(r"^\s*2\.\s", listing, re.MULTILINE)
    assert re.search(r"^\s*3\.\s", listing, re.MULTILINE)


# --- the two declared limits: both CRASH in this commit, and say why ----------


@pytest.mark.integration
def test_more_than_max_chunks_files_does_not_chunk_and_crashes(tmp_path):
    """>10 files: partial_review would be a lie (nothing was reviewed). Crash.

    The split refuses whole and the trail carries the count and the cap, which is
    what separates a declared limit from a silent truncation.
    """
    many = "".join(_bare(f"f{i}.py", "x", "y") for i in range(11))

    def handler(system, messages):
        raise _StubAnthropic400(OVERFLOW_MSG)  # the whole-diff review overflows

    from loopward.agents.chunk_diff import TooManyFilesToChunk

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    with pytest.raises(TooManyFilesToChunk):
        _claude_orch(audit, handler).run(many)

    env = _trail(audit.run_dir_on_disk)
    assert env["summary"]["status"] == "crashed"
    result = [e for e in env["events"] if e["kind"] == "result"][-1]
    assert "11" in result["message"] and "10" in result["message"]
    # never got a per-file review going
    assert not any(e["kind"] == "chunk_review" for e in env["events"])


@pytest.mark.integration
def test_a_headerless_diff_that_overflows_then_wont_split_crashes(tmp_path):
    """400 -> chunk -> the diff has no file header -> MalformedDiffError -> crash.

    The malformed case gets its own test precisely because the old
    classifier test used a headerless diff and would now go green for a *different
    reason*. This one asserts the crash lands with the malformed cause.
    """

    def handler(system, messages):
        raise _StubAnthropic400(OVERFLOW_MSG)

    from loopward.agents.chunk_diff import MalformedDiffError

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    with pytest.raises(MalformedDiffError):
        _claude_orch(audit, handler).run("just prose, nothing that looks like a diff\n")

    assert _trail(audit.run_dir_on_disk)["summary"]["status"] == "crashed"


# --- THE HARD RULE: the widened capture re-raises UNKNOWN / unmapped ----------
# This is what keeps crash_trail test 5 green and stops the widening from eating
# exceptions that kill the run today. If someone drops the re-raise, this reddens.


@pytest.mark.integration
def test_an_unmapped_failure_in_review_is_re_raised_not_swallowed(tmp_path):
    """A 500 classifies UNKNOWN. The widened except must let it crash the run.

    The whole point of the widening is to route the ONE mapped class. Everything
    else must reach the crash path exactly as it does today — same exception,
    finalized crashed.
    """

    def handler(system, messages):
        raise _StubAnthropic500("upstream is having a bad day")

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    with pytest.raises(_StubAnthropic500):
        _claude_orch(audit, handler).run(TWO_FILES)

    env = _trail(audit.run_dir_on_disk)
    assert env["summary"]["status"] == "crashed"
    assert not any(e["kind"] == "chunk_review" for e in env["events"])
    # it never became a partial review, and never landed in a strategy
    assert not any(e["kind"] == "class_jump" for e in env["events"])
