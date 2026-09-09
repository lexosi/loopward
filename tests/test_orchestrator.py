"""Integration tests for the orchestrator (fake provider, offline)."""


import re

import pytest

from loopward.agents.reviewer import STRATEGIES
from loopward.engine import anti_loop as anti_loop_mod
from loopward.engine import orchestrator as orch_mod
from loopward.engine.anti_loop import AttemptTracker
from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import TASK_REVIEW, TASK_VERIFY, LLMClient, Message
from loopward.engine.orchestrator import (
    STATUS_EXHAUSTED,
    STATUS_GATE_DENIED,
    STATUS_OK,
    Orchestrator,
)
from loopward.engine.stop_gate import StopGate

DIFF = "return now <= self.expires_at"


def _audit(tmp):
    return AuditLog(run_id="test", base_dir=tmp)


@pytest.mark.integration
def test_happy_path_ok(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_OK
    assert result.verify is not None
    assert result.verify.confirmed


@pytest.mark.integration
def test_gate_denied_stops_before_verify(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
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
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "no tags here, just prose")
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert result.findings == []


@pytest.mark.integration
def test_audit_files_written(tmp_path):
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    result = orch.run(DIFF)
    from pathlib import Path

    run_dir = Path(result.run_dir)
    assert (run_dir / "audit.json").exists()
    assert (run_dir / "audit.md").exists()


# ---- fake reply helpers ---------------------------------------------------


def _confirm_all(messages: list[Message]) -> str:
    """Adjudicate every listed finding: the verifier requires full coverage."""
    listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
    n = len(re.findall(r"^\s*(\d+)\.\s", listing, re.MULTILINE))
    return "\n".join(f"CONFIRM {i}" for i in range(1, n + 1))


def _reply(messages: list[Message], task: str | None) -> str:
    if task == TASK_VERIFY:
        return _confirm_all(messages)
    return "HIGH: token expiry uses <=, use <"


def _force_class_jump(messages: list[Message], task: str | None) -> str:
    if task == TASK_VERIFY:
        return _confirm_all(messages)
    if task == f"{TASK_REVIEW}:structured":
        return "HIGH: token expiry uses <=, use <"
    return "vague prose with no severity tags"  # concise strategy: unparseable


# ---- the gate must leave a trace in the trail -------------------------------
# loopward's third headline is the audit trail, and the stop-gate is the
# primitive the README leads with. A gate decision that happens but is not
# recorded is indistinguishable, after the fact, from a gate that never ran.


@pytest.mark.integration
def test_gate_decision_lands_in_the_trail(tmp_path):
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)
    assert "gate" in {e.kind for e in audit.events}


@pytest.mark.integration
def test_gate_denied_is_recorded_too(tmp_path):
    # A refusal is the decision most worth having in the trail.
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    Orchestrator(llm, StopGate(mode="deny"), audit=audit).run(DIFF)
    gate_events = [e for e in audit.events if e.kind == "gate"]
    assert len(gate_events) == 1
    assert gate_events[0].data["verdict"] == "deny"


@pytest.mark.integration
def test_gate_event_carries_structured_data(tmp_path):
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)
    data = next(e for e in audit.events if e.kind == "gate").data
    assert data["mode"] == "auto"
    assert data["verdict"] == "approve"
    assert data["reason"]
    # who decided, and whether that identity was declared or inferred
    assert data["approver_source"] in ("env", "os_user", "unknown")
    assert "approver" in data


@pytest.mark.integration
def test_injected_tracker_still_records_attempts(tmp_path):
    """Same defect class as the gate: a collaborator passed in ready-made.

    The wiring used to live inside ``tracker or AttemptTracker(audit=...)``, so
    it only ever reached the branch the orchestrator built itself.
    """
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=_force_class_jump)
    Orchestrator(llm, StopGate(mode="auto"), audit=audit,
                 tracker=AttemptTracker()).run(DIFF)
    assert [e for e in audit.events if e.kind == "attempt"]


@pytest.mark.integration
def test_orchestrator_does_not_override_an_explicit_sink(tmp_path):
    """Adopting unwired collaborators must not hijack a caller's own sink."""
    mine: list[tuple[str, str]] = []
    audit = _audit(tmp_path)
    gate = StopGate(mode="auto", audit=lambda kind, msg, **data: mine.append((kind, msg)))
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    Orchestrator(llm, gate, audit=audit).run(DIFF)
    assert mine and mine[0][0] == "gate"
    assert "gate" not in {e.kind for e in audit.events}


# ---- the bound belongs to the loop, not to the collaborator -----------------
# `Orchestrator(..., tracker=..., strategies=...)` are public parameters. Both
# multiply into the anti-loop's budget, so if the loop trusts them for its
# termination, "never spins" is a promise the caller gets to break.

SPIN_GUARD = 600  # far above MAX_TOTAL_ATTEMPTS; trips only if the loop is unbounded


class LoopUnbounded(RuntimeError):
    """Raised by the test double instead of letting the suite hang."""


def _never_parses():
    """Fake LLM whose output never parses, and that screams rather than spin."""
    calls = {"n": 0}

    def reply(_messages: list[Message], _task: str | None) -> str:
        calls["n"] += 1
        if calls["n"] > SPIN_GUARD:
            raise LoopUnbounded(f"{calls['n']} LLM calls and still going")
        return "vague prose with no severity tags"

    return reply, calls


@pytest.mark.integration
def test_injected_tracker_cannot_spin(tmp_path):
    """A tracker that never grants a class-jump must not buy an unbounded loop."""
    reply, calls = _never_parses()
    llm = LLMClient(provider="fake", fake_script=reply)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                          tracker=AttemptTracker(max_attempts=10**9)).run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert calls["n"] <= orch_mod.MAX_TOTAL_ATTEMPTS


@pytest.mark.integration
def test_injected_strategies_cannot_spin(tmp_path):
    """Same defect, the other multiplier: the default tracker is untouched here.

    The list is real strategies, repeated — the cap counts how many there are,
    not whether the names differ. It used to inject invented names ('s0', 's1',
    ...), which only ran at all because the reviewer resolved an unknown strategy
    to 'concise' in silence: a guard that leaned on the very defect it guarded.
    With that fallback gone, an unknown strategy is rejected, so the huge list
    must be built from strategies the reviewer actually knows.
    """
    reply, calls = _never_parses()
    llm = LLMClient(provider="fake", fake_script=reply)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                          strategies=STRATEGIES * (10**6)).run(DIFF)
    assert result.status == STATUS_EXHAUSTED
    assert calls["n"] <= orch_mod.MAX_TOTAL_ATTEMPTS


# --- the tracker's verdict is checked, not just read ------------------------
#
# `tracker` is a public constructor parameter, so what the loops read
# `must_class_jump` off is caller-supplied. `is_genuine_outcome` had no
# production caller at all; now it guards both places a verdict enters.


class DuckOutcome:
    """A stand-in a hand-rolled tracker might plausibly return."""

    def __init__(self, must_class_jump: bool) -> None:
        self.must_class_jump = must_class_jump
        self.attempt = 1


class DuckTracker:
    """Not an AttemptTracker. Mints nothing; returns something that looks right."""

    max_attempts = 3

    def __init__(self, verdict: bool = False) -> None:
        self._verdict = verdict

    def attach_audit(self, sink):
        pass

    def record_failure(self, subtask_id, strategy):
        return DuckOutcome(self._verdict)

    def reset(self, subtask_id):
        pass


@pytest.mark.integration
def test_review_rejects_a_verdict_the_tracker_never_minted(tmp_path):
    """A duck-typed outcome is refused where review reads it, not obeyed."""
    reply, _ = _never_parses()
    llm = LLMClient(provider="fake", fake_script=reply)
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                        tracker=DuckTracker())
    with pytest.raises(TypeError, match="did not mint"):
        orch.run(DIFF)


@pytest.mark.integration
def test_verify_rejects_a_verdict_the_tracker_never_minted(tmp_path):
    """The same guard on the verify loop: one check, not half of one."""
    def reply(_messages: list[Message], task: str | None) -> str:
        # Review parses; the adjudication does not cover finding 1, so the
        # verify loop records a failure and reads the verdict.
        return "nothing adjudicated" if task == TASK_VERIFY else "HIGH: real finding"

    llm = LLMClient(provider="fake", fake_script=reply)
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path),
                        tracker=DuckTracker())
    with pytest.raises(TypeError, match="did not mint"):
        orch.run(DIFF)


@pytest.mark.unit
def test_a_genuine_verdict_still_passes_the_check(tmp_path):
    """The guard must not cost the ordinary path anything."""
    llm = LLMClient(provider="fake", fake_script=lambda m, task: _reply(m, task))
    orch = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path))
    assert orch.run(DIFF).status == STATUS_OK


# --- the trail keeps both lists, not just the counters ----------------------


@pytest.mark.integration
def test_verify_event_names_which_findings_were_dropped(tmp_path):
    """"The audit trail keeps both" was a claim the counters could not support.

    `verify: confirmed 1, rejected 1` tells a reader that something was dropped
    and never which one. The review event already carried its findings in
    `data`; the verify event now does the same.
    """
    def reply(_messages: list[Message], task: str | None) -> str:
        if task == TASK_VERIFY:
            return "CONFIRM 1\nREJECT 2"
        return "HIGH: real bug here\nLOW: probably a false positive"

    llm = LLMClient(provider="fake", fake_script=reply)
    audit = _audit(tmp_path)
    result = Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)
    assert result.status == STATUS_OK

    event = next(e for e in audit.events if e.message.startswith("verify: confirmed"))
    # The message is quoted verbatim by the demo and the README: unchanged.
    assert event.message == "verify: confirmed 1, rejected 1"
    assert event.data["confirmed"] == ["HIGH: real bug here"]
    assert event.data["rejected"] == ["LOW: probably a false positive"]
    assert event.data["verified"] is True


# --- a cut-off is never silent, and never claims a jump it did not make -----
#
# Two halves of one defect, on adjacent lines of the same loop. The review loop
# has two exits; only the inner `if` was instrumented. So the exit that runs in
# every default configuration recorded nothing, while the last `class_jump`
# announced a switch to a strategy that does not exist. One stayed quiet about
# what happened; the other asserted what did not.


@pytest.mark.integration
def test_strategies_exhausted_is_recorded_like_any_other_cut_off(tmp_path):
    """The default exhaustion path emits its stop event, in the same shape.

    This is the exit 64 of 85 tracker/strategy combinations take, the demo's
    sibling path and the one the benchmark runs — and it went through no
    `_record_stop` at all. `strategies_exhausted` was declared, commented, and
    never emitted anywhere.
    """
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "no tags here, just prose")

    audit = _audit(tmp_path)
    assert Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF).status == (
        STATUS_EXHAUSTED
    )
    stops = [e for e in audit.events if e.kind == "anti_loop"]
    assert len(stops) == 1, "the review loop cut the run short and recorded nothing"
    assert stops[0].data["stop_reason"] == orch_mod.STOP_STRATEGIES_EXHAUSTED
    assert stops[0].data["attempts"] == 6  # MAX_ATTEMPTS x len(STRATEGIES)

    # One kind, one schema. A trail reader must not learn two shapes for
    # `anti_loop` depending on which exit of the same loop produced it.
    ceiling = _audit(tmp_path)
    Orchestrator(llm, StopGate(mode="auto"), audit=ceiling,
                 tracker=AttemptTracker(max_attempts=10**9)).run(DIFF)
    other = next(e for e in ceiling.events if e.kind == "anti_loop")
    assert other.data["stop_reason"] == orch_mod.STOP_TOTAL_CAP
    assert set(stops[0].data) == set(other.data)


@pytest.mark.integration
def test_the_summary_never_contradicts_the_stop_event(tmp_path):
    """One run, one story. The `result` summary must agree with `anti_loop`.

    `_run` picked its summary with a two-way branch — `total_cap`, or else "every
    strategy was tried" — while `_record_stop` emits three reasons. So a
    `tracker_budget` cut-off produced a trail that said "budget exhausted" in one
    event and "exhausted all strategies" in the next.

    A trail that is merely incomplete leaves the reader knowing less. A trail
    that contradicts itself leaves them unable to trust either half, which is
    worse, and it is the failure mode this package is named after.

    `tracker_budget` is reachable: a subclass of `AttemptTracker` that mints
    through `_mint_outcome` passes the genuineness check with a perpetual
    `retry`, so the strategy index never advances and the budget runs out first.
    That branch is not being fixed here — only its description, and this.
    """

    class NeverJumps(AttemptTracker):
        """Genuine tracker, genuine verdicts, and it never grants a class-jump."""

        def record_failure(self, subtask_id, strategy):
            n = self._counts.get(subtask_id, 0) + 1
            self._counts[subtask_id] = n
            return anti_loop_mod._mint_outcome(
                subtask_id, n, anti_loop_mod.RETRY, strategy, "no jump"
            )

    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "no tags here, just prose")
    result = Orchestrator(
        llm, StopGate(mode="auto"), audit=audit, tracker=NeverJumps(max_attempts=4)
    ).run(DIFF)

    assert result.status == STATUS_EXHAUSTED
    stop = next(e for e in audit.events if e.kind == "anti_loop")
    assert stop.data["stop_reason"] == orch_mod.STOP_TRACKER_BUDGET
    assert stop.data["attempts"] == 8  # 4 attempts x 2 strategies, no jump taken

    # The strategy list was never exhausted — the budget ran out on the first
    # strategy. Saying otherwise files a cut-off that did not happen.
    assert "exhausted all strategies" not in result.summary
    assert "budget" in result.summary
    # The summary the caller gets and the one in the trail are the same sentence.
    assert [e for e in audit.events if e.kind == "result"][-1].message == result.summary


@pytest.mark.integration
def test_every_stop_reason_gets_its_own_summary(tmp_path):
    """The three cut-offs read differently, because they mean different things."""
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "no tags here, just prose")

    exhausted = Orchestrator(llm, StopGate(mode="auto"), audit=_audit(tmp_path)).run(DIFF)
    ceiling = Orchestrator(
        llm, StopGate(mode="auto"), audit=_audit(tmp_path),
        tracker=AttemptTracker(max_attempts=10**9),
    ).run(DIFF)

    assert "every strategy" in exhausted.summary or "all strategies" in exhausted.summary
    assert str(orch_mod.MAX_TOTAL_ATTEMPTS) in ceiling.summary
    assert exhausted.summary != ceiling.summary


@pytest.mark.integration
def test_no_class_jump_is_announced_without_a_strategy_to_jump_to(tmp_path):
    """The tracker's verdict is not the event; the switch is.

    On the last strategy the tracker still returns `class_jump` — correctly:
    "no further retries with the same approach". But there is nothing to switch
    to, so recording a switch files a move that never happened. The verdict is
    not lost: the tracker's own `attempt` event still carries it.
    """
    audit = _audit(tmp_path)
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "no tags here, just prose")
    Orchestrator(llm, StopGate(mode="auto"), audit=audit).run(DIFF)

    jumps = [e for e in audit.events if e.kind == "class_jump"]
    # N strategies means at most N-1 switches between them.
    assert len(jumps) == len(STRATEGIES) - 1
    assert not [e for e in jumps if f"'{STRATEGIES[-1]}'" in e.message], (
        "the final strategy has no successor; no jump away from it can happen"
    )
    # The verdict that ended the run is still on the record.
    assert [e for e in audit.events if e.kind == "attempt" and "class-jump required" in e.message]
