"""orchestrator.py — the single coordinator.

Runs a code-review as two phases with reliability baked in:

1. **review** — ask the Reviewer for findings. If the output won't parse, the
   failure is fed to the :class:`AttemptTracker`. After ``MAX_ATTEMPTS`` the
   tracker forces a *class-jump*: the orchestrator switches to a different review
   strategy instead of retrying the same one forever. The loop owns its own
   budget (``max_attempts * len(strategies)``, capped by
   :data:`MAX_TOTAL_ATTEMPTS`), so it terminates whatever the injected tracker
   and strategy list say.
2. **verify** — gated by the :class:`StopGate`. A human (or ``auto`` mode in CI)
   approves before the Verifier re-checks the findings.

Everything — every attempt, gate decision, and token cost — lands in the
:class:`AuditLog`, along with the raw reason each provider gave for stopping.
That last one is carried and recorded, never consulted: no branch in this
module reads it. The orchestrator is the only component that calls agents;
agents never call each other.

All collaborators are injected, so the whole flow runs offline and is trivial to
test with the ``fake`` provider.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

from loopward.agents.chunk_diff import CHUNK_DIFF_INSTRUCTION, split_diff
from loopward.agents.reviewer import STRATEGIES, Finding, Reviewer, ReviewParseError
from loopward.agents.verifier import PHASE_VERIFY, Verifier, VerifyParseError, VerifyResult
from loopward.engine.anti_loop import AttemptOutcome, AttemptTracker, is_genuine_outcome
from loopward.engine.audit import AuditAlreadyFinalizedError, AuditLog
from loopward.engine.failure_classifier import (
    CHUNK_DIFF_STRATEGY,
    NEXT_STRATEGY,
    classify,
    classify_signal,
    extract_signal,
)
from loopward.engine.llm_wrapper import TASK_REVIEW, EmptyCompletionError, LLMClient
from loopward.engine.stop_gate import StopGate, consume_approval

REVIEW_SUBTASK = "review-parse"
VERIFY_SUBTASK = "verify-parse"
VERIFY_STRATEGY = "adjudicate"

STATUS_OK = "ok"
STATUS_GATE_DENIED = "gate_denied"
STATUS_EXHAUSTED = "exhausted"
#: The run did not reach a verdict: something escaped. The trail is still
#: written, because "we do not know what happened" is the one thing an audit
#: trail must never say about a run that happened.
STATUS_CRASHED = "crashed"
#: The diff overflowed the context window whole, so it was reviewed one file at
#: a time. The review reached a verdict, but its coverage is incomplete: a defect
#: that spans two files cannot be seen when each file is reviewed alone. NOT
#: `ok` — returning `ok` would certify coverage that did not happen. This status
#: carries only that consequence; the trail's `chunk_review` event carries the
#: mechanism (how many pieces, which were unreadable).
STATUS_PARTIAL_REVIEW = "partial_review"

#: Absolute ceiling on review attempts in a single run, whatever the injected
#: tracker and strategy list add up to. This is a SPEND limit, not a design
#: limit: the anti-loop's real budget is ``max_attempts * len(strategies)``
#: (6 by default, and the benchmark asserts it), and this exists only so that
#: no caller-supplied number can turn a bounded loop into an unbounded bill.
MAX_TOTAL_ATTEMPTS = 100

# Why the review loop stopped, recorded so a trail reader can tell an ordinary
# exhaustion from a budget running out and from a ceiling being hit — the three
# mean different things. All three reach the trail as `anti_loop.stop_reason`;
# the first was declared here and emitted nowhere, which left the cut-off that
# every default configuration takes as the one the trail never mentioned.
STOP_STRATEGIES_EXHAUSTED = "strategies_exhausted"  # every strategy was tried
STOP_TRACKER_BUDGET = "tracker_budget"  # max_attempts x strategies used up
STOP_TOTAL_CAP = "total_cap"  # MAX_TOTAL_ATTEMPTS reached first

# What the caller is told for each of them. Keyed on the same constant the trail
# records, so the `result` summary and the `anti_loop` event cannot drift apart:
# a two-way branch here against three reasons there is how a `tracker_budget`
# cut-off came to be summarised as "exhausted all strategies", leaving one run
# telling a reader two different stories. The fallback names an unmapped reason
# verbatim rather than guessing at one, because a vague sentence that sounds
# right is the failure this mapping exists to stop.
_EXHAUSTED_SUMMARY = {
    STOP_STRATEGIES_EXHAUSTED: ("review exhausted all strategies without parseable findings"),
    STOP_TRACKER_BUDGET: (
        "review stopped when its attempt budget ran out without parseable findings"
    ),
    STOP_TOTAL_CAP: (
        f"review stopped at the absolute ceiling of "
        f"{MAX_TOTAL_ATTEMPTS} attempts without parseable findings"
    ),
}


@dataclass(frozen=True)
class RunResult:
    """Final outcome of a run."""

    status: str
    findings: list[Finding]
    verify: VerifyResult | None
    run_dir: str
    summary: str


class Orchestrator:
    """Coordinates reviewer + verifier under anti-loop and a stop-gate."""

    def __init__(
        self,
        llm: LLMClient,
        gate: StopGate,
        audit: AuditLog | None = None,
        tracker: AttemptTracker | None = None,
        strategies: tuple[str, ...] = STRATEGIES,
    ) -> None:
        self._llm = llm
        self._gate = gate
        self._audit = audit or AuditLog(run_id="code-review")
        self._tracker = tracker or AttemptTracker()
        # One wiring point for every collaborator, however it got here. Putting
        # it inside `tracker or AttemptTracker(audit=...)` only ever reaches the
        # branch this class builds itself — which is how the gate, always passed
        # in ready-made, ended up recording nothing at all. Each collaborator
        # keeps a sink it was given explicitly.
        for collaborator in (self._gate, self._tracker):
            collaborator.attach_audit(self._audit.record)
        self._strategies = strategies
        self._reviewer = Reviewer(llm)
        self._verifier = Verifier(llm)

    def run(self, diff: str) -> RunResult:
        """Execute the review→verify flow and finalize the audit trail.

        Nothing leaves this method without the trail being written first. That
        is the whole point: for a package whose third headline is "audit
        trail", an unhandled exception used to mean zero record of what
        happened — the default failure mode was to audit nothing.

        Two failure shapes, deliberately not treated alike:

        - **The work failed** (network, provider, EOF at the gate, Ctrl-C).
          The trail is finalized as :data:`STATUS_CRASHED` and the exception is
          re-raised. Any findings already produced stay in the trail and do
          *not* come back through the return value: at that point they are
          unverified reviewer output, and handing them over the result channel
          would present the generator's guess as the adjudicator's verdict.
        - **Only the write failed** (:meth:`_record_usage_and_finalize`). The
          review itself was complete and correct, so it is reported normally
          and nothing is raised. A lost record does not retroactively make a
          good run a bad one.
        """
        try:
            return self._run(diff)
        except BaseException as exc:
            # BaseException on purpose: KeyboardInterrupt at the interactive
            # gate is the single most likely way a real run ends early, and it
            # is not an Exception. The trail is written, then the original
            # exception continues on its way untouched.
            self._finalize_crashed(exc)
            raise

    def _run(self, diff: str) -> RunResult:
        self._audit.record("phase", "review: start")
        findings, strategy, stop_reason = self._review_with_anti_loop(diff)

        # The review was carried out one file at a time after a context-length
        # overflow. It reached findings, so it is not an exhaustion; but the run
        # cannot end `ok`, because a per-file review does not see cross-file
        # defects. The flag rides the existing `strategy` channel — no widened
        # return signature — and decides only the final status and summary.
        partial = strategy == CHUNK_DIFF_STRATEGY

        if findings is None:
            summary = _EXHAUSTED_SUMMARY.get(
                stop_reason,
                f"review stopped without parseable findings ({stop_reason})",
            )
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_EXHAUSTED, summary)
            return RunResult(STATUS_EXHAUSTED, [], None, str(run_dir), summary)

        # The findings themselves ride in `data`, never in `message`: the
        # message is what the demo and the README quote, and it stays
        # byte-identical. `verified=False` is not decoration — at this point
        # these are the reviewer's claims and nothing has adjudicated them. A
        # trail that stored them unmarked would be filing the generator's
        # output as if it were the evaluator's verdict.
        review_message = (
            f"review: partial via chunk-diff — {len(findings)} finding(s) across "
            f"separately reviewed files"
            if partial
            else f"review: ok via strategy '{strategy}' ({len(findings)} findings)"
        )
        self._audit.record(
            "phase",
            review_message,
            stage="review",
            verified=False,
            findings=[str(f) for f in findings],
        )

        gate_summary = f"{len(findings)} findings ready to verify"
        decision = self._gate.request(PHASE_VERIFY, gate_summary)
        if not decision.approved:
            summary = f"stop-gate denied verification ({decision.reason})"
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_GATE_DENIED, summary)
            return RunResult(STATUS_GATE_DENIED, findings, None, str(run_dir), summary)

        self._audit.record("phase", "verify: start")
        verify = self._verify_with_anti_loop(findings, diff, decision.approval)
        if verify is None:
            summary = "verification produced no usable adjudication"
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_EXHAUSTED, summary)
            return RunResult(STATUS_EXHAUSTED, findings, None, str(run_dir), summary)

        # Which findings, not just how many. The counters alone made "the
        # verifier kept both" a claim the trail could not support: a reader
        # could see that one was rejected and never learn which. The review
        # event above already carries its findings in `data`; the asymmetry was
        # an omission. The `message` is unchanged — the demo and the README
        # quote it byte-identically.
        self._audit.record(
            "phase",
            f"verify: confirmed {len(verify.confirmed)}, rejected {len(verify.rejected)}",
            stage="verify",
            verified=True,
            confirmed=[str(f) for f in verify.confirmed],
            rejected=[str(f) for f in verify.rejected],
        )

        blocking = ", blocking" if verify.has_blocking else ", none blocking"
        if partial:
            final_status = STATUS_PARTIAL_REVIEW
            summary = (
                f"{len(verify.confirmed)} confirmed finding(s){blocking}; partial "
                f"review — the diff was reviewed one file at a time after a "
                f"context-length overflow, so a defect spanning two files was not "
                f"covered"
            )
        else:
            final_status = STATUS_OK
            summary = f"{len(verify.confirmed)} confirmed finding(s){blocking}"
        self._audit.record("result", summary)
        run_dir = self._record_usage_and_finalize(final_status, summary)
        return RunResult(final_status, findings, verify, str(run_dir), summary)

    # ---- internals --------------------------------------------------------

    def _review_with_anti_loop(self, diff: str) -> tuple[list[Finding] | None, str, str]:
        """Try strategies in order, advancing on a class-jump. Never loops forever.

        The budget is computed here, before the loop, and enforced here. It used
        to be delegated to the tracker's verdict — but ``tracker`` and
        ``strategies`` are both public constructor parameters, so a caller could
        set the budget arbitrarily high (or hand over a tracker that never grants
        a class-jump) and this loop would obey. The guarantee lived in the
        collaborator instead of in the code that promises it.

        Returns ``(findings, strategy, stop_reason)``; ``stop_reason`` is empty
        when a strategy produced findings.
        """
        tracker_budget = self._tracker.max_attempts * len(self._strategies)
        budget = min(tracker_budget, MAX_TOTAL_ATTEMPTS)
        attempts = 0
        strat_idx = 0

        while strat_idx < len(self._strategies):
            if attempts >= budget:
                return None, "", self._record_stop(attempts, budget, tracker_budget)
            strategy = self._strategies[strat_idx]
            try:
                findings, dropped = self._reviewer.review(diff, strategy=strategy)
            except Exception as exc:
                # Upstream capture point. A parse failure or a blank completion is
                # a spent attempt (an empty reply is a failed attempt, the same as
                # an unparseable one — it used to cross this loop untouched and
                # crash the run). Everything else is classified by WIRE SIGNAL,
                # never by exception class (see failure_classifier): the ONE
                # mapped class routes to a per-file review, and anything else —
                # UNKNOWN included — is re-raised untouched, so a failure that
                # kills the run today still does, with its raw signal on the crash
                # trail. Dropping that re-raise would make the widening swallow
                # exceptions; test_chunk_route guards it, and crash_trail's
                # missing-key test depends on it.
                if not isinstance(exc, (ReviewParseError, EmptyCompletionError)):
                    if (
                        NEXT_STRATEGY.get(classify(exc, getattr(self._llm, "provider", None)))
                        == CHUNK_DIFF_STRATEGY
                    ):
                        return self._review_chunked(diff), CHUNK_DIFF_STRATEGY, ""
                    raise
                attempts += 1
                outcome = self._record_failure(REVIEW_SUBTASK, strategy)
                if outcome.must_class_jump:
                    next_idx = strat_idx + 1
                    if next_idx < len(self._strategies):
                        # Only a switch that happens is recorded as one. On the
                        # last strategy the verdict is still `class_jump`, and
                        # still right — no more retries with this approach — but
                        # there is nowhere to jump to, and an event announcing a
                        # move the loop cannot make is the trail asserting
                        # something false. Nothing is lost by the silence: the
                        # tracker's own `attempt` event carries the verdict, and
                        # the stop event below records how the run ended.
                        self._audit.record(
                            "class_jump",
                            f"switching review strategy after {outcome.attempt} failures "
                            f"on '{strategy}'",
                        )
                    self._tracker.reset(REVIEW_SUBTASK)
                    strat_idx = next_idx
                continue
            if dropped:
                # Not a decision, a datum: the parse succeeded, but this many
                # lines of the reply were something other than a finding.
                self._audit.record(
                    "review_parse",
                    f"{len(findings)} finding(s) parsed, {dropped} line(s) dropped "
                    f"on strategy '{strategy}'",
                    matched=len(findings),
                    dropped=dropped,
                    strategy=strategy,
                )
            return findings, strategy, ""
        # The loop ran out of strategies rather than out of budget. Reaching
        # here means `attempts` never met `budget`, so `tracker_budget` cannot
        # have exceeded the ceiling: this exit is always an ordinary exhaustion,
        # never a refused configuration.
        return (
            None,
            "",
            self._record_stop(attempts, budget, tracker_budget, strategies_exhausted=True),
        )

    def _review_chunked(self, diff: str) -> list[Finding]:
        """Review a too-large diff one file at a time, and return the merged findings.

        Reached only from the review loop's capture point, for the one mapped
        failure class. It replaces the review phase and nothing else: the merged
        findings flow on to the same gate and verifier as an ordinary review, and
        the verifier renumbers them 1..n when it adjudicates.

        Two limits are deliberately NOT smoothed over, both by letting
        :func:`split_diff` raise straight through to the crash path:

        - a diff with no usable file header (:class:`MalformedDiffError`), and
        - more than ``MAX_CHUNKS`` files (:class:`TooManyFilesToChunk`).

        Neither becomes ``partial_review``: a run that reviewed nothing must not
        claim a partial review. The crash trail carries the cause — for the file
        cap, the count it saw and the cap it exceeded — which is what marks a
        declared limit apart from a silent truncation.

        Each piece gets ONE attempt, no retry: a piece whose reply will not parse
        (or is blank) is recorded as unreadable and the review moves on. The trail
        names how many pieces there were, which were unreadable, and the blind
        spot the split cannot cover.
        """
        pieces = split_diff(diff)  # Malformed / TooManyFiles propagate -> crash
        merged: list[Finding] = []
        unreadable: list[int] = []
        for idx, piece in enumerate(pieces):
            try:
                found, _dropped = self._reviewer.review(
                    piece,
                    instruction=CHUNK_DIFF_INSTRUCTION,
                    task=f"{TASK_REVIEW}:{CHUNK_DIFF_STRATEGY}",
                )
            except (ReviewParseError, EmptyCompletionError):
                unreadable.append(idx)  # one chance per piece, no retry
                continue
            merged.extend(found)
        self._audit.record(
            "chunk_review",
            f"context-length overflow: reviewed {len(pieces)} file-piece(s) "
            f"separately, {len(unreadable)} unreadable; a defect spanning two "
            f"files cannot be seen this way",
            chunks=len(pieces),
            unreadable_pieces=unreadable,
            coverage=("partial: each file reviewed alone; a cross-file defect is invisible"),
        )
        return merged

    def _verify_with_anti_loop(
        self, findings: list[Finding], diff: str, approval: object
    ) -> VerifyResult | None:
        """Adjudicate findings, retrying an unusable answer under the anti-loop.

        The verify phase used to sit outside the anti-loop, because it could not
        fail: any reply was accepted. Now that an unusable adjudication is an
        error, it gets the same treatment as review — including the budget
        living here, in the loop, so an injected tracker cannot spin it either.
        """
        # The approval authorises this phase once. Every exit spends it on the
        # way past — a usable adjudication, an exhausted budget, or an exception
        # leaving through here. `finally`, not a line after the return: a crash
        # inside verify must not leak a live token, the same discipline that
        # keeps `run` from ever exiting without a trail. Consuming here, not in
        # `verify()`, is deliberate: `verify()` is retried within the one phase,
        # and each retry is the same authorised action, not a fresh one.
        try:
            budget = min(self._tracker.max_attempts, MAX_TOTAL_ATTEMPTS)
            attempts = 0
            last = ""
            while attempts < budget:
                try:
                    return self._verifier.verify(findings, diff, approval)
                except (VerifyParseError, EmptyCompletionError) as exc:
                    # A blank adjudication is a failed attempt, symmetric with the
                    # review loop — it used to cross this loop untouched and crash
                    # the run. The failure-class map is NOT consulted here:
                    # chunk-diff is a review strategy with no verify equivalent, so
                    # verify widens only for the empty completion and any other
                    # exception propagates, exactly as before.
                    attempts += 1
                    last = str(exc)
                    outcome = self._record_failure(VERIFY_SUBTASK, VERIFY_STRATEGY)
                    if outcome.must_class_jump:
                        break
            reason = (
                STOP_TOTAL_CAP
                if self._tracker.max_attempts > MAX_TOTAL_ATTEMPTS
                else STOP_TRACKER_BUDGET
            )
            self._audit.record(
                "anti_loop",
                f"verify: no usable adjudication after {attempts} attempt(s) ({last})",
                stop_reason=reason,
                attempts=attempts,
                budget=budget,
            )
            return None
        finally:
            consume_approval(approval)

    def _record_failure(self, subtask_id: str, strategy: str) -> AttemptOutcome:
        """Record a failed attempt and check the verdict was really minted.

        ``tracker`` is a public constructor parameter, so the object these loops
        read ``must_class_jump`` off is caller-supplied. This is the same check
        ``Verifier.verify`` makes on its ``Approval``, against the same threat
        the threat model names: composition error, not a hostile process. A
        hand-rolled tracker returning a duck-typed stand-in gets a ``TypeError``
        here instead of quietly steering the loop.

        Defence in depth, and nothing more. It does not make the verdict
        tamper-proof: ``object.__setattr__`` still mutates a genuine outcome and
        this check still passes it (see :class:`AttemptOutcome`). And it is not
        the bound on retries — that is the budget computed above, in the loop.
        """
        outcome = self._tracker.record_failure(subtask_id, strategy=strategy)
        if not is_genuine_outcome(outcome):
            raise TypeError(
                "tracker returned a verdict AttemptTracker.record_failure() did "
                "not mint; a hand-built, forged, or duck-typed outcome is rejected"
            )
        return outcome

    def _record_stop(
        self,
        attempts: int,
        budget: int,
        tracker_budget: int,
        *,
        strategies_exhausted: bool = False,
    ) -> str:
        """Record why the loop cut itself short, and return the reason.

        A cut-off is never silent, and the review loop has three of them — not
        the two this docstring used to name. They are different events for
        whoever reads the trail:

        - :data:`STOP_STRATEGIES_EXHAUSTED` — every strategy was tried and none
          parsed. The anti-loop did its whole job; the model never produced
          usable output. This is the exit every default configuration takes,
          and it went through no ``_record_stop`` at all: the most common
          cut-off was the one the trail said nothing about.
        - :data:`STOP_TRACKER_BUDGET` — the budget ran out before the strategy
          list did. The anti-loop working as configured. An *unmodified*
          :class:`AttemptTracker` never reaches it: it grants the class-jump on
          the same attempt the budget runs out, so the strategy list always
          empties first. A subclass does reach it. One that mints through
          ``_mint_outcome`` returns genuine verdicts, so a perpetual ``retry``
          passes the check in :meth:`_record_failure`, the strategy index never
          advances, and the budget runs out on its own. Reachable, then — this
          said "unreachable today" and was measured otherwise. Kept, not fixed,
          because ``tracker`` is a public parameter and the invariant is the
          tracker's, not this loop's.
        - :data:`STOP_TOTAL_CAP` — the absolute ceiling was reached first: a
          configuration this package refused to honour.

        All three exits of *this* loop leave an ``anti_loop`` event with the
        same set of fields, so a reader learns one shape and reads
        ``stop_reason`` to tell them apart. The verify loop emits the same kind
        with a narrower set; aligning the two is not done here.
        """
        if strategies_exhausted:
            reason = STOP_STRATEGIES_EXHAUSTED
            message = (
                f"stopped after {attempts} attempts: every review strategy was "
                f"tried and none produced parseable findings "
                f"({self._tracker.max_attempts} attempts x "
                f"{len(self._strategies)} strategies)"
            )
        elif tracker_budget > MAX_TOTAL_ATTEMPTS:
            reason = STOP_TOTAL_CAP
            message = (
                f"stopped after {attempts} attempts: absolute ceiling "
                f"MAX_TOTAL_ATTEMPTS={MAX_TOTAL_ATTEMPTS} reached "
                f"(the configured budget was {tracker_budget})"
            )
        else:
            reason = STOP_TRACKER_BUDGET
            message = (
                f"stopped after {attempts} attempts: budget exhausted "
                f"({self._tracker.max_attempts} attempts x "
                f"{len(self._strategies)} strategies)"
            )
        self._audit.record(
            "anti_loop",
            message,
            stop_reason=reason,
            attempts=attempts,
            budget=budget,
            tracker_budget=tracker_budget,
            max_total_attempts=MAX_TOTAL_ATTEMPTS,
        )
        return reason

    def _fold_usage(self) -> None:
        """Pull accumulated usage from the client into the trail, once. Never raises.

        ``llm`` is a public constructor parameter, so reading ``totals`` is a
        call into a caller-supplied object — the one step of both finalize paths
        that is. It used to be unguarded in both, in two different ways that had
        the same consequence: outside the ``try`` in
        :meth:`_record_usage_and_finalize`, and *inside* the one in
        :meth:`_finalize_crashed` but before the ``finalize()`` it precedes. A
        broken accounting backend therefore produced no trail at all — the exact
        failure the crash path exists to prevent, surviving inside the guard
        that prevents it.

        The guard lives here rather than at the two call sites because the
        invariant is "folding usage cannot cost you the trail", and stated here
        it holds for every caller, including the next one. What a failure costs
        is the token and cost totals: a missing number inside a trail that
        exists, not a missing trail.
        """
        try:
            t = self._llm.totals
            self._audit.record_usage(
                prompt=int(t["prompt"]),
                completion=int(t["completion"]),
                cost_usd=t["cost_usd"],
                basis=getattr(self._llm, "cost_basis", None),
            )
        except Exception as exc:
            print(
                f"warning: the run's token and cost usage could not be folded into "
                f"the trail: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _fold_provider_stop_reasons(self) -> None:
        """Pull the run's raw provider stop reasons into the trail, once. Never raises.

        A sibling of :meth:`_fold_usage` rather than a branch inside it, and
        separate for the same reason the guard exists at all: ``llm`` is a public
        constructor parameter, so both reads are calls into a caller-supplied
        object, and one of them failing must not cost the other. A wrapper whose
        accounting backend is down still yields its stop reasons, and vice
        versa. What a failure here costs is the reasons — inside a trail that
        exists, with its token totals intact.

        Both fold sites call this, including the crash path, which is where it
        earns its keep: a run that ended because the provider truncated or
        refused is exactly the run whose trail would otherwise not say so.
        """
        try:
            self._audit.record_provider_stop_reasons(self._llm.stop_reasons)
        except Exception as exc:
            print(
                f"warning: the run's provider stop reasons could not be folded "
                f"into the trail: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _record_usage_and_finalize(self, status: str, result: str) -> str:
        """Fold usage in and write the trail. Never raises.

        This is the f1 path: the run reached a verdict and only the write
        failed. Raising here would turn a correct review into a failed one over
        a storage problem, so the caller gets its result and the reason the
        record is missing goes to stderr.

        The path returned is what is **on disk for this run**, not what was
        wanted — empty when this run created nothing. A reserved-but-empty
        directory, or a pair where only ``audit.json`` landed, are both real
        things a user can go and look at; a directory that was never created is
        not, and neither is one holding somebody else's trail.

        That last case is why the two failures are told apart. A reused
        ``AuditLog`` refuses the second finalize, and its directory is on disk
        and full — of the *first* run. Handing that path back would answer "no
        record was kept of this run" with a path to a record of another one,
        which is the precise untruth :attr:`AuditLog.run_dir_on_disk` exists to
        avoid. Reuse itself is not fixed here; the path stops lying about it.
        """
        self._fold_usage()
        self._fold_provider_stop_reasons()
        try:
            return str(self._audit.finalize(status, result))
        except Exception as exc:
            print(
                f"warning: run finished with status '{status}' but its audit trail "
                f"could not be written: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if isinstance(exc, AuditAlreadyFinalizedError):
                return ""
            return self._audit.run_dir_on_disk

    def _finalize_crashed(self, exc: BaseException) -> None:
        """Write the trail for a run that never reached a verdict.

        Never raises. It runs inside an exception handler, and a guard that
        blows up while saving is the same bug it exists to fix: the caller would
        get the storage failure *instead of* the thing that actually went wrong.
        So a failure here is reported on stderr and the original exception is
        left to continue untouched.
        """
        if self._audit.finalized:
            # Only `run` finalizes, exactly once. Getting here would mean a
            # finished trail is already on disk, and overwriting it with a crash
            # record would destroy the better of the two.
            return
        summary = f"run crashed: {type(exc).__name__}: {exc}"
        # Classify by WIRE SIGNAL, not by exception class — the class is not
        # stable across SDK versions (see failure_classifier). This path only
        # records: `_finalize_crashed` re-raises whatever it got, so a known
        # class changes nothing here and the map's destination is not acted on.
        # For an UNKNOWN, the raw signal is what makes the repo able to say,
        # later and with data, which class is worth adding next. Both calls are
        # inside the guard below because neither may cost the trail.
        try:
            signal = extract_signal(exc, getattr(self._llm, "provider", None))
            self._audit.record(
                "result",
                summary,
                error_type=type(exc).__name__,
                failure_class=classify_signal(signal),
                # Verbatim, so an UNKNOWN's unclassified signal is on the record.
                # http_status/body_type are null when the exception carried none
                # (e.g. the anthropic 1.4.0 "Streaming is required" ValueError).
                wire_signal={
                    "provider": signal.provider,
                    "http_status": signal.http_status,
                    "body_type": signal.body_type,
                    "message": signal.message,
                    # What extraction could not read, so an UNKNOWN's trace says
                    # *what* was missing, not just that classification failed.
                    "unresolved": list(signal.unresolved),
                },
                # The full traceback lives here so the entry points never have
                # to print one at the user.
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            self._fold_usage()
            self._fold_provider_stop_reasons()
            self._audit.finalize(STATUS_CRASHED, summary)
        except Exception as write_exc:
            print(
                f"warning: the run failed and its audit trail could not be written "
                f"either: {type(write_exc).__name__}: {write_exc}",
                file=sys.stderr,
            )
