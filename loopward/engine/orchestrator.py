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
:class:`AuditLog`. The orchestrator is the only component that calls agents;
agents never call each other.

All collaborators are injected, so the whole flow runs offline and is trivial to
test with the ``fake`` provider.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

from loopward.agents.reviewer import STRATEGIES, Finding, Reviewer, ReviewParseError
from loopward.agents.verifier import Verifier, VerifyParseError, VerifyResult
from loopward.engine.anti_loop import AttemptOutcome, AttemptTracker, is_genuine_outcome
from loopward.engine.audit import AuditAlreadyFinalizedError, AuditLog
from loopward.engine.llm_wrapper import LLMClient
from loopward.engine.stop_gate import StopGate

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

#: Absolute ceiling on review attempts in a single run, whatever the injected
#: tracker and strategy list add up to. This is a SPEND limit, not a design
#: limit: the anti-loop's real budget is ``max_attempts * len(strategies)``
#: (6 by default, and the benchmark asserts it), and this exists only so that
#: no caller-supplied number can turn a bounded loop into an unbounded bill.
MAX_TOTAL_ATTEMPTS = 100

# Why the review loop stopped, recorded so a trail reader can tell an ordinary
# exhaustion from a ceiling being hit — they mean different things.
STOP_STRATEGIES_EXHAUSTED = "strategies_exhausted"  # every strategy was tried
STOP_TRACKER_BUDGET = "tracker_budget"  # max_attempts x strategies used up
STOP_TOTAL_CAP = "total_cap"  # MAX_TOTAL_ATTEMPTS reached first


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

        if findings is None:
            if stop_reason == STOP_TOTAL_CAP:
                summary = (
                    f"review stopped at the absolute ceiling of "
                    f"{MAX_TOTAL_ATTEMPTS} attempts without parseable findings"
                )
            else:
                summary = "review exhausted all strategies without parseable findings"
            self._audit.record("result", summary)
            run_dir = self._record_usage_and_finalize(STATUS_EXHAUSTED, summary)
            return RunResult(STATUS_EXHAUSTED, [], None, str(run_dir), summary)

        # The findings themselves ride in `data`, never in `message`: the
        # message is what the demo and the README quote, and it stays
        # byte-identical. `verified=False` is not decoration — at this point
        # these are the reviewer's claims and nothing has adjudicated them. A
        # trail that stored them unmarked would be filing the generator's
        # output as if it were the evaluator's verdict.
        self._audit.record(
            "phase",
            f"review: ok via strategy '{strategy}' ({len(findings)} findings)",
            stage="review",
            verified=False,
            findings=[str(f) for f in findings],
        )

        gate_summary = f"{len(findings)} findings ready to verify"
        decision = self._gate.request("verify", gate_summary)
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

        summary = (
            f"{len(verify.confirmed)} confirmed finding(s)"
            + (", blocking" if verify.has_blocking else ", none blocking")
        )
        self._audit.record("result", summary)
        run_dir = self._record_usage_and_finalize(STATUS_OK, summary)
        return RunResult(STATUS_OK, findings, verify, str(run_dir), summary)

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
            except ReviewParseError:
                attempts += 1
                outcome = self._record_failure(REVIEW_SUBTASK, strategy)
                if outcome.must_class_jump:
                    self._audit.record(
                        "class_jump",
                        f"switching review strategy after {outcome.attempt} failures "
                        f"on '{strategy}'",
                    )
                    self._tracker.reset(REVIEW_SUBTASK)
                    strat_idx += 1
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
        return None, "", STOP_STRATEGIES_EXHAUSTED

    def _verify_with_anti_loop(
        self, findings: list[Finding], diff: str, approval: object
    ) -> VerifyResult | None:
        """Adjudicate findings, retrying an unusable answer under the anti-loop.

        The verify phase used to sit outside the anti-loop, because it could not
        fail: any reply was accepted. Now that an unusable adjudication is an
        error, it gets the same treatment as review — including the budget
        living here, in the loop, so an injected tracker cannot spin it either.
        """
        budget = min(self._tracker.max_attempts, MAX_TOTAL_ATTEMPTS)
        attempts = 0
        last = ""
        while attempts < budget:
            try:
                return self._verifier.verify(findings, diff, approval)
            except VerifyParseError as exc:
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

    def _record_stop(self, attempts: int, budget: int, tracker_budget: int) -> str:
        """Record why the loop cut itself short, and return the reason.

        A cut-off is never silent: hitting the tracker's own budget and hitting
        the absolute ceiling are different events for whoever reads the trail —
        the first is the anti-loop working as configured, the second is a
        configuration this package refused to honour.
        """
        if tracker_budget > MAX_TOTAL_ATTEMPTS:
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
        """Pull accumulated usage from the client into the trail, once."""
        t = self._llm.totals
        self._audit.record_usage(
            prompt=int(t["prompt"]), completion=int(t["completion"]), cost_usd=t["cost_usd"]
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
        try:
            self._audit.record(
                "result",
                summary,
                error_type=type(exc).__name__,
                # The full traceback lives here so the entry points never have
                # to print one at the user.
                traceback="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            self._fold_usage()
            self._audit.finalize(STATUS_CRASHED, summary)
        except Exception as write_exc:
            print(
                f"warning: the run failed and its audit trail could not be written "
                f"either: {type(write_exc).__name__}: {write_exc}",
                file=sys.stderr,
            )
