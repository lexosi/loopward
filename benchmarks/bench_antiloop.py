"""bench_antiloop.py — the honest number: a hard bound vs. an unbounded loop.

WHAT THIS MEASURES
------------------
On a code-review task that never converges (the model never emits a parseable
``SEVERITY:`` finding), loopward's anti-loop caps spend at a HARD, deterministic
ceiling — it stops on its own in ``STATUS_EXHAUSTED`` after a fixed number of
LLM calls. A naive retry loop (what a dev writes without loopward) does NOT
self-terminate: it only stops when a human or a timeout kills it. This script
measures both and prints the ratio.

TWO ceilings, not one
---------------------
The non-chunked ceiling above is ``MAX_ATTEMPTS * len(STRATEGIES)`` (6 by
default). A run that OVERFLOWS the context window takes a different path: it
reviews the diff one file at a time, and its ceiling is a SEPARATE, ADDITIVE sum
that ``MAX_TOTAL_ATTEMPTS`` does not fold ``MAX_CHUNKS`` under —

    MAX_ATTEMPTS * (len(STRATEGIES) - 1)   every EARLIER strategy parse-fails its
                                           whole budget and class-jumps
  + 1                                      the strategy that overflows, on its
                                           first call, triggering the split
  + MAX_CHUNKS                             one review per file-piece, single attempt
  + MAX_ATTEMPTS                           the verify loop's own budget

The whole-diff term is NOT just 1. The strategies do not share a prompt length —
``structured`` is a longer instruction than ``concise`` — so a diff can fit one
strategy and overflow the next. In the worst case every earlier strategy
parse-fails its full ``MAX_ATTEMPTS`` and class-jumps, and only the last strategy
overflows and routes to the split. An overflow on the FIRST strategy is a cheaper
case, not the ceiling. (Overflowing and parse-failing to exhaustion on the SAME
strategy are mutually exclusive: an overflow routes on that strategy's first
call.) This script MEASURES the worst case by running it, and asserts the count
against the formula DERIVED FROM THE CONSTANTS — add a strategy and the number
moves on its own; nothing is written from memory.

Reaching the chunk path needs a ``context_length_exceeded`` classification, which
the failure map gates to the ``claude`` provider — so the chunked path exists for
Anthropic only; another provider's overflow classifies UNKNOWN and crashes. The
offline vehicle is therefore a stubbed ``claude`` client (no network, no key,
deterministic) whose ``messages.create`` parse-fails the earlier strategy, raises
a duck-typed 400 on the overflowing one, and replies for each piece — the same
technique ``tests/test_chunk_route.py`` uses. It is a ``fake`` in the sense that
matters here: fully offline and deterministic.

CONDITIONS (read before trusting any figure)
--------------------------------------------
- **Fake provider, fully offline, deterministic.** No network, no API key. Token
  counts come from ``loopward``'s own ``~len//4`` estimator, read straight from
  ``LLMClient.totals`` — nothing here is hardcoded.
- **Non-convergent task.** The fake reply is a fixed sentence with no
  ``SEVERITY:`` line, so ``parse_findings`` raises ``ReviewParseError`` on every
  call. This is the adversarial worst case, on purpose.
- **The hard bound is the measurement.** loopward terminating in exactly
  ``MAX_ATTEMPTS * len(STRATEGIES)`` calls is measured and asserted, not assumed.
- **The ratio depends on K, which is DECLARED, not measured.** ``K`` (``--naive-kill``)
  is the operator's kill point — the assumption of how many blind retries a human
  or timeout tolerates before pulling the plug. It is an input, not a property of
  the code. Different K → different ratio. That is the whole point: the naive loop
  has no ceiling of its own.
- **Cost for the fake provider is null.** The fake has no rate in the benchmark's
  supplied ``EXAMPLE_RATES``, so its
  ``cost_usd`` is None (unpriced) — not 0, which would read as "free". CALLS are
  measured (counted invocations); TOKENS are an estimate (``len//4``), not
  measured — ``len//4`` is a formula, so calling that count "measured" would be
  the same untruth this benchmark refuses for cost. A projected USD cost
  (multiplying the estimated tokens by a real provider's published rate via
  ``_calc_cost``) is a clearly-labeled DERIVED projection, not a measurement.

REPRODUCE: python benchmarks/bench_antiloop.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Robustness: allow running from the repo root without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loopward.agents.chunk_diff import MAX_CHUNKS  # noqa: E402
from loopward.agents.reviewer import (  # noqa: E402  (after sys.path shim)
    _STRATEGY_INSTRUCTIONS,
    STRATEGIES,
    Reviewer,
    ReviewParseError,
)
from loopward.engine.anti_loop import MAX_ATTEMPTS  # noqa: E402
from loopward.engine.audit import AuditLog  # noqa: E402
from loopward.engine.llm_wrapper import LLMClient, Message, _calc_cost  # noqa: E402
from loopward.engine.orchestrator import STATUS_EXHAUSTED, Orchestrator  # noqa: E402
from loopward.engine.stop_gate import GATE_AUTO, StopGate  # noqa: E402

# A fixed, "hard" diff. Its content is irrelevant to the outcome (the fake never
# parses) but keeping it constant makes token counts fully deterministic.
HARD_DIFF = """\
--- a/auth/session.py
+++ b/auth/session.py
@@ -10,7 +10,12 @@ def validate(token, now):
-    return token.expires_at > now
+    for i in range(len(token.scopes) + 1):
+        try:
+            check(token.scopes[i])
+        except Exception:
+            pass
+    return token.expires_at <= now  # expiry boundary + swallowed errors
"""

# The non-convergent reply: no line begins with a SEVERITY tag, so
# parse_findings() raises ReviewParseError every single time.
NONCONVERGENT_REPLY = "I looked at the diff and everything seems fine to me overall."

# Theoretical hard ceiling; the script MEASURES and asserts this, never assumes it.
HARD_BOUND = MAX_ATTEMPTS * len(STRATEGIES)

# The chunked ceiling, DERIVED from the constants so a change to the strategy list
# moves it on its own: every earlier strategy parse-fails its whole budget
# (MAX_ATTEMPTS each) and class-jumps, the last one overflows on its first call and
# triggers the split, then one review per file-piece (MAX_CHUNKS) and the verify
# loop's budget (MAX_ATTEMPTS). Separate from HARD_BOUND and additive — MAX_CHUNKS
# is NOT folded under it. MEASURED and asserted below, never assumed.
_EARLIER_STRATEGY_PARSE_FAILS = MAX_ATTEMPTS * (len(STRATEGIES) - 1)
CHUNK_HARD_BOUND = _EARLIER_STRATEGY_PARSE_FAILS + 1 + MAX_CHUNKS + MAX_ATTEMPTS

# Default operator kill points to sweep (K = blind retries tolerated before kill).
DEFAULT_NAIVE_KILLS = [10, 25, 50, 100]

# Provider/model used only to PROJECT a USD cost from measured tokens (derived).
PROJECTION_PROVIDER = "deepseek"
PROJECTION_MODEL = "deepseek-v4-flash"

# EXAMPLE rates the benchmark uses to project a cost — supplied HERE, by the
# caller, exactly as loopward now requires of anyone who wants a cost figure. The
# library asserts no rates; these are the benchmark's own, and they are dated so
# they cannot age in silence. Flat DeepSeek rates this repo had verified on
# 2026-07-31; DeepSeek's live pricing is now time-of-day dependent, so these are
# an illustrative constant for a reproducible projection, NOT a current price.
EXAMPLE_RATES_LABEL = "loopward benchmark example rates, verified 2026-07-31"
EXAMPLE_RATES: dict[tuple[str, str], dict[str, float]] = {
    ("deepseek", "deepseek-v4-flash"): {"prompt": 0.14, "completion": 0.28},
    ("deepseek", "deepseek-v4-pro"): {"prompt": 0.435, "completion": 0.87},
    ("claude", "claude-haiku-4-5"): {"prompt": 1.0, "completion": 5.0},
    ("claude", "claude-sonnet-4-6"): {"prompt": 3.0, "completion": 15.0},
}


@dataclass(frozen=True)
class Measurement:
    """Usage read from LLMClient.totals after a run. Nothing hardcoded."""

    calls: int
    prompt: int
    completion: int

    @property
    def total_tokens(self) -> int:
        return self.prompt + self.completion


def _nonconvergent_reply(_messages: list[Message], _task: str | None) -> str:
    """Fake callable: always returns text with no parseable finding."""
    return NONCONVERGENT_REPLY


def _fresh_client() -> LLMClient:
    """A brand-new fake client (totals accumulate, so each run needs its own)."""
    return LLMClient(provider="fake", fake_script=_nonconvergent_reply)


def _read(client: LLMClient) -> Measurement:
    t = client.totals
    return Measurement(
        calls=int(t["calls"]),
        prompt=int(t["prompt"]),
        completion=int(t["completion"]),
    )


def measure_loopward() -> Measurement:
    """Run the orchestrator on the non-convergent task; assert the hard bound."""
    client = _fresh_client()
    orchestrator = Orchestrator(llm=client, gate=StopGate(GATE_AUTO))
    result = orchestrator.run(HARD_DIFF)
    if result.status != STATUS_EXHAUSTED:
        raise AssertionError(
            f"expected status {STATUS_EXHAUSTED!r}, got {result.status!r} "
            "(the fake was supposed to never converge)"
        )
    measured = _read(client)
    if measured.calls != HARD_BOUND:
        raise AssertionError(
            f"measured {measured.calls} calls but the hard bound is "
            f"MAX_ATTEMPTS*len(STRATEGIES) = {MAX_ATTEMPTS}*{len(STRATEGIES)} = {HARD_BOUND}"
        )
    return measured


def measure_naive(naive_kill: int) -> Measurement:
    """A blind retry loop with no real ceiling — stops only at the imposed kill K."""
    client = _fresh_client()
    reviewer = Reviewer(client)
    for _ in range(naive_kill):
        try:
            reviewer.review(HARD_DIFF, strategy="concise")
            break  # would stop early IF it ever parsed — it never does here
        except ReviewParseError:
            continue  # blind retry, same strategy, no self-imposed bound
    return _read(client)


# --- the chunked ceiling ----------------------------------------------------
# A context-length overflow reviews the diff a file at a time. Reaching that path
# needs a `context_length_exceeded` classification, gated to the `claude` provider
# by the failure map, so the offline vehicle is a stubbed `claude` client: no
# network, no key, deterministic. Same technique as tests/test_chunk_route.py.

_OVERFLOW_MSG = "prompt is too long: 250000 tokens > 200000 maximum"

# The instruction the LAST strategy sends — used to tell "the overflowing strategy"
# apart from the earlier ones by their own wording, so the worst-case scenario
# tracks the strategy list instead of a hardcoded name.
_LAST_STRATEGY_INSTRUCTION = _STRATEGY_INSTRUCTIONS[STRATEGIES[-1]]


class _StubOverflow400(Exception):
    """Duck-types the fields ``extract_signal`` reads off a real BadRequestError."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400
        self.type = "invalid_request_error"
        self.message = message


def _stub_resp(text: str):
    block = type("_Block", (), {"type": "text", "text": text})()
    return type("_Resp", (), {"content": [block], "stop_reason": None})()


def _bare_file(name: str) -> str:
    """One minimal, standalone single-file diff (its own header + one hunk)."""
    return f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-x\n+y\n"


@dataclass(frozen=True)
class ChunkMeasurement:
    """Usage from a chunked run. ``provider_calls`` counts EVERY invocation of the
    stub's ``messages.create`` — including the call that overflows, which raises
    before ``LLMClient`` increments its own ``calls``. That is the honest ceiling;
    ``counted_calls`` (from ``totals``) omits the raised one."""

    status: str
    provider_calls: int
    counted_calls: int
    tokens: int


def _measure_chunked(verify_fails: bool) -> ChunkMeasurement:
    """Run the worst-case chunked scenario and read every measured figure.

    The earlier strategy parse-fails its whole budget and class-jumps; the last
    strategy overflows and routes to the per-file split. ``verify_fails=True`` then
    makes the verify loop burn its whole budget (the absolute spend ceiling);
    ``False`` lets it confirm once (a run that reaches a verdict,
    ``partial_review``). Nothing is hardcoded — call counts come from the stub's
    own tally and ``LLMClient.totals``.
    """
    diff = "".join(_bare_file(f"f{i}.py") for i in range(MAX_CHUNKS))
    tally = {"create": 0}

    def handler(system: str, messages: list[Message]):
        tally["create"] += 1
        low = system.lower()
        if "adjudicate" in low:  # the verify phase
            if verify_fails:
                return _stub_resp("cannot adjudicate this")  # unparseable -> retry
            listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
            k = max(1, len(re.findall(r"^\s*(\d+)\.\s", listing, re.MULTILINE)))
            return _stub_resp("\n".join(f"CONFIRM {i}" for i in range(1, k + 1)))
        if "one file from a larger diff" in low:  # a per-file piece review
            return _stub_resp("HIGH: issue in this file.")
        if _LAST_STRATEGY_INSTRUCTION.lower() in low:
            # The LAST strategy in the list overflows — reached only after every
            # earlier strategy has parse-failed its whole budget and class-jumped.
            # Keyed off the strategy's own instruction text, so this stays correct
            # if the list is reordered or extended.
            raise _StubOverflow400(_OVERFLOW_MSG)
        # An earlier strategy: fits, but returns unparseable prose -> ReviewParseError,
        # spending an attempt until the anti-loop class-jumps to the next strategy.
        return _stub_resp("looks mostly fine, nothing tagged")

    class _Messages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            return handler(system or "", messages)

    client = LLMClient(provider="claude", model="claude-haiku-4-5")
    client._client = type("_Client", (), {"messages": _Messages()})()
    audit = AuditLog(run_id="bench-chunk")
    result = Orchestrator(llm=client, gate=StopGate(GATE_AUTO), audit=audit).run(diff)
    t = client.totals
    return ChunkMeasurement(
        status=result.status,
        provider_calls=tally["create"],
        counted_calls=int(t["calls"]),
        tokens=int(t["prompt"]) + int(t["completion"]),
    )


def measure_loopward_chunked() -> ChunkMeasurement:
    """Measure the chunked spend ceiling; assert it against the DERIVED formula.

    The worst case: every earlier strategy parse-fails its whole budget, the last
    strategy overflows and routes to the split, and the verify budget then runs
    out too — the run ends EXHAUSTED after the maximum number of provider calls.
    That count must equal ``MAX_ATTEMPTS*(len(STRATEGIES)-1) + 1 + MAX_CHUNKS +
    MAX_ATTEMPTS``, computed from the constants so a new strategy moves it on its
    own. Measured by running it, then checked — never written from memory.
    """
    worst = _measure_chunked(verify_fails=True)
    if worst.status != STATUS_EXHAUSTED:
        raise AssertionError(
            f"expected {STATUS_EXHAUSTED!r} for the verify-exhausting worst case, "
            f"got {worst.status!r}"
        )
    if worst.provider_calls != CHUNK_HARD_BOUND:
        raise AssertionError(
            f"measured {worst.provider_calls} provider calls but the chunked bound "
            f"is MAX_ATTEMPTS*(len(STRATEGIES)-1) + 1 + MAX_CHUNKS + MAX_ATTEMPTS = "
            f"{MAX_ATTEMPTS}*{len(STRATEGIES) - 1} + 1 + {MAX_CHUNKS} + {MAX_ATTEMPTS} "
            f"= {CHUNK_HARD_BOUND}"
        )
    return worst


def project_cost(m: Measurement, provider: str, model: str) -> float:
    """DERIVED projection: estimated (len//4) tokens x a real provider's published rate.

    NOT a measurement. The fake provider is unpriced (cost None); this multiplies
    the estimated (``len//4``) token counts by the benchmark's own
    ``EXAMPLE_RATES[(provider, model)]`` via the repo's helper — a rate supplied
    by this caller, not one the library asserts.
    """
    return _calc_cost(provider, model, m.prompt, m.completion, EXAMPLE_RATES)


def build_report(
    naive_kills: list[int],
    provider: str = PROJECTION_PROVIDER,
    model: str = PROJECTION_MODEL,
) -> dict:
    """Assemble every measured figure into a citable, machine-readable dict."""
    gl = measure_loopward()
    chunk_worst = measure_loopward_chunked()  # asserts the additive formula
    chunk_verdict = _measure_chunked(verify_fails=False)  # a run that reaches a verdict
    priced = (provider, model) in EXAMPLE_RATES
    rows = []
    for k in naive_kills:
        nv = measure_naive(k)
        rows.append(
            {
                "kill": k,
                "calls": nv.calls,
                "tokens": nv.total_tokens,
                "projected_cost_usd": project_cost(nv, provider, model),
                "ratio_tokens_vs_loopward": round(nv.total_tokens / gl.total_tokens, 2),
                "ratio_calls_vs_loopward": round(nv.calls / gl.calls, 2),
            }
        )
    return {
        "conditions": {
            "provider": "fake",
            "offline": True,
            "deterministic": True,
            "task": "non-convergent code-review (reply has no SEVERITY: line)",
            "k_is_declared_not_measured": True,
        },
        "hard_bound": {
            "formula": "MAX_ATTEMPTS * len(STRATEGIES)",
            "max_attempts": MAX_ATTEMPTS,
            "strategies": len(STRATEGIES),
            "value": HARD_BOUND,
        },
        "loopward": {
            "status": STATUS_EXHAUSTED,
            "self_terminates": True,
            "calls": gl.calls,
            "tokens": gl.total_tokens,
            "prompt_tokens": gl.prompt,
            "completion_tokens": gl.completion,
            "projected_cost_usd": project_cost(gl, provider, model),
        },
        "chunked_bound": {
            "formula": "MAX_ATTEMPTS*(len(STRATEGIES)-1) + 1 + MAX_CHUNKS + MAX_ATTEMPTS",
            "earlier_strategy_parse_fails": _EARLIER_STRATEGY_PARSE_FAILS,
            "overflow_call": 1,
            "max_chunks": MAX_CHUNKS,
            "verify_max_attempts": MAX_ATTEMPTS,
            "value": CHUNK_HARD_BOUND,
            "note": (
                "Separate, additive ceiling for a run that overflows the context "
                "window and reviews the diff a file at a time. The whole-diff term "
                "is NOT 1: a diff can fit an earlier strategy and overflow a later "
                "one (their prompt lengths differ), so every earlier strategy can "
                "parse-fail its whole budget before the overflow. Anthropic only: "
                "the failure map gates context_length to provider='claude'; another "
                "provider's overflow classifies UNKNOWN and crashes. MAX_CHUNKS is "
                "NOT folded under MAX_TOTAL_ATTEMPTS."
            ),
        },
        "chunked": {
            "worst_case": {
                "status": chunk_worst.status,
                "provider_calls": chunk_worst.provider_calls,
                "counted_calls": chunk_worst.counted_calls,
                "tokens": chunk_worst.tokens,
                "note": "verify budget also exhausted -> maximum spend",
            },
            "reaches_verdict": {
                "status": chunk_verdict.status,
                "provider_calls": chunk_verdict.provider_calls,
                "counted_calls": chunk_verdict.counted_calls,
                "tokens": chunk_verdict.tokens,
                "note": "verify confirms once -> partial_review",
            },
        },
        "naive": rows,
        "naive_self_terminates": False,
        "projection": {
            "provider": provider,
            "model": model,
            "priced": priced,
            "rates_label": EXAMPLE_RATES_LABEL,
            "note": (
                "DERIVED, not measured: fake provider is unpriced (cost None). "
                "USD = estimated (len//4) tokens x a rate this benchmark supplies "
                "(EXAMPLE_RATES), not one the library asserts, via _calc_cost. "
                "Pair not in the supplied rates -> None."
            ),
        },
    }


def format_table(report: dict) -> str:
    """Human-readable table. All figures come straight from the measured report."""
    gl = report["loopward"]
    proj = report["projection"]
    lines: list[str] = []
    header = (
        f"{'K (kill)':>9} | {'naive calls':>11} | {'naive tokens':>12} | "
        f"{'gl calls':>8} | {'gl tokens':>9} | {'tok ratio':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["naive"]:
        lines.append(
            f"{row['kill']:>9} | {row['calls']:>11} | {row['tokens']:>12} | "
            f"{gl['calls']:>8} | {gl['tokens']:>9} | {row['ratio_tokens_vs_loopward']:>8}x"
        )
    lines.append("")
    lines.append(
        f"DERIVED cost projection ({proj['provider']}/{proj['model']}, "
        f"priced={proj['priced']}): loopward ${gl['projected_cost_usd']:.6f} "
        "(naive per-K in --json). Fake provider is unpriced -> cost null."
    )
    cb = report["chunked_bound"]
    cw = report["chunked"]["worst_case"]
    expansion = (
        f"{MAX_ATTEMPTS}*{len(STRATEGIES) - 1} + 1 + "
        f"{cb['max_chunks']} + {cb['verify_max_attempts']}"
    )
    lines.append("")
    lines.append(
        f"CHUNKED ceiling (Anthropic-only context-window overflow -> per-file "
        f"review): {cb['value']} provider calls, MEASURED = {cw['provider_calls']} "
        f"({cb['formula']} = {expansion}). Additive to the "
        f"{report['hard_bound']['value']}-call bound above, not folded under it."
    )
    return "\n".join(lines)


def _claim(report: dict) -> str:
    hb = report["hard_bound"]
    cb = report["chunked_bound"]
    gl = report["loopward"]
    ks = ", ".join(str(r["kill"]) for r in report["naive"])
    return (
        "MEASURED CLAIM\n"
        f"- loopward TERMINATES on its own in status {gl['status'].upper()} after "
        f"exactly {gl['calls']} LLM calls "
        f"({hb['formula']} = {hb['max_attempts']}*{hb['strategies']} = {hb['value']}), "
        "deterministic.\n"
        f"- naive does NOT self-terminate: it ran to each imposed kill K in {{{ks}}} "
        "and would keep going without one. Its ceiling is the operator, not the code.\n"
        f"- a run that overflows the context window reviews the diff a file at a "
        f"time (Anthropic only), with its own SEPARATE, ADDITIVE ceiling of "
        f"{cb['value']} provider calls ({cb['formula']} = "
        f"{MAX_ATTEMPTS}*{len(STRATEGIES) - 1} + 1 + {cb['max_chunks']} + "
        f"{cb['verify_max_attempts']}), MEASURED and asserted."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure loopward's hard anti-loop bound vs. an unbounded naive retry loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--naive-kill",
        type=int,
        nargs="+",
        default=DEFAULT_NAIVE_KILLS,
        help=(
            "Operator kill point(s) K: blind retries tolerated before kill "
            "(DECLARED, not measured)."
        ),
    )
    parser.add_argument(
        "--project-provider",
        default=PROJECTION_PROVIDER,
        help="Provider for DERIVED cost projection.",
    )
    parser.add_argument(
        "--project-model",
        default=PROJECTION_MODEL,
        help="Model for DERIVED cost projection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the measured report as machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.naive_kill, provider=args.project_provider, model=args.project_model)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(format_table(report))
    print()
    print(_claim(report))
    print()
    print("REPRODUCE: python benchmarks/bench_antiloop.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
