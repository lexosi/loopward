"""bench_antiloop.py — the honest number: a hard bound vs. an unbounded loop.

WHAT THIS MEASURES
------------------
On a code-review task that never converges (the model never emits a parseable
``SEVERITY:`` finding), guardloop's anti-loop caps spend at a HARD, deterministic
ceiling — it stops on its own in ``STATUS_EXHAUSTED`` after a fixed number of
LLM calls. A naive retry loop (what a dev writes without guardloop) does NOT
self-terminate: it only stops when a human or a timeout kills it. This script
measures both and prints the ratio.

CONDITIONS (read before trusting any figure)
--------------------------------------------
- **Fake provider, fully offline, deterministic.** No network, no API key. Token
  counts come from ``guardloop``'s own ``~len//4`` estimator, read straight from
  ``LLMClient.totals`` — nothing here is hardcoded.
- **Non-convergent task.** The fake reply is a fixed sentence with no
  ``SEVERITY:`` line, so ``parse_findings`` raises ``ReviewParseError`` on every
  call. This is the adversarial worst case, on purpose.
- **The hard bound is the measurement.** guardloop terminating in exactly
  ``MAX_ATTEMPTS * len(STRATEGIES)`` calls is measured and asserted, not assumed.
- **The ratio depends on K, which is DECLARED, not measured.** ``K`` (``--naive-kill``)
  is the operator's kill point — the assumption of how many blind retries a human
  or timeout tolerates before pulling the plug. It is an input, not a property of
  the code. Different K → different ratio. That is the whole point: the naive loop
  has no ceiling of its own.
- **Cost is fake=0.** The fake provider is not in ``PRICING`` so measured
  ``cost_usd`` is 0. The honest measured numbers are TOKENS and CALLS. A projected
  USD cost (multiplying measured tokens by a real provider's published rate via
  ``_calc_cost``) is offered as a clearly-labeled DERIVED projection, not a
  measurement.

REPRODUCE: python benchmarks/bench_antiloop.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Robustness: allow running from the repo root without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from guardloop.agents.reviewer import (  # noqa: E402  (after sys.path shim)
    STRATEGIES,
    Reviewer,
    ReviewParseError,
)
from guardloop.engine.anti_loop import MAX_ATTEMPTS  # noqa: E402
from guardloop.engine.llm_wrapper import PRICING, LLMClient, Message, _calc_cost  # noqa: E402
from guardloop.engine.orchestrator import STATUS_EXHAUSTED, Orchestrator  # noqa: E402
from guardloop.engine.stop_gate import GATE_AUTO, StopGate  # noqa: E402

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

# Default operator kill points to sweep (K = blind retries tolerated before kill).
DEFAULT_NAIVE_KILLS = [10, 25, 50, 100]

# Provider/model used only to PROJECT a USD cost from measured tokens (derived).
PROJECTION_PROVIDER = "deepseek"
PROJECTION_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class Measurement:
    """Usage read from LLMClient.totals after a run. Nothing hardcoded."""

    calls: int
    prompt: int
    completion: int

    @property
    def total_tokens(self) -> int:
        return self.prompt + self.completion


def _nonconvergent_reply(_messages: list[Message]) -> str:
    """Fake callable: always returns text with no parseable finding."""
    return NONCONVERGENT_REPLY


def _fresh_client() -> LLMClient:
    """A brand-new fake client (totals accumulate, so each run needs its own)."""
    return LLMClient(provider="fake", fake_script=_nonconvergent_reply)


def _read(client: LLMClient) -> Measurement:
    t = client.totals
    return Measurement(calls=int(t["calls"]), prompt=int(t["prompt"]), completion=int(t["completion"]))


def measure_guardloop() -> Measurement:
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


def project_cost(m: Measurement, provider: str, model: str) -> float:
    """DERIVED projection: measured tokens x a real provider's published rate.

    NOT a measurement. The fake provider costs 0; this multiplies the honest
    token counts by ``PRICING[(provider, model)]`` via the repo's own helper.
    """
    return _calc_cost(provider, model, m.prompt, m.completion)


def build_report(
    naive_kills: list[int],
    provider: str = PROJECTION_PROVIDER,
    model: str = PROJECTION_MODEL,
) -> dict:
    """Assemble every measured figure into a citable, machine-readable dict."""
    gl = measure_guardloop()
    priced = (provider, model) in PRICING
    rows = []
    for k in naive_kills:
        nv = measure_naive(k)
        rows.append(
            {
                "kill": k,
                "calls": nv.calls,
                "tokens": nv.total_tokens,
                "projected_cost_usd": project_cost(nv, provider, model),
                "ratio_tokens_vs_guardloop": round(nv.total_tokens / gl.total_tokens, 2),
                "ratio_calls_vs_guardloop": round(nv.calls / gl.calls, 2),
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
        "guardloop": {
            "status": STATUS_EXHAUSTED,
            "self_terminates": True,
            "calls": gl.calls,
            "tokens": gl.total_tokens,
            "prompt_tokens": gl.prompt,
            "completion_tokens": gl.completion,
            "projected_cost_usd": project_cost(gl, provider, model),
        },
        "naive": rows,
        "naive_self_terminates": False,
        "projection": {
            "provider": provider,
            "model": model,
            "priced": priced,
            "note": (
                "DERIVED, not measured: fake provider cost is 0. USD = measured "
                "tokens x published rate via _calc_cost. Unknown (provider,model) -> 0."
            ),
        },
    }


def format_table(report: dict) -> str:
    """Human-readable table. All figures come straight from the measured report."""
    gl = report["guardloop"]
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
            f"{gl['calls']:>8} | {gl['tokens']:>9} | {row['ratio_tokens_vs_guardloop']:>8}x"
        )
    lines.append("")
    lines.append(
        f"DERIVED cost projection ({proj['provider']}/{proj['model']}, "
        f"priced={proj['priced']}): guardloop ${gl['projected_cost_usd']:.6f} "
        "(naive per-K in --json). Fake measured cost = $0."
    )
    return "\n".join(lines)


def _claim(report: dict) -> str:
    hb = report["hard_bound"]
    gl = report["guardloop"]
    ks = ", ".join(str(r["kill"]) for r in report["naive"])
    return (
        "MEASURED CLAIM\n"
        f"- guardloop TERMINATES on its own in status {gl['status'].upper()} after "
        f"exactly {gl['calls']} LLM calls "
        f"({hb['formula']} = {hb['max_attempts']}*{hb['strategies']} = {hb['value']}), "
        "deterministic.\n"
        f"- naive does NOT self-terminate: it ran to each imposed kill K in {{{ks}}} "
        "and would keep going without one. Its ceiling is the operator, not the code."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure guardloop's hard anti-loop bound vs. an unbounded naive retry loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--naive-kill",
        type=int,
        nargs="+",
        default=DEFAULT_NAIVE_KILLS,
        help="Operator kill point(s) K: blind retries tolerated before kill (DECLARED, not measured).",
    )
    parser.add_argument("--project-provider", default=PROJECTION_PROVIDER, help="Provider for DERIVED cost projection.")
    parser.add_argument("--project-model", default=PROJECTION_MODEL, help="Model for DERIVED cost projection.")
    parser.add_argument("--json", action="store_true", help="Emit the measured report as machine-readable JSON.")
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
