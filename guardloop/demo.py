"""demo.py — packaged offline demo (powers the `guardloop-demo` console script).

Shows all three reliability primitives in one run, with NO API key:

  1. anti-loop  — the first review strategy ("concise") never produces parseable
                  findings, so after 3 attempts the orchestrator class-jumps to
                  the "structured" strategy, which succeeds.
  2. stop-gate  — verification is gated; this demo uses 'auto' mode so it runs
                  unattended, but the gate decision is still recorded.
  3. audit      — a full trail (events, tokens, cost) is written to runs/<ts>/.

Entry points
------------
- ``main(argv)``  : ``python demo/run_demo.py`` (default delay 0.0 — fast for CI).
- ``console()``   : ``guardloop-demo`` (default delay 0.4 — paced for recording).

The delay is presentation-only and never touches the engine.
"""

from __future__ import annotations

import argparse
import sys
import time

from guardloop.agents.reviewer import parse_findings
from guardloop.engine.audit import AuditLog
from guardloop.engine.llm_wrapper import LLMClient, Message
from guardloop.engine.orchestrator import Orchestrator
from guardloop.engine.stop_gate import StopGate

# Inline sample so the packaged demo works after install regardless of cwd.
# (A standalone demo/sample_diff.patch also exists for `guardloop <file>` examples.)
SAMPLE_DIFF = """\
--- a/auth/session.py
+++ b/auth/session.py
@@ def is_valid(self, now):
-        return now < self.expires_at
+        return now <= self.expires_at   # accept tokens exactly at expiry
@@ def load_all(items):
-    for i in range(len(items)):
+    for i in range(len(items) + 1):      # iterate one past the end
         results.append(transform(items[i]))
"""


def scripted_reviewer(messages: list[Message]) -> str:
    """Deterministic fake LLM that forces an anti-loop class-jump.

    - 'concise' strategy   -> chatty prose with no severity tags (parse fails)
    - 'structured' strategy -> valid severity-tagged findings
    - verification request  -> confirms everything
    """
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    if "confirm or reject each finding" in system.lower():
        return "CONFIRM all findings."
    if "strict code reviewer" in system.lower():
        return (
            "HIGH: token expiry uses `<=`; tokens are accepted exactly at expiry. Use `<`.\n"
            "HIGH: off-by-one; `range(len(items) + 1)` indexes one past the end."
        )
    # 'concise' strategy: unparseable on purpose (no SEVERITY: tags).
    return "Looks mostly fine, a couple of small things you might want to glance at."


def main(argv: list[str] | None = None, *, default_delay: float = 0.0) -> int:
    parser = argparse.ArgumentParser(description="guardloop offline demo")
    parser.add_argument(
        "--demo-delay",
        type=float,
        default=default_delay,
        help="seconds to pause between printed lines (presentation only). "
        "0.0 = no delay (fast for CI). Try 0.4 for screen recording.",
    )
    args = parser.parse_args(argv)
    delay = max(0.0, args.demo_delay)

    def line(text: str) -> None:
        """Print one line, then pause `delay` seconds (presentation pacing only)."""
        print(text)
        if delay:
            time.sleep(delay)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    llm = LLMClient(provider="fake", fake_script=scripted_reviewer)
    gate = StopGate(mode="auto")  # unattended; decision still audited
    audit = AuditLog(run_id="demo-code-review")
    orch = Orchestrator(llm=llm, gate=gate, audit=audit)

    line("=" * 60)
    line("guardloop demo — offline, no API key")
    line("=" * 60)
    result = orch.run(SAMPLE_DIFF)

    line("\n--- events ---")
    for e in audit.events:
        line(f"  [{e.kind}] {e.message}")

    line("\n--- result ---")
    line(f"  status:  {result.status}")
    line(f"  summary: {result.summary}")
    if result.verify is not None:
        for f in result.verify.confirmed:
            line(f"  confirmed: {f}")

    line(f"\n--- audit trail written to: {result.run_dir} ---")

    # Sanity: the structured-strategy output must parse.
    _strict = [{"role": "system", "content": "strict code reviewer"}]
    assert parse_findings(scripted_reviewer(_strict))
    return 0 if result.status == "ok" else 1


def console() -> int:
    """Console-script entry: paced output for screen recording (delay default 0.4)."""
    return main(default_delay=0.4)


if __name__ == "__main__":
    raise SystemExit(main())
