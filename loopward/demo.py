"""demo.py — packaged offline demo (powers the `loopward-demo` console script).

Shows all three reliability primitives in one run, with NO API key:

  1. anti-loop  — the scripted model returns unparseable prose for the first
                  strategy ("concise"), so after 3 attempts the orchestrator
                  class-jumps to "structured", which succeeds.
  2. stop-gate  — verification is gated; this demo uses 'auto' mode so it runs
                  unattended, but the gate decision is still recorded.
  3. audit      — a full trail (events, tokens, cost) is written to runs/<ts>/.

THE FAILURE IS A FIXTURE, ON PURPOSE
------------------------------------
``scripted_reviewer`` below is written to fail. It routes on ``task`` and
returns prose for anything that is not the structured strategy — whatever the
prompt says, and it never reads the prompt at all. No model refused and no
prompt is at fault: this is a canned reply chosen to make the class-jump fire
deterministically and offline, the same way ``benchmarks/bench_antiloop.py``
uses a canned non-convergent reply. What this demo shows is the MECHANISM
firing. It has never shown it firing on a real model's real failure, and the
GIF in the README shows this same scripted run.

WHAT IT DOES NOT SHOW: A REJECTED FINDING
-----------------------------------------
``scripted_reviewer`` confirms every finding by construction, so the
``verify: confirmed 2, rejected 0`` line below is a foregone conclusion rather
than an adjudication. The built-in ``_fake_heuristic`` in ``llm_wrapper`` does
the same. The verifier's rejection path — the "drops false positives" half of
what it is for — appears in no runnable artifact of this repo; it is exercised
only in ``tests/test_verifier.py`` and ``tests/test_orchestrator.py``.

Entry points
------------
- ``main(argv)``  : ``python demo/run_demo.py`` (default delay 0.0 — fast for CI).
- ``console()``   : ``loopward-demo`` (default delay 0.4 — paced for recording).

The delay is presentation-only and never touches the engine.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from loopward.agents.reviewer import parse_findings
from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import TASK_REVIEW, TASK_VERIFY, LLMClient, Message
from loopward.engine.orchestrator import Orchestrator
from loopward.engine.stop_gate import StopGate

# Inline sample so the packaged demo works after install regardless of cwd.
# (A standalone demo/sample_diff.patch also exists for `loopward <file>` examples.)
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


def scripted_reviewer(messages: list[Message], task: str | None = None) -> str:
    """Deterministic fake LLM written to force an anti-loop class-jump.

    - 'concise' strategy    -> chatty prose with no severity tags (parse fails)
    - 'structured' strategy -> valid severity-tagged findings
    - verification request  -> confirms every listed finding

    The first line is the fixture: the failure is authored here, not observed.
    Both review strategies now ask for the severity-tag format, so a real model
    would not necessarily fail on 'concise' — this fake fails regardless,
    because a demo that only sometimes shows a class-jump shows nothing.

    Routing is on ``task``, never on the prompt's wording: a fake that reads
    the prompt breaks silently the day someone improves it.
    """
    if task == TASK_VERIFY:
        listing = "\n".join(m["content"] for m in messages if m.get("role") == "user")
        n = max(1, len(re.findall(r"^\s*(\d+)\.\s", listing, re.MULTILINE)))
        return "\n".join(f"CONFIRM {i}" for i in range(1, n + 1))
    if task == f"{TASK_REVIEW}:structured":
        return (
            "HIGH: token expiry uses `<=`; tokens are accepted exactly at expiry. Use `<`.\n"
            "HIGH: off-by-one; `range(len(items) + 1)` indexes one past the end."
        )
    # 'concise' strategy: unparseable on purpose (no SEVERITY: tags). Authored
    # to fail, not a transcript of a model that did — see the module docstring.
    return "Looks mostly fine, a couple of small things you might want to glance at."


def main(argv: list[str] | None = None, *, default_delay: float = 0.0) -> int:
    parser = argparse.ArgumentParser(description="loopward offline demo")
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
    line("loopward demo — offline, no API key")
    line("=" * 60)
    try:
        result = orch.run(SAMPLE_DIFF)
    except KeyboardInterrupt:
        return _report_failure(audit, "interrupted by user", 130)
    except Exception as exc:
        return _report_failure(audit, f"{type(exc).__name__}: {exc}", 1)

    line("\n--- events ---")
    for e in audit.events:
        line(f"  [{e.kind}] {e.message}")

    line("\n--- result ---")
    line(f"  status:  {result.status}")
    line(f"  summary: {result.summary}")
    if result.verify is not None:
        for f in result.verify.confirmed:
            line(f"  confirmed: {f}")

    if result.run_dir:
        line(f"\n--- audit trail written to: {result.run_dir} ---")

    # Sanity: the structured-strategy output must parse (explicit raise so this
    # holds even under `python -O`, which strips `assert`).
    _strict = [{"role": "system", "content": "structured"}]
    if not parse_findings(scripted_reviewer(_strict, f"{TASK_REVIEW}:structured")):
        raise RuntimeError(
            "demo self-check failed: structured strategy produced no parseable findings"
        )
    return 0 if result.status == "ok" else 1


def _report_failure(audit: AuditLog, reason: str, code: int) -> int:
    """Same contract as the CLI's handler: message, trail location, exit code.

    The demo is the recorded path, so it gets its own rather than inheriting a
    bare traceback. It does not finalize the trail either — ``Orchestrator.run``
    already did, as ``crashed``.
    """
    print(f"\nerror: demo run failed: {reason}", file=sys.stderr)
    trail = audit.run_dir_on_disk
    if trail:
        print(f"error: audit trail: {trail}", file=sys.stderr)
    else:
        print("error: no audit trail could be written", file=sys.stderr)
    return code


def console() -> int:
    """Console-script entry: paced output for screen recording (delay default 0.4)."""
    return main(default_delay=0.4)


if __name__ == "__main__":
    raise SystemExit(main())
