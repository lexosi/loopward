"""run_demo.py — guardloop end-to-end, fully offline.

Shows all three reliability primitives in one run, with NO API key:

  1. anti-loop  — the first review strategy ("concise") never produces parseable
                  findings, so after 3 attempts the orchestrator class-jumps to
                  the "structured" strategy, which succeeds.
  2. stop-gate  — verification is gated; this demo uses 'auto' mode so it runs
                  unattended, but the gate decision is still recorded.
  3. audit      — a full trail (events, tokens, cost) is written to runs/<ts>/.

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from guardloop.agents.reviewer import parse_findings
from guardloop.engine.audit import AuditLog
from guardloop.engine.llm_wrapper import LLMClient, Message
from guardloop.engine.orchestrator import Orchestrator
from guardloop.engine.stop_gate import StopGate

SAMPLE = Path(__file__).parent / "sample_diff.patch"


def scripted_reviewer(messages: list[Message]) -> str:
    """Deterministic fake LLM that forces an anti-loop class-jump.

    - 'concise' strategy  -> returns chatty prose with no severity tags (parse fails)
    - 'structured' strategy -> returns valid severity-tagged findings
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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    diff = SAMPLE.read_text(encoding="utf-8")

    llm = LLMClient(provider="fake", fake_script=scripted_reviewer)
    gate = StopGate(mode="auto")  # unattended; decision still audited
    audit = AuditLog(run_id="demo-code-review")
    orch = Orchestrator(llm=llm, gate=gate, audit=audit)

    print("=" * 60)
    print("guardloop demo — offline, no API key")
    print("=" * 60)
    result = orch.run(diff)

    print("\n--- events ---")
    for e in audit.events:
        print(f"  [{e.kind}] {e.message}")

    print("\n--- result ---")
    print(f"  status:  {result.status}")
    print(f"  summary: {result.summary}")
    if result.verify is not None:
        for f in result.verify.confirmed:
            print(f"  confirmed: {f}")

    print(f"\n--- audit trail written to: {result.run_dir} ---")

    # Sanity: the structured-strategy output must parse.
    assert parse_findings(scripted_reviewer(
        [{"role": "system", "content": "strict code reviewer"}]
    ))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
