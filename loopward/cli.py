"""cli.py — `loopward` / `python -m loopward`.

Runs a code-review over a diff file with the reliability engine wired up.
Defaults to the offline ``fake`` provider so it works with no API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import DEFAULT_MODEL, LLMClient
from loopward.engine.orchestrator import Orchestrator
from loopward.engine.stop_gate import StopGate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loopward",
        description="Run a code-review with anti-loop, stop-gate, and audit.",
    )
    p.add_argument("diff", type=Path, help="path to a diff/patch file to review")
    p.add_argument(
        "--provider",
        default="fake",
        choices=["fake", "deepseek", "claude"],
        help="LLM provider (default: fake, offline, no key needed)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="model id (default: per-provider sensible default)",
    )
    p.add_argument(
        "--gate",
        default="interactive",
        choices=["interactive", "auto", "deny"],
        help="stop-gate mode (default: interactive)",
    )
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="base directory for audit output (default: ./runs)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows terminals default to cp1252; force UTF-8 so non-ASCII output is clean.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    args = build_parser().parse_args(argv)

    if not args.diff.exists():
        print(f"error: diff file not found: {args.diff}", file=sys.stderr)
        return 2
    diff = args.diff.read_text(encoding="utf-8")

    model = args.model or DEFAULT_MODEL[args.provider]
    llm = LLMClient(provider=args.provider, model=model)
    gate = StopGate(mode=args.gate)
    audit = AuditLog(run_id="code-review", base_dir=args.runs_dir)
    orch = Orchestrator(llm=llm, gate=gate, audit=audit)

    print(f"[loopward] provider={args.provider} model={model} gate={args.gate}")
    result = orch.run(diff)

    print(f"\n[loopward] status: {result.status}")
    print(f"[loopward] {result.summary}")
    if result.verify is not None:
        for f in result.verify.confirmed:
            print(f"  confirmed: {f}")
        for f in result.verify.rejected:
            print(f"  rejected:  {f}")
    print(f"[loopward] audit trail: {result.run_dir}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
