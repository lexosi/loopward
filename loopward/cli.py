"""cli.py — `loopward` / `python -m loopward`.

Runs a code-review over a diff file with the reliability engine wired up.
Defaults to the offline ``fake`` provider so it works with no API key.

A bad diff argument — missing, or present but unreadable as UTF-8 text — is
rejected here, before any run exists, so it correctly leaves no audit trail:
the guarantee is not that every invocation writes one, but that no run ever
ends without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from loopward import __version__
from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import DEFAULT_MODEL, LLMClient
from loopward.engine.orchestrator import Orchestrator
from loopward.engine.stop_gate import StopGate


def report_failure(audit: AuditLog, reason: str, code: int) -> int:
    """Report a run that failed: what happened, where the trail is, what code.

    Three things and no more. In particular it does NOT finalize anything —
    :meth:`Orchestrator.run` owns the trail and has already written it as
    ``crashed`` before the exception reached here. And it never prints a
    traceback: the full one is inside the trail, where it can be read later
    rather than scrolled past now.
    """
    print(f"\nerror: run failed: {reason}", file=sys.stderr)
    trail = audit.run_dir_on_disk
    if trail:
        print(f"error: audit trail: {trail}", file=sys.stderr)
    else:
        print("error: no audit trail could be written", file=sys.stderr)
    return code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loopward",
        description="Run a code-review with anti-loop, stop-gate, and audit.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"loopward {__version__}",
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
    try:
        diff = args.diff.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        # Same category as "not found": the diff argument yields no usable diff,
        # detected before any run or trail exists. UnicodeDecodeError (non-UTF-8
        # bytes) and OSError (a directory, a permission failure, a file removed
        # after the exists() check) both mean "cannot read this diff" — reported
        # with the shape of the not-found error, exit code 2, and no traceback.
        # Decoding errors are surfaced, never repaired: errors="replace" or
        # sniffing an encoding would turn an unreadable input into a silently
        # different diff, which is worse than refusing it.
        print(f"error: could not read diff file: {args.diff}: {exc}", file=sys.stderr)
        return 2

    model = args.model or DEFAULT_MODEL[args.provider]
    llm = LLMClient(provider=args.provider, model=model)
    gate = StopGate(mode=args.gate)
    audit = AuditLog(run_id="code-review", base_dir=args.runs_dir)
    orch = Orchestrator(llm=llm, gate=gate, audit=audit)

    print(f"[loopward] provider={args.provider} model={model} gate={args.gate}")
    try:
        result = orch.run(diff)
    except KeyboardInterrupt:
        # 130 = 128 + SIGINT. "The user stopped it" and "the diff has problems"
        # are different outcomes; the exit code should not conflate them.
        return report_failure(audit, "interrupted by user", 130)
    except Exception as exc:
        return report_failure(audit, f"{type(exc).__name__}: {exc}", 1)

    print(f"\n[loopward] status: {result.status}")
    print(f"[loopward] {result.summary}")
    if result.verify is not None:
        for f in result.verify.confirmed:
            print(f"  confirmed: {f}")
        for f in result.verify.rejected:
            print(f"  rejected:  {f}")
    if result.run_dir:
        print(f"[loopward] audit trail: {result.run_dir}")
    # Empty means the review ran but its trail could not be written; the
    # orchestrator has already said so on stderr. The exit code answers "does
    # this diff pass?", not "was the paperwork filed?", so it is unaffected.
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
