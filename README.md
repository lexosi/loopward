# loopward

**A reliability layer for multi-agent LLM systems.** Anti-loop, human stop-gates,
and a per-run audit trail — enforced structurally, not by convention. Runs
**offline with zero API keys** out of the box.

![loopward demo — anti-loop class-jump, stop-gate, audit trail](docs/demo.gif)

*Rendered from actual `loopward-demo` output (deterministic, runs offline — try it yourself below).*

## The problem

Turn agents loose and they loop on a broken approach without ever converging,
slip past the checkpoint before the irreversible step, or burn tokens with no
ceiling — and leave no record of what they decided or what it cost.

## What it does

- **Anti-loop** — after `N` failed attempts on a subtask, refuse another naive
  retry and force a **class-jump** to a different strategy. Bounded loop that
  terminates in `EXHAUSTED`, never spins.
- **Stop-gates** — pause for human approval between critical phases
  (`interactive` / `auto` for CI / `deny`). The gate is an **object-capability**:
  `verify()` can't run without an approval token the gate alone can mint.
- **Audit trail** — every attempt, gate decision, token, and cost written per run
  to `runs/<timestamp>/audit.{json,md}`.

## Quickstart (no API key)

```bash
pip install git+https://github.com/lexosi/loopward.git
loopward-demo
```

Runs in seconds, fully offline (deterministic `fake` provider). You'll watch the
first review strategy fail to parse three times, **class-jump** to a stricter
strategy that works, pass the stop-gate, verify, and write an audit trail:

```text
[class_jump] switching review strategy after 3 failures on 'concise'
[phase] review: ok via strategy 'structured' (2 findings)
[result] 2 confirmed finding(s), blocking
--- audit trail written to: runs/20260805T101006Z ---
```

Review your own diff:

```bash
loopward path/to/change.patch --gate auto          # offline fake provider
loopward path/to/change.patch --provider deepseek  # real LLM (needs key)
```

## Battle-tested

loopward is the reliability core extracted from a **19-agent orchestration system
running in production** ([multiagent-system-lex][mas]). The **deny-by-default**
design is not a preference — it's a measured result. In that system, **advisory
rules produced 0/3 compliance immediately after canonization; deny-by-default
enforcement blocked 9/9 unauthorized writes over two months.** Convention did not
hold; structural enforcement did. loopward carries that lesson as its default.

[mas]: https://github.com/lexosi/multiagent-system-lex

### Enforced structurally, not by convention

The stop-gate is an object-capability. `Verifier.verify(findings, approval)`
requires an `Approval`, and genuineness is checked by **identity membership** in a
private registry that **only `StopGate.request()` populates** (on APPROVE) — not
by `isinstance`. Every forgery path is closed within the threat model:

- calling `verify()` with no approval → `TypeError`;
- a hand-built `Approval(...)`, a subclass, or an `object.__new__(Approval)`
  instance → rejected (not in the registry / subclassing raises).

The anti-loop's hard bound is the orchestrator's bounded loop, and the
`AttemptOutcome` verdict is a tamper-evident token only
`AttemptTracker.record_failure()` mints. [`tests/test_no_evasion.py`][evade]
re-runs every forgery attempt and asserts each is rejected — that's the proof.

[evade]: tests/test_no_evasion.py

## Status

**v0.1, beta.** The public API (`loopward.Orchestrator`) is small and the mechanism
is stable, but the surface may still change before 1.0. Not yet on PyPI — install
from git (above).

Roadmap:

1. Publish to PyPI (`pip install loopward`).
2. Pluggable strategy sets beyond the built-in review→verify flow.
3. Structured audit export (JSONL stream) for external dashboards.

## Benchmark

The number that matters is **categorical, not a ratio**. On a code-review task
that never converges, loopward has a hard, deterministic ceiling: it stops on its
own after exactly **6 LLM calls / 819 tokens** and returns `EXHAUSTED` — asserted
at runtime against `MAX_ATTEMPTS × len(STRATEGIES)` (`3 × 2 = 6`), so a core
change breaks the benchmark loudly instead of reporting a false number. A **naive
retry loop has no ceiling at all** (`naive_self_terminates: false`) — only a
human or a timeout stops it.

The token *ratio* depends on **when a human kills the naive loop** (`K`), so it is
labeled as such — illustrative, not the headline:

| K (human kills naive loop at) | naive tokens | loopward | token ratio |
| --- | --- | --- | --- |
| 10 | 1160 | 819 | 1.42× |
| 50 | 5800 | 819 | 7.08× |
| 100 | 11600 | 819 | 14.16× |

The absolute counts are tiny (819) because this is one small task on the
deterministic `fake` provider — the benchmark demonstrates the *mechanism*
(unbounded → bounded), not a large bill. Reproduce:

```bash
python benchmarks/bench_antiloop.py          # human-readable table
python benchmarks/bench_antiloop.py --json   # machine-readable measurements
```

## Multi-LLM

One client, swappable providers — the model is always configurable, never
hardcoded:

```bash
loopward diff.patch --provider deepseek --model deepseek-v4-pro
loopward diff.patch --provider claude   --model claude-sonnet-4-6
loopward diff.patch --provider fake                     # offline, default
```

Keys are read from the environment only (`DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`
— see `.env.example`). The real SDKs are optional installs
(`pip install "loopward[deepseek]"` or `"loopward[claude]"`).

## Architecture

```text
            ┌──────────────────────────────────────────────┐
            │                Orchestrator                   │
            │   (the only component that calls agents)      │
            └───────────────┬───────────────┬──────────────┘
              review phase   │               │  verify phase
                             ▼               ▼
                      ┌────────────┐   ┌────────────┐
                      │  Reviewer  │   │  Verifier  │
                      └─────┬──────┘   └─────┬──────┘
                            │ parse fails?    │
            ┌───────────────▼──────┐   ┌──────▼───────────┐
            │  AttemptTracker      │   │   StopGate       │
            │  (3 → class-jump)    │   │ (human approval) │
            └───────────────┬──────┘   └──────┬───────────┘
                            └──────┬───────────┘
                                   ▼
                             ┌──────────┐     ┌──────────────┐
                             │ LLMClient│     │   AuditLog   │
                             │ provider │     │ runs/<ts>/   │
                             └──────────┘     └──────────────┘
```

Agents never call each other — the orchestrator coordinates, and every
collaborator is injected, so the whole flow is testable offline.

## Tests

```bash
pytest                             # all green, offline, no secrets
pytest tests/test_no_evasion.py    # the gate cannot be bypassed (structural proof)
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
