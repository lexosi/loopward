# loopward

[![ci](https://github.com/lexosi/loopward/actions/workflows/ci.yml/badge.svg)](https://github.com/lexosi/loopward/actions/workflows/ci.yml)

**A reliability layer for multi-agent LLM systems.** Anti-loop, human stop-gates,
and a per-run audit trail — enforced structurally, not by convention. Runs
**offline with zero API keys** out of the box.

![loopward demo — anti-loop class-jump, stop-gate, audit trail](https://raw.githubusercontent.com/lexosi/loopward/main/docs/demo.gif)

*Illustrative, and currently one event behind the tool: the GIF shows 9 events,
`loopward-demo` now prints 10 — the extra line is the stop-gate's own event,
`[gate] phase 'verify' [auto] -> approve (auto-approved)`, added after the GIF
was recorded. See
[docs/RECORDING.md](https://github.com/lexosi/loopward/blob/main/docs/RECORDING.md)
for the full account.*

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
  to `runs/<timestamp>/audit.{json,md}`. Set `LOOPWARD_APPROVER` to declare who
  decided (e.g. `ci:github-actions`); otherwise the OS account is recorded, or
  nothing when the host offers no identity — `approver_source` says which of the
  three it was (`env` / `os_user` / `unknown`).

## Quickstart (no API key)

```bash
pip install git+https://github.com/lexosi/loopward.git
loopward-demo
```

Runs in seconds, fully offline (deterministic `fake` provider). You'll watch the
first review strategy fail to parse three times, **class-jump** to a stricter
strategy that works, pass the stop-gate, verify, and write an audit trail.
Abridged excerpt below — the full run prints 10 events; `...` marks omitted
lines:

```text
...
[class_jump] switching review strategy after 3 failures on 'concise'
[phase] review: ok via strategy 'structured' (2 findings)
[gate] phase 'verify' [auto] -> approve (auto-approved)
...
[result] 2 confirmed finding(s), blocking
--- audit trail written to: runs/20260805T101006Z ---
```

**What the demo is, on purpose.** It is a fixture, in the same sense as the
benchmark below. The code paths are the production ones; three of the *inputs*
are authored rather than observed, and it is worth knowing which:

- **The failure is scripted.** `scripted_reviewer` in
  [`loopward/demo.py`](https://github.com/lexosi/loopward/blob/main/loopward/demo.py) returns unparseable prose for the first
  strategy whatever the prompt says — it never reads the prompt. No model
  refused. That makes the class-jump fire deterministically and offline; it is
  not a recording of a model failing, and loopward does not ship one.
- **The findings are planted.** The sample diff carries two deliberate bugs and
  the offline `fake` provider matches on exactly those substrings, so it finds
  what it was written to find.
- **`rejected 0` is a foregone conclusion.** Both offline fakes confirm every
  finding by construction. The verifier's false-positive rejection — the other
  half of what it is for — is exercised in the test suite and in no runnable
  artifact here.

Review your own diff:

```bash
loopward path/to/change.patch --gate auto          # offline fake provider
loopward path/to/change.patch --provider deepseek  # real LLM (needs key)
```

## Where this comes from

loopward is the reliability layer extracted from [a personal 19-agent
harness][mas] (13 Claude reasoners + 6 DeepSeek workers) I run daily.

After canonizing a set of advisory rules, compliance was 0/3. After moving
to deny-by-default enforcement (out-of-process PreToolUse hooks), the
observed result over the window 2026-06-02 -> 2026-08-04 was:
**9 blocked tool-calls across 7 runs, 4 via explicit logged override, 0 unauthorized writes.**

**Scope note.** n=3 runs for the advisory baseline, n=7 for the enforcement
window, single operator. The per-run enforcement logs are not part of the
public snapshot, so treat this as a dated, documented observation — not a
benchmark you can re-run. What you *can* re-run is
`benchmarks/bench_antiloop.py`, which reproduces the table in the Benchmark
section below, exactly.

[mas]: https://github.com/lexosi/multiagent-system-lex

### Enforced structurally, not by convention

The stop-gate is an object-capability. `Verifier.verify(findings, diff, approval)`
requires an `Approval`, and genuineness is checked by **identity membership** in a
private registry that **only `StopGate.request()` populates** (on APPROVE) — not
by `isinstance`. Every forgery path is closed within the threat model:

- calling `verify()` with no approval → `TypeError`;
- a hand-built `Approval(...)`, a subclass, or an `object.__new__(Approval)`
  instance → rejected (not in the registry / subclassing raises);
- a genuine token replayed after its phase concluded, or one minted for a
  different phase → `TypeError`: an approval authorises one phase, once.

The anti-loop's hard bound is the orchestrator's bounded loop — a budget computed
before the loop runs, which no injected collaborator can make unbounded or push
past `MAX_TOTAL_ATTEMPTS` (100). Within that ceiling a collaborator *can* raise
it: the default budget is `3 × 2 = 6` attempts. The `AttemptOutcome` verdict is a
separate, weaker thing: only
`AttemptTracker.record_failure()` can mint one, and the orchestrator checks that
before acting on it, but a minted verdict can still be mutated in place through
`object.__setattr__` and nothing detects it. Defence in depth, not the bound —
see [THREAT_MODEL.md](https://github.com/lexosi/loopward/blob/main/THREAT_MODEL.md). [`tests/test_no_evasion.py`][evade]
re-runs every forgery attempt and asserts each is rejected — that's the proof.

[evade]: https://github.com/lexosi/loopward/blob/main/tests/test_no_evasion.py

See [THREAT_MODEL.md](https://github.com/lexosi/loopward/blob/main/THREAT_MODEL.md) for what this does and does not defend against.

### Retries are blind, and the ladder is why

Every attempt is built clean. The orchestrator hands the reviewer the same diff
and the reviewer reconstructs the prompt from scratch each time
([`reviewer.py`](https://github.com/lexosi/loopward/blob/main/loopward/agents/reviewer.py) rebuilds the two messages per call);
nothing carries over between attempts but the running token and cost totals. There
is no accumulated conversation to prune, and loopward does not claim to manage one.

That is the whole argument for the strategy ladder. If three clean attempts at the
same prompt fail, a fourth clean attempt is not unlucky — the prompt is the
problem, and reshaping it is the only move left. So after `MAX_ATTEMPTS` the loop
stops retrying and **class-jumps** to the next strategy. The built-in ladder is
`concise → structured`: two shapes of one prompt for the same output contract, not
two different kinds of attack. (A genuinely different *mechanism* — the per-file
review below — is reached by a different path, a measured failure class, not by
this counter.)

This is a **decision with a cost**, not a free win. A blind retry cannot converge
on what the last attempt nearly got right, because it never sees the last attempt.
An *informed* retry — feeding the previous answer back — might converge sooner, but
it risks suppressing the part of the response that was already correct. loopward
chooses the clean retry: bounded, reproducible, and unable to talk itself into a
worse answer.

**The context-window overflow, and `partial_review` (Anthropic only).** One
failure no prompt can prevent is the diff not fitting the model's context window.
This path exists **for the `claude` provider only**: the failure map that
recognises a context-window rejection is gated to Anthropic
([`failure_classifier.py`](https://github.com/lexosi/loopward/blob/main/loopward/engine/failure_classifier.py)). When a
`claude` run's whole-diff review is refused for length, loopward does not retry —
it splits the diff and reviews it one file at a time. Any other provider's
overflow classifies as `UNKNOWN` and **crashes** the run; there is no chunked
fallback for it. A chunked run ends in its own status, **`partial_review`**, never
`ok`: a per-file review structurally cannot see a defect that spans two files, and
returning `ok` would certify coverage that never happened. The audit trail records
the mechanism (how many pieces, which were unreadable, and the cross-file blind
spot); the status carries only the consequence.

**Two limits crash by design, and say why.** The split refuses two inputs rather
than paper over them, and both reach the crash trail with their cause:

- a diff with **no usable file header** — there is no boundary to split on, so
  `split_diff` raises instead of handing back "zero files to review";
- **more than `MAX_CHUNKS` (10) files** — refused whole, with the file count it
  saw and the cap it exceeded on the trail. That is what separates a declared limit
  from a silent truncation.

Neither becomes `partial_review`: a run that reviewed *nothing* must not report a
partial review of something.

## Status

**v0.1, beta.** The public API (`loopward.Orchestrator`) is small and the mechanism
is stable, but the surface may still change before 1.0. Not yet on PyPI — install
from git (above).

Roadmap:

1. Publish to PyPI (`pip install loopward`).
2. Pluggable strategy sets beyond the built-in review→verify flow.
3. Structured audit export (JSONL stream) for external dashboards.

Exercised daily in one personal system; no external users yet.

## Benchmark

The number that matters is **categorical, not a ratio**. On a code-review task
that never converges, loopward has a hard, deterministic ceiling: it stops on its
own after exactly **6 LLM calls / 1056 tokens** and returns `EXHAUSTED` — asserted
at runtime against `MAX_ATTEMPTS × len(STRATEGIES)` (`3 × 2 = 6`), so a core
change breaks the benchmark loudly instead of reporting a false number. A **naive
retry loop has no ceiling at all** (`naive_self_terminates: false`) — only a
human or a timeout stops it.

There is a **second** ceiling, separate and additive. A diff that overflows the
context window (Anthropic only) is not retried — it is reviewed one file at a time
(see [Retries are blind, and the ladder is why](#retries-are-blind-and-the-ladder-is-why)).
That chunked path has its own measured bound:

- **17 provider calls**, `MAX_ATTEMPTS × (len(STRATEGIES) − 1) + 1 + MAX_CHUNKS +
  MAX_ATTEMPTS` = `3 × 1 + 1 + 10 + 3`. The whole-diff term is **not** 1: the
  strategies have different prompt lengths (`structured` is longer than `concise`),
  so a diff can fit one strategy and overflow the next. In the worst case every
  earlier strategy parse-fails its full budget and class-jumps, and only the last
  strategy overflows and triggers the split — then one review per file-piece (up to
  `MAX_CHUNKS`) and the verify loop's own budget. `MAX_CHUNKS` is **not** folded
  under the 6-call bound above or under `MAX_TOTAL_ATTEMPTS`; it is a separate term.
  The benchmark measures this worst case by running it and asserts the count
  against the formula **derived from the constants** — add a strategy and the
  number moves on its own.
- A chunked run that *reaches a verdict* (verify confirms) costs **15** provider
  calls and ends in `partial_review`, not `ok`: a per-file review cannot see a
  defect that spans two files, so the status refuses to certify coverage that did
  not happen.

The token *ratio* depends on **when a human kills the naive loop** (`K`), so it is
labeled as such — illustrative, not the headline:

| K (human kills naive loop at) | naive tokens | loopward | token ratio |
| --- | --- | --- | --- |
| 10 | 1770 | 1056 | 1.68× |
| 25 | 4425 | 1056 | 4.19× |
| 50 | 8850 | 1056 | 8.38× |
| 100 | 17700 | 1056 | 16.76× |

The absolute counts are tiny (1056) because this is one small task on the
deterministic `fake` provider — the benchmark demonstrates the *mechanism*
(unbounded → bounded), not a large bill.

Token counts are estimated (len//4) and therefore move with prompt length;
they rose when the diff was wrapped in explicit delimiters to separate
untrusted input from instructions, and again when the default `concise`
strategy was rewritten to actually ask for the severity-tag format the parser
requires — 927 → 1056 tokens, and every ratio above with them. The categorical
bound — 6 calls, MAX_ATTEMPTS x len(STRATEGIES) — is unchanged and is the number
that matters.

Reproduce. The benchmark is not part of the installed package, so this one
starts from a clone rather than the `pip install` above:

```bash
git clone https://github.com/lexosi/loopward.git
cd loopward
pip install -e .
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
— see `.env.example`). The real SDKs are optional extras. loopward is not on
PyPI yet, so ask for them through the git URL, not by bare name:

```bash
pip install "loopward[deepseek] @ git+https://github.com/lexosi/loopward.git"
pip install "loopward[claude]   @ git+https://github.com/lexosi/loopward.git"
```

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

Also from a clone — `pip install` ships the package, not the test suite or the
tools that run it:

```bash
git clone https://github.com/lexosi/loopward.git
cd loopward
pip install -e ".[dev]"            # brings pytest and ruff
pytest                             # all green, offline, no secrets
pytest tests/test_no_evasion.py    # the gate cannot be bypassed (structural proof)
ruff check .
```

## License

MIT — see [LICENSE](https://github.com/lexosi/loopward/blob/main/LICENSE).
