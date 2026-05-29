# guardloop

**A reliability layer for multi-agent LLM systems.** Most agent frameworks give
you *capability* — more tools, more autonomy. `guardloop` gives you the thing
that makes autonomy safe to ship: **reliability**. Anti-loop. Human stop-gates.
A full audit trail of every decision and every dollar.

> Runs **offline with zero API keys** out of the box (a deterministic `fake`
> provider powers the demo and the whole test suite).

---

## The problem

Turn an LLM loose on a task and three failure modes show up fast:

1. **It loops.** Same broken approach, retried forever, burning tokens.
2. **It acts unsupervised.** No checkpoint before the irreversible step.
3. **It's a black box.** No record of what it decided, why, or what it cost.

`guardloop` is the small, boring layer that fixes exactly those three.

## The three guarantees

| Guarantee | What it does | Module |
|---|---|---|
| **Anti-loop** | After `N` failed attempts on a subtask, refuse another naive retry and force a **class-jump** to a different strategy. | `engine/anti_loop.py` |
| **Stop-gate** | Pause for human approval between critical phases (`interactive` / `auto` for CI / `deny`). Every decision is logged. | `engine/stop_gate.py` |
| **Audit** | Per-run trail: every attempt, gate, token, and cost → `runs/<timestamp>/audit.{json,md}`. | `engine/audit.py` |

The demo task is **automated code review**: a Reviewer agent finds issues, a
Verifier agent re-checks them (probe before you trust), and the orchestrator runs
both under anti-loop + stop-gate + audit.

## Quickstart (no API key)

```bash
pip install -e .[dev]
python demo/run_demo.py
```

You'll watch the reviewer's first strategy fail to produce parseable findings
three times, **class-jump** to a stricter strategy that works, pass the
stop-gate, verify, and write an audit trail:

```
[class_jump] switching review strategy after 3 failures on 'concise'
[gate] phase 'verify' [auto] -> approve (auto-approved)
[result] 2 confirmed finding(s), blocking
--- audit trail written to: runs/20260529T163401Z ---
```

Review your own diff:

```bash
guardloop path/to/change.patch --gate auto          # offline fake provider
guardloop path/to/change.patch --provider deepseek  # real LLM (needs key)
```

## Multi-LLM

One client, swappable providers — the model is **always configurable, never
hardcoded**:

```bash
guardloop diff.patch --provider deepseek --model deepseek-v4-pro
guardloop diff.patch --provider claude   --model claude-sonnet-4-6
guardloop diff.patch --provider fake                     # offline, default
```

Keys are read from the environment only (`DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`
— see `.env.example`). The real SDKs are optional installs (`pip install -e .[deepseek]`
or `.[claude]`).

## Architecture

```
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

Agents never call each other — the orchestrator coordinates. Every collaborator
is injected, so the whole flow is testable offline.

## Demo

<!-- TODO: add a terminal GIF of `python demo/run_demo.py` here.
     Record with e.g. asciinema/termtosvg and drop the asset in docs/. -->

_GIF placeholder — record `python demo/run_demo.py` and embed it here._

## Tests

```bash
pytest                 # all green, offline, no secrets
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
