# Changelog

All notable changes to loopward are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-09

First beta. A reliability layer for multi-agent LLM systems that runs offline
with zero API keys out of the box.

### Added

- **Anti-loop.** After `N` failed attempts on a subtask, loopward refuses another
  blind retry and forces a **class-jump** to a different review strategy. The loop
  is bounded — it terminates in `EXHAUSTED` rather than spinning — and its ceiling
  is a budget fixed before the loop runs, which no collaborator can push past
  `MAX_TOTAL_ATTEMPTS`. The built-in ladder is `concise → structured`.
- **Stop-gates.** Human approval between critical phases, in three modes:
  `interactive`, `auto` (for CI), and `deny`. The gate is an object-capability:
  `verify()` cannot run without an approval token the gate alone can mint, and a
  token authorises one phase, once.
- **Audit trail.** Every attempt, gate decision, token, and cost is written per
  run to `runs/<timestamp>/audit.{json,md}`. `LOOPWARD_APPROVER` declares who
  decided; otherwise the OS account is recorded, or nothing when the host offers
  no identity, with `approver_source` saying which.
- **Command-line tool.** `loopward <diff>` reviews a diff file; `loopward-demo`
  runs a fully offline, deterministic demonstration. Both default to the offline
  `fake` provider so they work with no API key.
- **`--version` flag** on the `loopward` command.
- **Multiple providers.** One client, swappable providers — `fake` (offline
  default), `deepseek`, and `claude` — with the model always configurable. API
  keys are read from the environment only; the real provider SDKs are optional
  extras.
- **Per-file review on context-window overflow (Anthropic only).** When a diff
  does not fit the model's context window, loopward splits it and reviews one file
  at a time instead of retrying. Such a run ends in `partial_review`, never `ok`,
  because a per-file review cannot see a defect that spans two files. A diff with
  no usable file header, or one exceeding the file cap, is refused with its cause
  on the trail rather than silently truncated.
- **Reproducible benchmark** demonstrating the categorical difference between an
  unbounded naive retry loop and loopward's hard, deterministic ceiling.
