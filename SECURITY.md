# Security policy

loopward is a one-person project in beta (v0.1). This file says what is in
scope, what is deliberately out of scope, and how to report something. It does
not promise a response time or describe a process that does not exist yet.

## Supported version

Only the latest commit on `main` (currently the v0.1.x line). There are no
backports to earlier tags.

## In scope

loopward's threat model is **composition error, not a hostile in-process
caller** (see
[THREAT_MODEL.md](https://github.com/lexosi/loopward/blob/main/THREAT_MODEL.md)).
A report is in scope when it shows one of:

- A way to reach `verify()` **by accident** — through a refactor, a naive
  `isinstance` check, or a new call path — without a genuine approval token.
  Defeating the stop-gate capability through ordinary composition, not
  deliberate forgery.
- The anti-loop failing to terminate, or its spend exceeding
  `MAX_TOTAL_ATTEMPTS`.
- The audit trail losing, misattributing, or silently dropping an event.
- A secret reaching the audit trail, logs, or output. Keys are read from the
  environment only; any path that writes one to disk is a bug.

## Out of scope

These are documented non-defenses, not vulnerabilities:

- **In-process forgery.** Code in the same Python process can mint its own
  `Approval` and pass it to `verify()`, or mutate a genuine token in place via
  `object.__setattr__`. Python offers no defense against this and loopward
  claims none — real enforcement lives outside the process (the PreToolUse
  hooks in the host system loopward was extracted from). THREAT_MODEL.md
  enumerates each such gap individually.
- A chunked review missing a defect that spans two files (the Anthropic-only
  overflow path). That is a stated correctness limit — such a run reports
  `partial_review`, not `ok`, on purpose.
- Anything that requires an attacker to already run arbitrary code inside your
  process.

## How to report

Open a private report through GitHub:
[Report a vulnerability](https://github.com/lexosi/loopward/security/advisories/new).
If private advisories are unavailable to you, open a normal issue at
[the issue tracker](https://github.com/lexosi/loopward/issues) saying only that
it is security-related — without the details — so it can be moved private
first.

There is no guaranteed response time. This is a personal project in beta;
expect best-effort handling, not an SLA.
