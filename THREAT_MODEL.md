# Threat model

**What the stop-gate defends against: composition error, not a hostile
in-process caller.**

The capability prevents reaching `verify()` by accident — through a refactor,
a naive `isinstance` check, or a new call path added six months later. That is
how an advisory rule actually gets bypassed in practice: nobody attacks it,
somebody forgets it.

**What it does NOT defend against.** Code running inside the same Python
process can mint its own `Approval` (`StopGate("auto").request(...).approval`)
and pass it to `verify()`. There is no defence against that in Python and this
project does not claim one.

Real enforcement against generated code has to live **outside the process**.
In the system loopward was extracted from, that is what the PreToolUse hooks
do — they run in the tool boundary, not in the agent's runtime. loopward is
the in-process half of that lesson.

**Known gaps, stated rather than hidden:**
- `Approval` is not single-use and is not bound to a phase; one approval
  authorises repeated verifications.
- `is_genuine_outcome()` has no production caller. The real bound on retries
  is the orchestrator's bounded loop, not the token.
- The verifier fails open on unparseable model output (deliberate: it does not
  silently delete real findings), and the anti-loop covers `review`, not
  `verify`.
