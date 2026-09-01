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
- The real bound on retries is the orchestrator's bounded loop, not the
  `AttemptOutcome` token. The orchestrator does check that a verdict was minted
  by the tracker before acting on it, which catches a hand-rolled tracker
  returning a duck-typed stand-in — the composition error above, not a bound.
- `AttemptOutcome` is mutable in place. `object.__setattr__(outcome, "action",
  "class_jump")` flips `must_class_jump` on a genuine, registered outcome and
  `is_genuine_outcome()` still returns True. `Approval` has the same hole, but
  there the mutable field is `phase`, which decides nothing; here `action` is
  the decision. The loop's budget is what makes it survivable.
- `AttemptOutcome` can also be *constructed* outside the tracker, by two routes
  that defeat the mint check in different ways. `dataclasses.replace(outcome,
  action="class_jump")` satisfies it — `replace` re-runs `__init__` and carries
  `_mint` over from the original, so `__post_init__` sees the real sentinel.
  `copy.copy` and `copy.deepcopy` bypass it — both rebuild through
  `__reduce_ex__`, which never calls `__init__` at all. None of the three lands
  in `_MINTED`, so `is_genuine_outcome()` rejects all of them and the
  orchestrator acts on none: the gap is the token's forgeability, not an escape
  from the bound. `Approval` resists all three, and not by luck — it is not a
  dataclass, so `replace` raises `TypeError`, and `__slots__` makes a copy
  restore state attribute by attribute, straight into a `__setattr__` that
  refuses. The hole is in `@dataclass(frozen=True)`, not in the mint pattern.
- The loop's budget is itself raisable, within a ceiling. `tracker` and
  `strategies` are public constructor parameters and the budget is
  `max_attempts * len(strategies)`, so the default `3 × 2 = 6` review attempts
  become up to `MAX_TOTAL_ATTEMPTS` (100) — measured at exactly 100 from either
  parameter on its own. The run still always terminates in `EXHAUSTED`; what
  moves is the spend, by up to ~17x. `MAX_TOTAL_ATTEMPTS` is a spend limit, not
  a design limit, and it is the one number here no collaborator can raise.
