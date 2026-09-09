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
- `Approval` is single-use and phase-bound, both as composition-error guards,
  not defences against a hostile caller. A genuine token is spent when its
  phase concludes — recorded in a module-level `_CONSUMED` registry, sibling of
  `_MINTED` — so a replayed token is refused; and `verify()` refuses a token
  minted for another phase. Neither closes the in-process mint above: a caller
  can still mint a fresh `verify` approval and spend it once.
- The real bound on retries is the orchestrator's bounded loop, not the
  `AttemptOutcome` token. The orchestrator does check that a verdict was minted
  by the tracker before acting on it, which catches a hand-rolled tracker
  returning a duck-typed stand-in — the composition error above, not a bound.
- `AttemptOutcome` is mutable in place. `object.__setattr__(outcome, "action",
  "class_jump")` flips `must_class_jump` on a genuine, registered outcome and
  `is_genuine_outcome()` still returns True. `Approval` has the same hole, and
  `phase` now feeds one decision — the phase-binding check — so
  `object.__setattr__(approval, "phase", "verify")` defeats that binding on a
  genuine token by the same direct-mutation route. Single-use survives it: the
  spent bit lives in `_CONSUMED`, off the object. Here `action` is the retry
  decision, and the loop's budget is what makes that case survivable.
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
- A run that overflows the context window has a **second, additive** spend
  ceiling, not folded under the one above:
  `MAX_ATTEMPTS × (len(STRATEGIES) − 1) + 1 + MAX_CHUNKS + MAX_ATTEMPTS`
  (`3 × 1 + 1 + 10 + 3 = 17` provider calls), measured and asserted by the
  benchmark against the formula derived from the constants. The whole-diff term is
  not 1: the strategies differ in prompt length, so a diff can fit an earlier
  strategy and overflow a later one, and every earlier strategy can parse-fail its
  whole budget before the overflow. The `MAX_CHUNKS` term is a separate budget, and
  `MAX_TOTAL_ATTEMPTS` does not cap it.

**Coverage, not just spend — the one gap `partial_review` names (Anthropic only).**
The chunked review exists **only for the `claude` provider**: the failure map that
recognises a context-window rejection is gated to Anthropic, so another provider's
overflow classifies `UNKNOWN` and crashes rather than chunking. When a `claude` run
*does* review a file at a time, a defect that **spans two files** is invisible to
every per-file review. The run says so: it ends in `partial_review`, not `ok`, and
the trail records the blind spot. This is a correctness limitation, stated rather
than smoothed over — loopward does not claim a chunked review is a complete one.

**A design decision with a cost: retries are blind.** Each attempt reconstructs
the prompt from scratch; nothing carries between attempts but token and cost
totals, so there is no accumulated context to manage. An *informed* retry (feeding
the previous answer back) could converge sooner, but it risks suppressing the part
of the answer that was already correct. loopward chooses the clean retry —
bounded, reproducible, and unable to argue itself into a worse answer. It is a
trade, not an oversight.
