"""loopward — a reliability layer for multi-agent LLM systems.

Three guarantees, enforced structurally (not by convention):
  - anti-loop: stop retrying a failing approach after N attempts; switch
    strategy. The bound is a budget the orchestrator computes before entering
    its own loop, so no injected collaborator can raise it. The class-jump
    verdict is an AttemptOutcome only the tracker mints, and the orchestrator
    checks that before acting on it — defence in depth, not the bound.
  - stop-gate: pause for human approval between critical phases. Verification
    requires an unforgeable Approval token minted only by the gate on APPROVE.
  - audit: a structured, per-run trail of decisions, attempts, cost, and result.

The supported entry point is :class:`Orchestrator`; ``Orchestrator.run()`` runs
the whole review→verify flow under the anti-loop and the stop-gate.
"""

from loopward.engine.orchestrator import Orchestrator, RunResult

__version__ = "0.1.0"

__all__ = ["Orchestrator", "RunResult", "__version__"]
