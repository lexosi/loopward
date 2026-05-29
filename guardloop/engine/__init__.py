"""guardloop engine: the reliability primitives.

Public surface:
  - AttemptTracker / AttemptOutcome   (anti_loop)
  - StopGate / GateMode / Decision    (stop_gate)
  - LLMClient / LLMResponse           (llm_wrapper)
  - AuditLog                          (audit)

The Orchestrator is the coordinator layer that *uses* these primitives and the
agents; import it directly from ``guardloop.engine.orchestrator`` to keep the
package import graph acyclic (orchestrator -> agents -> engine primitives).
"""

from guardloop.engine.anti_loop import AttemptOutcome, AttemptTracker
from guardloop.engine.audit import AuditLog
from guardloop.engine.llm_wrapper import LLMClient, LLMResponse
from guardloop.engine.stop_gate import Decision, GateMode, StopGate

__all__ = [
    "AttemptOutcome",
    "AttemptTracker",
    "AuditLog",
    "LLMClient",
    "LLMResponse",
    "Decision",
    "GateMode",
    "StopGate",
]
