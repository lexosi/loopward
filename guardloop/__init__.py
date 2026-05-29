"""guardloop — a reliability layer for multi-agent LLM systems.

Three guarantees:
  - anti-loop: stop retrying a failing approach after N attempts; switch strategy.
  - stop-gate: pause for human approval between critical phases.
  - audit: a structured, per-run trail of decisions, attempts, cost, and result.
"""

__version__ = "0.1.0"
