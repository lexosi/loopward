"""Cost/token representation in the audit trail.

Two failures this fixes, asserted end to end:

A) An unpriced (provider, model) must not report ``cost_usd: 0.0`` — a credible
   number a reader cannot tell from a run that genuinely cost nothing. It reports
   ``null`` (absence), in ``audit.json`` and ``audit.md`` alike.
B) Token and cost figures are derived (``len//4`` estimate x published rate),
   never billed accounting. Every surface that shows them says so, and carries
   the date the rates were last verified — not only the benchmark.
"""

import json

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import PRICING_VERIFIED, LLMClient
from loopward.engine.orchestrator import Orchestrator
from loopward.engine.stop_gate import GATE_AUTO, StopGate

_SAMPLE_DIFF = (
    "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
    "-    return now > self.expires_at\n"
    "+    return now <= self.expires_at  # expiry boundary\n"
)


def _basis(priced: bool) -> dict:
    return {
        "tokens_estimated": True,
        "cost_derived": True,
        "priced": priced,
        "rates_verified": PRICING_VERIFIED,
        "note": "token counts are a len//4 estimate; cost = estimated tokens x rate",
    }


@pytest.mark.unit
def test_uncalculable_cost_renders_null_in_json_and_md(tmp_path):
    # An unpriced run: cost_usd could not be calculated. Both files say `null`,
    # the same token audit already uses for a missing provider stop reason —
    # never 0.0, which reads as "free".
    log = AuditLog(run_id="unpriced", base_dir=tmp_path)
    log.record_usage(prompt=100, completion=40, cost_usd=None, basis=_basis(priced=False))
    run_dir = log.finalize("ok", "done")

    envelope = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    assert envelope["summary"]["cost_usd"] is None  # -> JSON null

    md = (run_dir / "audit.md").read_text(encoding="utf-8")
    assert "null" in md
    assert "0.000000" not in md  # the old misleading zero must be gone


@pytest.mark.unit
def test_real_zero_is_distinct_from_uncalculable_in_trail(tmp_path):
    # A priced run that genuinely cost 0.0 keeps the number; it is not collapsed
    # into the null used for an uncalculable one.
    log = AuditLog(run_id="priced-zero", base_dir=tmp_path)
    log.record_usage(prompt=0, completion=0, cost_usd=0.0, basis=_basis(priced=True))
    run_dir = log.finalize("ok", "done")

    envelope = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    assert envelope["summary"]["cost_usd"] == 0.0
    assert envelope["summary"]["cost_usd"] is not None


@pytest.mark.unit
def test_cost_basis_reaches_json_and_md_with_verification_date(tmp_path):
    # The derived marker + rates_verified date land in BOTH files, so an artifact
    # read without the benchmark in front of it can still tell estimate from
    # accounting and see how old the rates are.
    log = AuditLog(run_id="basis", base_dir=tmp_path)
    log.record_usage(prompt=100, completion=40, cost_usd=None, basis=_basis(priced=False))
    run_dir = log.finalize("ok", "done")

    summary = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))["summary"]
    basis = summary["cost_basis"]
    assert basis["tokens_estimated"] is True
    assert basis["priced"] is False
    assert basis["rates_verified"] == PRICING_VERIFIED

    md = (run_dir / "audit.md").read_text(encoding="utf-8")
    assert PRICING_VERIFIED in md
    assert "estimat" in md.lower()


@pytest.mark.integration
def test_orchestrator_fake_run_marks_cost_null_and_estimated(tmp_path):
    # End to end: a fake (unpriced) run through the orchestrator writes a trail
    # whose cost is null and whose token counts are marked as an estimate. The
    # orchestrator must carry the wrapper's cost basis into the trail.
    llm = LLMClient(provider="fake")
    audit = AuditLog(run_id="e2e", base_dir=tmp_path)
    Orchestrator(llm=llm, gate=StopGate(GATE_AUTO), audit=audit).run(_SAMPLE_DIFF)

    summary = json.loads((audit.run_dir / "audit.json").read_text(encoding="utf-8"))["summary"]
    assert summary["cost_usd"] is None
    assert summary["cost_basis"]["priced"] is False
    assert summary["cost_basis"]["tokens_estimated"] is True
    assert summary["tokens"]["total"] > 0  # tokens still counted, just estimated
