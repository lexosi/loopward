"""Regression: two runs starting in the same second must not share a directory.

The run directory is named with second precision. Before the fix, a second run
begun within the same second resolved to the same path and silently overwrote
the first run's trail.
"""

from __future__ import annotations

import json

import pytest

from loopward.engine import audit as audit_mod
from loopward.engine.audit import AuditLog

FROZEN_TS = "20260807T140943Z"


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze the directory timestamp: both runs land in the same second."""
    monkeypatch.setattr(audit_mod, "_ts_for_dir", lambda: FROZEN_TS)


@pytest.mark.unit
def test_two_runs_in_the_same_second_keep_separate_trails(tmp_path, frozen_clock):
    first = AuditLog(run_id="first-run", base_dir=tmp_path)
    first.record("phase", "review started")
    first_dir = first.finalize("ok", "first result")

    second = AuditLog(run_id="second-run", base_dir=tmp_path)
    second.record("phase", "review started")
    second_dir = second.finalize("ok", "second result")

    # 1. two distinct directories exist
    assert first_dir != second_dir
    assert first_dir.is_dir() and second_dir.is_dir()
    run_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(run_dirs) == 2, f"expected 2 run dirs, got {run_dirs}"

    # the uncollided name keeps the documented YYYYMMDDTHHMMSSZ format
    assert first_dir.name == FROZEN_TS
    assert second_dir.name.startswith(FROZEN_TS)

    # 2. both trails are intact and parseable
    for d in (first_dir, second_dir):
        envelope = json.loads((d / "audit.json").read_text(encoding="utf-8"))
        assert envelope["events"], f"{d.name} lost its events"
        assert (d / "audit.md").exists()

    # 3. the first run's identity survives the second run
    first_envelope = json.loads((first_dir / "audit.json").read_text(encoding="utf-8"))
    assert first_envelope["summary"]["run_id"] == "first-run"
    assert first_envelope["summary"]["result"] == "first result"
