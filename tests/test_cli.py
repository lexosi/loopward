"""Tests for the guardloop CLI (guardloop/cli.py)."""

import pytest

from guardloop.cli import build_parser, main

_DIFF_WITH_SMELL = """\
--- a/auth.py
+++ b/auth.py
@@ def is_valid(self, now):
-        return now < self.expires_at
+        return now <= self.expires_at   # accept tokens exactly at expiry
"""


@pytest.mark.unit
def test_parser_defaults():
    args = build_parser().parse_args(["change.patch"])
    assert args.provider == "fake"
    assert args.gate == "interactive"
    assert str(args.diff) == "change.patch"


@pytest.mark.unit
def test_main_missing_file_returns_2(tmp_path):
    missing = tmp_path / "nope.patch"
    assert main([str(missing), "--gate", "auto"]) == 2


@pytest.mark.unit
def test_main_reviews_ok(tmp_path):
    patch = tmp_path / "change.patch"
    patch.write_text(_DIFF_WITH_SMELL, encoding="utf-8")
    runs = tmp_path / "runs"
    rc = main([str(patch), "--gate", "auto", "--runs-dir", str(runs)])
    assert rc == 0
    # an audit trail was written under the tmp runs dir
    assert runs.exists() and any(runs.iterdir())


@pytest.mark.unit
def test_main_gate_deny_returns_1(tmp_path):
    patch = tmp_path / "change.patch"
    patch.write_text(_DIFF_WITH_SMELL, encoding="utf-8")
    runs = tmp_path / "runs"
    rc = main([str(patch), "--gate", "deny", "--runs-dir", str(runs)])
    assert rc == 1
