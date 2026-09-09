"""Tests for the loopward CLI (loopward/cli.py)."""

import pytest

from loopward.cli import build_parser, main

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


# ---- Unreadable diff: rejected before a run exists ---------------------------
# Same category as a missing file (exit 2, no run, no trail): the CLI cannot
# obtain a usable diff, so nothing runs. read_text raises two disjoint families
# from one line — UnicodeDecodeError (a ValueError) for non-UTF-8 bytes, and
# OSError for a path that exists but cannot be read as a file (a directory is
# the deterministic case; a permission or TOCTOU failure is the same except).
# `main` returning 2 rather than propagating is itself the "no traceback" proof.

# The exact bytes from the external audit's reproduction: an invalid UTF-8 lead
# byte (0xff) so decoding fails at position 0.
_NON_UTF8_DIFF = b"\xff\xfe\x00garbage"


@pytest.mark.unit
def test_main_non_utf8_diff_returns_2_without_traceback(tmp_path):
    patch = tmp_path / "bin.patch"
    patch.write_bytes(_NON_UTF8_DIFF)
    runs = tmp_path / "runs"
    # No pytest.raises: a clean exit code, not a propagated UnicodeDecodeError.
    assert main([str(patch), "--gate", "auto", "--runs-dir", str(runs)]) == 2


@pytest.mark.unit
def test_main_directory_as_diff_returns_2(tmp_path):
    # A directory passes the exists() guard, then read_text raises OSError
    # (IsADirectoryError on POSIX, PermissionError on Windows).
    a_dir = tmp_path / "adir"
    a_dir.mkdir()
    runs = tmp_path / "runs"
    assert main([str(a_dir), "--gate", "auto", "--runs-dir", str(runs)]) == 2


@pytest.mark.unit
def test_unreadable_diff_writes_no_run_dir(tmp_path):
    # The class, not the case: neither clean rejection may create a run
    # directory. The absence of a trail is a guarantee here, not an accident of
    # ordering — a run that never began must leave nothing half-written.
    bin_patch = tmp_path / "bin.patch"
    bin_patch.write_bytes(_NON_UTF8_DIFF)
    a_dir = tmp_path / "adir"
    a_dir.mkdir()

    for bad in (bin_patch, a_dir):
        runs = tmp_path / f"runs_{bad.name}"
        assert main([str(bad), "--gate", "auto", "--runs-dir", str(runs)]) == 2
        assert not runs.exists()
