"""Regression tests for B4: no exception may end a run without a trail.

The audit trail is the product. Before this, any exception during a run meant
zero record of what happened — for a package whose third headline is "audit
trail", the default failure mode was to audit nothing.

Two failure shapes, and they are not symmetric:

- **f1 — only the write failed.** The review reached a verdict; the storage did
  not. Nothing is raised: a lost record does not retroactively turn a good run
  into a bad one. ``run_dir`` then reports what is *on disk*, which is not
  always the same as the path that was wanted.
- **f2 — the work failed.** The trail is finalized as ``crashed`` and the
  original exception is re-raised. Findings already produced stay in the trail,
  marked unverified, and deliberately do not come back through the return value.
"""

import json
import pathlib

import pytest

from loopward.engine.audit import AuditLog
from loopward.engine.llm_wrapper import TASK_VERIFY, LLMClient
from loopward.engine.orchestrator import STATUS_CRASHED, STATUS_OK, Orchestrator
from loopward.engine.stop_gate import StopGate

DIFF = "return now <= self.expires_at"
FINDING = "HIGH: token expiry accepts tokens exactly at expiry."


def _reply(messages, task):
    """Minimal fake: one finding on review, a matching adjudication on verify."""
    if task == TASK_VERIFY:
        return "CONFIRM 1"
    return FINDING


def _orch(audit, gate=None):
    return Orchestrator(
        LLMClient(provider="fake", fake_script=_reply),
        gate or StopGate(mode="auto"),
        audit=audit,
    )


def _trail(run_dir):
    return json.loads((pathlib.Path(run_dir) / "audit.json").read_text(encoding="utf-8"))


def _raise_os_error(*args, **kwargs):
    raise OSError("disk full")


# --- f1: the run succeeded, only the write failed --------------------------


@pytest.mark.integration
def test_f1_unusable_output_path_does_not_raise(tmp_path, capsys):
    """B4-5: --runs-dir under an existing FILE. Nothing is created at all."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    audit = AuditLog(run_id="test", base_dir=blocker / "nested")

    result = _orch(audit).run(DIFF)  # must not raise

    assert result.status == STATUS_OK
    assert result.verify is not None
    # Nothing on disk, so no path is offered. Pointing the user at a directory
    # that was never created is the same untruth this package exists to stop.
    assert result.run_dir == ""
    err = capsys.readouterr().err
    assert "could not be written" in err
    assert "FileNotFoundError" in err or "NotADirectoryError" in err


@pytest.mark.integration
def test_f1_reserved_but_empty_dir_reports_the_path(tmp_path, monkeypatch, capsys):
    """A5: directory reserved, then the first write fails. The dir DOES exist."""
    monkeypatch.setattr(pathlib.Path, "write_text", _raise_os_error)
    audit = AuditLog(run_id="test", base_dir=tmp_path)

    result = _orch(audit).run(DIFF)  # must not raise

    assert result.status == STATUS_OK
    assert result.run_dir == str(audit.run_dir)
    assert pathlib.Path(result.run_dir).is_dir()
    assert not any(pathlib.Path(result.run_dir).iterdir())  # reserved and empty
    assert "could not be written" in capsys.readouterr().err


@pytest.mark.integration
def test_f1_half_written_trail_reports_the_path(tmp_path, monkeypatch, capsys):
    """A6: audit.json lands, audit.md fails. Half a product is still a product."""
    real_write = pathlib.Path.write_text

    def only_json(self, *args, **kwargs):
        if self.name == "audit.md":
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", only_json)
    audit = AuditLog(run_id="test", base_dir=tmp_path)

    result = _orch(audit).run(DIFF)  # must not raise

    assert result.status == STATUS_OK
    assert result.run_dir == str(audit.run_dir)
    assert (pathlib.Path(result.run_dir) / "audit.json").exists()
    assert not (pathlib.Path(result.run_dir) / "audit.md").exists()
    assert _trail(result.run_dir)["summary"]["status"] == STATUS_OK
    assert "could not be written" in capsys.readouterr().err


# --- f2: the work failed ---------------------------------------------------


@pytest.mark.integration
def test_f2_eof_at_the_gate_still_writes_the_trail(tmp_path):
    """B4-6: interactive gate, stdin closed. The findings were already paid for."""

    def closed_stdin(phase, summary):
        raise EOFError("EOF when reading a line")

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    gate = StopGate(mode="interactive", prompter=closed_stdin)

    with pytest.raises(EOFError):
        _orch(audit, gate).run(DIFF)

    envelope = _trail(audit.run_dir_on_disk)
    assert envelope["summary"]["status"] == STATUS_CRASHED
    assert "EOFError" in envelope["summary"]["result"]

    # The review's output is in the trail, and it is marked for what it is: the
    # generator's claims, which nothing has adjudicated.
    review = [e for e in envelope["events"] if e["data"].get("stage") == "review"]
    assert len(review) == 1
    assert review[0]["data"]["verified"] is False
    assert review[0]["data"]["findings"] == [FINDING]

    # The full traceback is in the trail, so no entry point has to print one.
    results = [e for e in envelope["events"] if e["kind"] == "result"]
    assert "EOFError" in results[-1]["data"]["traceback"]


@pytest.mark.integration
def test_f2_provider_failure_before_any_finding_still_writes_the_trail(tmp_path):
    """B4-1: missing API key. It dies on the first call, before any finding."""

    def no_key(messages, task):
        raise RuntimeError("DEEPSEEK_API_KEY not set (provider=deepseek)")

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    orch = Orchestrator(
        LLMClient(provider="fake", fake_script=no_key),
        StopGate(mode="auto"),
        audit=audit,
    )

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        orch.run(DIFF)

    envelope = _trail(audit.run_dir_on_disk)
    assert envelope["summary"]["status"] == STATUS_CRASHED
    assert "DEEPSEEK_API_KEY" in envelope["summary"]["result"]
    assert envelope["events"], "a crashed run still records what it got through"


@pytest.mark.integration
def test_f2_keyboard_interrupt_is_recorded_and_re_raised(tmp_path):
    """Ctrl-C is a BaseException. It still gets a trail, and still interrupts."""

    def interrupted(phase, summary):
        raise KeyboardInterrupt

    audit = AuditLog(run_id="test", base_dir=tmp_path)
    gate = StopGate(mode="interactive", prompter=interrupted)

    with pytest.raises(KeyboardInterrupt):
        _orch(audit, gate).run(DIFF)

    assert _trail(audit.run_dir_on_disk)["summary"]["status"] == STATUS_CRASHED


@pytest.mark.integration
def test_f2_handler_that_cannot_write_propagates_the_original(tmp_path, capsys):
    """Contract (d): a guard that blows up while saving is the same bug again.

    The caller must get the thing that actually went wrong, never the storage
    failure that happened while trying to record it.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")

    def closed_stdin(phase, summary):
        raise EOFError("EOF when reading a line")

    audit = AuditLog(run_id="test", base_dir=blocker / "nested")
    gate = StopGate(mode="interactive", prompter=closed_stdin)

    with pytest.raises(EOFError):  # the ORIGINAL, not the OSError from the write
        _orch(audit, gate).run(DIFF)

    assert "could not be written" in capsys.readouterr().err


# --- the ownership rule ----------------------------------------------------


@pytest.mark.unit
def test_finalize_twice_is_refused(tmp_path):
    """Only Orchestrator.run finalizes, exactly once. The flag guards the rule."""
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    audit.finalize(STATUS_OK, "first")
    with pytest.raises(RuntimeError, match="already finalized"):
        audit.finalize(STATUS_CRASHED, "second")


@pytest.mark.unit
def test_run_dir_on_disk_is_empty_until_something_exists(tmp_path):
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    assert audit.run_dir_on_disk == ""
    audit.finalize(STATUS_OK, "done")
    assert audit.run_dir_on_disk == str(audit.run_dir)


@pytest.mark.integration
def test_a_finished_trail_is_never_written_over(tmp_path, capsys):
    """The trail is written once. A second run must not overwrite the first.

    Reusing an ``AuditLog`` is B2-C and is not fixed here — but the guard turns
    it from "silently overwrites the trail" into "refuses and says so", and the
    refusal arrives through the f1 path: the second review really did run, so it
    is reported rather than raised.
    """
    audit = AuditLog(run_id="test", base_dir=tmp_path)
    result = _orch(audit).run(DIFF)
    assert _trail(result.run_dir)["summary"]["status"] == STATUS_OK

    second = _orch(audit).run(DIFF)

    assert second.status == STATUS_OK
    assert "already finalized" in capsys.readouterr().err
    # Still the first trail, byte for byte — not a second run written over it.
    envelope = _trail(result.run_dir)
    assert envelope["summary"]["status"] == STATUS_OK
    assert envelope["summary"]["started_at"] == audit.started_at
