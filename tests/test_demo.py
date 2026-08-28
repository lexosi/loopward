"""Tests for the packaged offline demo (loopward/demo.py)."""

import pytest

from loopward.agents.reviewer import ReviewParseError, parse_findings
from loopward.demo import SAMPLE_DIFF, console, main, scripted_reviewer
from loopward.engine.llm_wrapper import TASK_REVIEW, TASK_VERIFY


@pytest.mark.unit
def test_demo_main_returns_ok():
    assert main([]) == 0


@pytest.mark.unit
def test_demo_main_delay_zero_does_not_sleep(monkeypatch):
    # If main honored a delay here it would call time.sleep; assert it does not
    # when delay is 0 (the CI-fast default).
    calls = []
    monkeypatch.setattr("loopward.demo.time.sleep", lambda s: calls.append(s))
    assert main(["--demo-delay", "0"]) == 0
    assert calls == []


@pytest.mark.unit
def test_scripted_reviewer_concise_is_unparseable():
    # 'concise' strategy -> prose without severity tags.
    text = scripted_reviewer([], f"{TASK_REVIEW}:concise")
    with pytest.raises(ReviewParseError):
        parse_findings(text)


@pytest.mark.unit
def test_scripted_reviewer_structured_parses():
    text = scripted_reviewer([], f"{TASK_REVIEW}:structured")
    findings = parse_findings(text)
    assert len(findings) == 2
    assert all(f.severity == "HIGH" for f in findings)


@pytest.mark.unit
def test_scripted_reviewer_verify_confirms_all():
    listing = {"role": "user", "content": "1. HIGH: a\n2. LOW: b"}
    text = scripted_reviewer([listing], TASK_VERIFY)
    # full coverage: one adjudication line per listed finding
    assert text.splitlines() == ["CONFIRM 1", "CONFIRM 2"]
    # a verify reply must contain NO 'REJECT' lines (confirms everything)
    assert "REJECT" not in text.upper()


@pytest.mark.unit
def test_sample_diff_has_the_planted_smells():
    # The demo's whole point is these two smells drive the findings.
    assert "<=" in SAMPLE_DIFF
    assert "len(items) + 1" in SAMPLE_DIFF


@pytest.mark.unit
def test_console_uses_paced_delay(monkeypatch):
    captured = {}

    def fake_main(argv=None, *, default_delay=0.0):
        captured["default_delay"] = default_delay
        return 0

    monkeypatch.setattr("loopward.demo.main", fake_main)
    assert console() == 0
    assert captured["default_delay"] == 0.4
