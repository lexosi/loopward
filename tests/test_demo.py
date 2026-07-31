"""Tests for the packaged offline demo (guardloop/demo.py)."""

import pytest

from guardloop.agents.reviewer import ReviewParseError, parse_findings
from guardloop.demo import SAMPLE_DIFF, console, main, scripted_reviewer


@pytest.mark.unit
def test_demo_main_returns_ok():
    assert main([]) == 0


@pytest.mark.unit
def test_demo_main_delay_zero_does_not_sleep(monkeypatch):
    # If main honored a delay here it would call time.sleep; assert it does not
    # when delay is 0 (the CI-fast default).
    calls = []
    monkeypatch.setattr("guardloop.demo.time.sleep", lambda s: calls.append(s))
    assert main(["--demo-delay", "0"]) == 0
    assert calls == []


@pytest.mark.unit
def test_scripted_reviewer_concise_is_unparseable():
    # 'concise' strategy: no system marker -> prose without severity tags.
    text = scripted_reviewer([{"role": "system", "content": "Review the diff."}])
    with pytest.raises(ReviewParseError):
        parse_findings(text)


@pytest.mark.unit
def test_scripted_reviewer_structured_parses():
    text = scripted_reviewer([{"role": "system", "content": "You are a strict code reviewer"}])
    findings = parse_findings(text)
    assert len(findings) == 2
    assert all(f.severity == "HIGH" for f in findings)


@pytest.mark.unit
def test_scripted_reviewer_verify_confirms_all():
    text = scripted_reviewer([{"role": "system", "content": "Confirm or reject each finding"}])
    assert "CONFIRM" in text
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

    monkeypatch.setattr("guardloop.demo.main", fake_main)
    assert console() == 0
    assert captured["default_delay"] == 0.4
