"""Tests for the anti-loop tracker."""

import pytest

from guardloop.engine.anti_loop import (
    CLASS_JUMP,
    MAX_ATTEMPTS,
    RETRY,
    AttemptTracker,
    decide,
)


@pytest.mark.unit
def test_decide_is_pure_and_thresholded():
    assert decide(1) == RETRY
    assert decide(MAX_ATTEMPTS - 1) == RETRY
    assert decide(MAX_ATTEMPTS) == CLASS_JUMP
    assert decide(MAX_ATTEMPTS + 5) == CLASS_JUMP


@pytest.mark.unit
def test_retries_then_forces_class_jump():
    # Arrange
    t = AttemptTracker(max_attempts=3)

    # Act / Assert: first two failures retry, third forces a class-jump.
    assert t.record_failure("sub", "regex").action == RETRY
    assert t.record_failure("sub", "regex").action == RETRY
    outcome = t.record_failure("sub", "regex")
    assert outcome.action == CLASS_JUMP
    assert outcome.must_class_jump is True
    assert outcome.attempt == 3


@pytest.mark.unit
def test_counts_are_per_subtask():
    t = AttemptTracker(max_attempts=2)
    t.record_failure("a", "s")
    assert t.attempts("a") == 1
    assert t.attempts("b") == 0


@pytest.mark.unit
def test_reset_clears_counter():
    t = AttemptTracker(max_attempts=2)
    t.record_failure("a", "s")
    t.reset("a")
    assert t.attempts("a") == 0


@pytest.mark.unit
def test_audit_sink_receives_events():
    events = []
    t = AttemptTracker(max_attempts=2, audit=lambda kind, msg: events.append((kind, msg)))
    t.record_failure("a", "s")
    assert events and events[0][0] == "attempt"


@pytest.mark.unit
def test_invalid_max_attempts_rejected():
    with pytest.raises(ValueError):
        AttemptTracker(max_attempts=0)
