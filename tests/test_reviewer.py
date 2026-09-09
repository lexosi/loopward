"""Tests for the reviewer's parser and its prompts — the pair that has to agree.

The parser is what the whole anti-loop rests on: a `ReviewParseError` is the
failure signal that feeds the AttemptTracker. It had no test file at all, which
is how it went unnoticed that the *default* prompt could not satisfy it. The
default strategy read "Review the diff. List issues, most severe first." and
never mentioned a severity tag, a colon, or one-finding-per-line, so a compliant
model produced prose and the parse failed every single time. The anti-loop was
being demonstrated against a defect in the prompt that asked the question.

Two things are pinned here, because they broke in two different ways:

1. **The prompts ask for the format the parser demands.** `test_every_strategy_*`
   below is the regression guard for the finding above.
2. **The parser accepts what a real model actually returns.** `- HIGH: …` was
   rejected — a markdown list is the single most likely shape a model produces,
   and the regex anchored the tag with `^\\s*`, so the bullet defeated it. The
   accept/reject tables below are transcribed from the measurements taken when
   that was fixed; they exist so that hardening the regex tomorrow fails loudly
   here instead of silently narrowing what counts as a finding.
"""

import pytest

from loopward.agents.reviewer import (
    _STRATEGY_INSTRUCTIONS,
    STRATEGIES,
    Reviewer,
    ReviewParseError,
    UnknownStrategyError,
    parse_findings,
    scan_findings,
)
from loopward.engine.llm_wrapper import LLMClient

# ---- what the parser accepts ------------------------------------------------
# One list marker in front of the tag. Measured, not assumed: every one of these
# was rejected before the marker was allowed.

ACCEPTED_MARKERS = [
    ("dash", "- HIGH: off-by-one in loop bound."),
    ("asterisk", "* HIGH: off-by-one in loop bound."),
    ("plus", "+ HIGH: off-by-one in loop bound."),
    ("numbered dot", "1. HIGH: off-by-one in loop bound."),
    ("numbered paren", "1) HIGH: off-by-one in loop bound."),
    ("indented dash", "  - HIGH: off-by-one in loop bound."),
    ("bare tag", "HIGH: off-by-one in loop bound."),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,line", ACCEPTED_MARKERS, ids=[c[0] for c in ACCEPTED_MARKERS])
def test_one_list_marker_before_the_tag_is_accepted(label, line):
    findings = parse_findings(line)
    assert len(findings) == 1
    assert findings[0].severity == "HIGH"
    # The marker is not part of the finding: it is stripped, not carried along.
    assert findings[0].message == "off-by-one in loop bound."


# ---- what the parser rejects, deliberately ----------------------------------
# Not oversight. Widening to cover these would mean parsing decoration rather
# than a format, and the failure signal is worth more than the extra finding.

REJECTED_DECORATION = [
    ("bold inside the tag", "**HIGH**: off-by-one in loop bound."),
    ("markdown heading", "### HIGH: off-by-one in loop bound."),
    ("tag not at line start", "I would rate HIGH: off-by-one in loop bound."),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,line", REJECTED_DECORATION, ids=[c[0] for c in REJECTED_DECORATION])
def test_decoration_around_the_tag_is_rejected_on_purpose(label, line):
    with pytest.raises(ReviewParseError):
        parse_findings(line)


# ---- replies with no severity tag at all ------------------------------------
# The shapes a model actually returns when the prompt does not ask for the
# format. Every one of these is a genuine parse failure and must stay one.

NO_TAG_AT_ALL = [
    (
        "markdown bullets",
        "- The expiry check uses `<=`, accepting tokens at the boundary.\n"
        "- Off-by-one in the loop bound.",
    ),
    (
        "numbered list",
        "1. Expiry comparison changed to `<=` — accepts expired tokens.\n"
        "2. `range(len(items)+1)` reads past the end.",
    ),
    (
        "prose",
        "The most severe issue is the expiry boundary change. Secondly, there is an off-by-one.",
    ),
    ("severity word in parens", "1. (High) expiry boundary\n2. (Medium) off-by-one"),
    ("heading and bold", "### Issues\n**Critical** - expiry check\n**High** - off-by-one"),
    ("severity word, no colon", "HIGH severity - the expiry check accepts boundary tokens."),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,text", NO_TAG_AT_ALL, ids=[c[0] for c in NO_TAG_AT_ALL])
def test_reply_without_a_severity_tag_raises(label, text):
    with pytest.raises(ReviewParseError):
        parse_findings(text)


@pytest.mark.unit
def test_unmatched_lines_are_counted_not_hidden():
    # A partial match is accepted, and the dropped lines travel with it so the
    # trail can show the finding came out of a wall of something else.
    findings, dropped = scan_findings("Here is my review:\n- HIGH: real finding.\nHope that helps!")
    assert [f.message for f in findings] == ["real finding."]
    assert dropped == 2


# ---- the prompts must ask for what the parser demands -----------------------
# The regression guard for the founder finding: a default strategy whose output
# the repo's own parser could not read.


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_states_the_format_contract(strategy):
    instruction = _STRATEGY_INSTRUCTIONS[strategy]
    lowered = instruction.lower()
    assert "severity" in lowered, f"{strategy!r} never mentions a severity tag"
    assert "colon" in lowered, f"{strategy!r} never mentions the colon"
    assert "per line" in lowered, f"{strategy!r} never asks for one finding per line"


@pytest.mark.unit
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_gives_an_example_its_own_parser_accepts(strategy):
    # The strongest form of the guard: whatever each prompt holds up as a model
    # answer must survive parse_findings. A prompt whose own example does not
    # parse is the defect this file exists to prevent coming back.
    example = _STRATEGY_INSTRUCTIONS[strategy].splitlines()[-1]
    findings = parse_findings(example)
    assert len(findings) == 1


@pytest.mark.unit
def test_the_two_strategies_are_not_the_same_prompt():
    # Both must state the contract; neither fix may be a copy-paste of the
    # other. Identical instructions would make the class-jump a switch between
    # two indistinguishable approaches — a different untruth in the same place.
    assert _STRATEGY_INSTRUCTIONS["concise"] != _STRATEGY_INSTRUCTIONS["structured"]


@pytest.mark.unit
def test_default_strategy_is_the_first_one_and_is_a_real_strategy():
    # STRATEGIES[0] is the strategy a review starts on. It has to be a real,
    # prompt-backed strategy — that half is unchanged. What used to sit here too,
    # that an unknown strategy silently borrowed this same wording, is gone: an
    # unknown strategy is now rejected (see the test below), not resolved.
    assert STRATEGIES[0] == "concise"
    assert "concise" in _STRATEGY_INSTRUCTIONS
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "HIGH: x")
    default = Reviewer(llm).build_messages("diff", "concise")
    assert _STRATEGY_INSTRUCTIONS["concise"] in default[0]["content"]


@pytest.mark.unit
def test_an_unknown_strategy_is_rejected_with_its_name_not_silently_resolved():
    # The reviewer builds no prompt for a strategy it does not know. It raises,
    # and the cause names the missing strategy, so an unbuilt class-jump
    # destination cannot slip through as 'concise' and have the trail record a
    # jump that never happened.
    llm = LLMClient(provider="fake", fake_script=lambda m, task: "HIGH: x")
    with pytest.raises(UnknownStrategyError) as exc_info:
        Reviewer(llm).build_messages("diff", "no-such-strategy")
    assert "no-such-strategy" in str(exc_info.value)
