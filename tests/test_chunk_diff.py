"""Tests for chunk_diff — the map's destination, built as a standalone agent.

chunk-diff is reached by a MEASURED failure class (``context_length_exceeded``),
never by exhausting the ordered strategy list. Those are two different
mechanisms and this file pins the one built here: ``split_diff`` cuts a diff into
one piece per file, ``MAX_CHUNKS`` caps how many files it will cut, and
``CHUNK_DIFF_INSTRUCTION`` carries a prompt that asks for the same severity-tag
format the reviewer's parser demands.

Two header shapes are tested SEPARATELY, on purpose. A git diff opens each file
with ``diff --git a/… b/…``; a plain unified diff has only the ``--- ``/``+++ ``
pair. The repo's own fixtures (``demo.SAMPLE_DIFF``, ``bench.HARD_DIFF``) use the
bare pair with no ``diff --git`` line, so a splitter that only understood git
headers would pass a mixed test and still be blind to every fixture in the tree.

The orchestrator now reaches this on a context-length overflow — the wiring and
its ``partial_review`` outcome are tested in ``tests/test_chunk_route.py``; this
file pins the splitter itself. The review-time limit — a single file whose own
diff still overflows the context window, where no further splitting is possible —
is still not modelled here: a piece that overflows on review raises through to
the crash path, and a dedicated symbol for it is built when it earns one.
"""

import pytest

from loopward.agents.chunk_diff import (
    CHUNK_DIFF_INSTRUCTION,
    MAX_CHUNKS,
    MalformedDiffError,
    TooManyFilesToChunk,
    split_diff,
)
from loopward.agents.reviewer import parse_findings


def _bare(n: int) -> str:
    """One file of a plain unified diff (no ``diff --git`` line)."""
    return f"--- a/f{n}.py\n+++ b/f{n}.py\n@@ -1,2 +1,2 @@\n-old{n}\n+new{n}\n"


def _git(n: int) -> str:
    """One file of a git-format diff (opens with ``diff --git``)."""
    return (
        f"diff --git a/f{n}.py b/f{n}.py\n"
        f"index 1111111..2222222 100644\n"
        f"--- a/f{n}.py\n+++ b/f{n}.py\n@@ -1,2 +1,2 @@\n-old{n}\n+new{n}\n"
    )


# ---- one piece per file, each header shape on its own -----------------------


@pytest.mark.unit
def test_split_bare_diff_of_n_files_yields_n_pieces():
    diff = "".join(_bare(i) for i in range(3))
    assert len(split_diff(diff)) == 3


@pytest.mark.unit
def test_split_git_diff_of_n_files_yields_n_pieces():
    diff = "".join(_git(i) for i in range(3))
    assert len(split_diff(diff)) == 3


# ---- the cap: over MAX_CHUNKS files is refused, count and cap both named -----


@pytest.mark.unit
def test_the_cap_is_ten():
    assert MAX_CHUNKS == 10


@pytest.mark.unit
def test_a_diff_over_the_cap_is_not_chunked_and_names_the_count_and_the_cap():
    diff = "".join(_bare(i) for i in range(MAX_CHUNKS + 1))
    with pytest.raises(TooManyFilesToChunk) as exc_info:
        split_diff(diff)
    # The limit is declared, not hidden: the count that tripped it and the cap
    # it tripped against are both on the exception and both in its message.
    assert exc_info.value.count == MAX_CHUNKS + 1
    assert exc_info.value.cap == MAX_CHUNKS
    message = str(exc_info.value)
    assert str(MAX_CHUNKS + 1) in message
    assert str(MAX_CHUNKS) in message


# ---- each piece stands alone as a diff --------------------------------------


@pytest.mark.unit
def test_each_bare_piece_is_a_valid_standalone_diff():
    diff = "".join(_bare(i) for i in range(3))
    for piece in split_diff(diff):
        # Idempotent: a single-file piece splits back into exactly itself.
        assert split_diff(piece) == [piece]
        assert piece.startswith("--- ")
        assert "@@ " in piece


@pytest.mark.unit
def test_each_git_piece_is_a_valid_standalone_diff():
    diff = "".join(_git(i) for i in range(3))
    for piece in split_diff(diff):
        assert split_diff(piece) == [piece]
        assert piece.startswith("diff --git ")
        assert "@@ " in piece


# ---- unusable input fails loud; it never returns an empty list --------------
# The same defect the previous commit killed on the classifier, wearing another
# face: an empty result would read as "zero files to review" and the trail would
# certify a review of nothing. So split_diff raises, naming the cause, rather
# than handing back [].

MALFORMED = [
    ("empty", ""),
    ("whitespace only", "   \n\t\n"),
    ("no diff headers at all", "just some notes\nnothing that looks like a diff\n"),
    ("--- without a +++ pair", "--- a/f.py\nno plus line follows this one\n"),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,diff", MALFORMED, ids=[c[0] for c in MALFORMED])
def test_split_diff_fails_loud_on_unusable_input(label, diff):
    with pytest.raises(MalformedDiffError):
        split_diff(diff)


# ---- the prompt asks for the format the parser demands ----------------------
# The equivalent of test_reviewer.py's per-strategy sweep. chunk-diff is not in
# STRATEGIES, so that sweep never reaches it; this is its own guard for the same
# founding bug — a prompt whose output the repo's own parser cannot read.


@pytest.mark.unit
def test_chunk_diff_prompt_states_the_same_format_contract():
    lowered = CHUNK_DIFF_INSTRUCTION.lower()
    assert "severity" in lowered, "chunk-diff prompt never mentions a severity tag"
    assert "colon" in lowered, "chunk-diff prompt never mentions the colon"
    assert "per line" in lowered, "chunk-diff prompt never asks for one finding per line"


@pytest.mark.unit
def test_chunk_diff_prompt_example_parses_under_the_reviewers_own_parser():
    example = CHUNK_DIFF_INSTRUCTION.splitlines()[-1]
    findings = parse_findings(example)
    assert len(findings) == 1
