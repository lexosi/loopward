"""chunk_diff.py — split a diff into per-file pieces, and the prompt to review one.

This is the destination of the failure-class map in
``engine.failure_classifier``: a ``context_length_exceeded`` failure should be
sent here, to review the diff a piece at a time instead of whole. It is reached
by a MEASURED failure class, never by exhausting the ordered strategy list — the
two are different mechanisms, and this one is deliberately kept OUT of
``reviewer.STRATEGIES`` so that adding it does not lengthen the ordered list and
move the anti-loop's hard bound.

The dependency direction is ``orchestrator -> agents -> engine``. This module is
an agent, so it may lean on ``engine``; ``engine.failure_classifier`` must not
import it back — its map holds the destination as a bare string, and a string in
a dict needs no import. That keeps the graph acyclic.

What "split" means, and its two declared limits
------------------------------------------------
One piece per file, whole file at a time — never split within a file by hunk.
Each piece is a standalone diff: it carries its own file header and hunks, and
re-splitting a piece returns that same piece unchanged.

- **Too many files.** More than :data:`MAX_CHUNKS` files is not split at all.
  :func:`split_diff` raises :class:`TooManyFilesToChunk`, which names both the
  file count it saw and the cap it exceeded. The limit is declared, not hidden.
- **Unusable input.** An empty diff, or one with no recognisable file header,
  raises :class:`MalformedDiffError` naming the cause. It never returns an empty
  list: "zero files to review" handed back silently would let a caller certify a
  review of nothing, the same swallow this package exists to catch.

One further limit is real but NOT modelled here: a single file whose own diff
still overflows the context window. At that point there is no smaller unit to
split into — one file is already the floor. The chunked review has a caller now
and does not smooth this over: a piece whose own review overflows raises straight
through to the crash path, the same as any unmapped failure. A dedicated symbol
for "one file is still too big" is built when it earns one.

The orchestrator's review loop reaches this module when a context-length
overflow is classified: it splits the diff and reviews the pieces one at a time
(``Orchestrator._review_chunked``), ending the run in ``partial_review``.
"""

from __future__ import annotations

from loopward.agents.reviewer import SEVERITIES

#: The most files chunk-diff will split a diff into. A diff with more than this
#: is refused whole (see :class:`TooManyFilesToChunk`) rather than split, so the
#: number of review pieces has a fixed ceiling.
MAX_CHUNKS = 10

#: Prompt for reviewing ONE file of a larger diff. It states the same format
#: contract the reviewer's parser demands — a severity tag, a colon, one finding
#: per line — because a prompt whose output that parser cannot read is the
#: founding bug of this repo. The last line is an example the parser accepts.
CHUNK_DIFF_INSTRUCTION = (
    "Review this one file from a larger diff, most severe first. One finding per "
    "line, each line starting with a severity tag from "
    f"{', '.join(SEVERITIES)} and a colon — no bullet or number before the tag. "
    "Report only on the file shown. Example:\n"
    "HIGH: off-by-one in loop bound."
)


class MalformedDiffError(ValueError):
    """Raised when a diff cannot be split because it has no usable structure.

    Empty, whitespace-only, or missing any recognisable file header. Raising
    rather than returning ``[]`` is the point: an empty result would read as
    "nothing to review" and let the caller record a review of nothing.
    """


class TooManyFilesToChunk(ValueError):
    """Raised when a diff has more files than :data:`MAX_CHUNKS` allows.

    Carries the ``count`` it saw and the ``cap`` it exceeded, so the limit lands
    in the trail as a declared number, not as a silent truncation.
    """

    def __init__(self, count: int, cap: int) -> None:
        self.count = count
        self.cap = cap
        super().__init__(
            f"diff has {count} files, over the chunk-diff cap of {cap} "
            f"(MAX_CHUNKS); not split. Review it whole or reduce its scope."
        )


def _file_start_indices(lines: list[str]) -> list[int]:
    """Line indices where each file's section begins.

    Git diffs open every file with ``diff --git``; when any such line is present
    it is the authoritative boundary. A plain unified diff has no such line, so
    the boundary is the ``--- `` old-file header — but only when the very next
    line is its ``+++ `` partner, which is what tells a real file header apart
    from a ``---``-looking line inside a hunk body.
    """
    git_starts = [i for i, line in enumerate(lines) if line.startswith("diff --git ")]
    if git_starts:
        return git_starts
    starts: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            starts.append(i)
    return starts


def split_diff(diff: str) -> list[str]:
    """Split ``diff`` into one standalone piece per file.

    Returns a non-empty list of diffs, each reviewable on its own. Raises
    :class:`MalformedDiffError` on input with no usable file header, and
    :class:`TooManyFilesToChunk` when the file count exceeds :data:`MAX_CHUNKS`.
    """
    if not diff or not diff.strip():
        raise MalformedDiffError("empty diff: there is nothing to split")

    lines = diff.splitlines(keepends=True)
    starts = _file_start_indices(lines)
    if not starts:
        raise MalformedDiffError(
            "unrecognisable diff: found neither a 'diff --git' header nor a "
            "'--- '/'+++ ' file-header pair, so no file boundary could be located"
        )

    bounds = starts + [len(lines)]
    pieces = ["".join(lines[bounds[i] : bounds[i + 1]]) for i in range(len(starts))]

    if len(pieces) > MAX_CHUNKS:
        raise TooManyFilesToChunk(len(pieces), MAX_CHUNKS)
    return pieces
