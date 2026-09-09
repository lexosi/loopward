"""failure_classifier.py — a wire signal becomes a failure class.

Pure and caller-agnostic on purpose. :func:`classify` takes an exception (and the
provider whose endpoint answered) and returns a failure-class string. It holds no
reference to whoever called it, reads nothing but the exception, and never raises.

Why it is written this way: the one place that catches a failed run today,
``Orchestrator._finalize_crashed``, records and *re-raises* — it cannot act on a
verdict, only file one. Unit 3 needs a capture point *upstream* that can retry or
switch strategy, and it will call this **same** function from there. Written
isolated, that is one more call site, not a refactor.

Detection is by WIRE SIGNAL, never by exception class
-----------------------------------------------------
The exception class is not stable. The ``anthropic`` SDK pinned in this repo
(1.4.0) raises a *plain* ``ValueError`` ("Streaming is required...") that does not
exist in the ``>=0.39`` floor ``pyproject`` declares, so ``isinstance`` checks
rot across versions. What is stable is what came off the wire: the HTTP status,
the error body's ``type``, and the message. This module reads those three
(:class:`WireSignal`) and matches them against a data table (:data:`_TABLE`) — no
``if/elif`` chain of exception types.

What is measured, and only that
-------------------------------
One row, measured 2026-09-09: anthropic, HTTP 400,
``body.type == "invalid_request_error"``, an overflow message ->
:data:`CONTEXT_LENGTH_EXCEEDED`. Everything else -> :data:`UNKNOWN`. No other
class is declared, because declaring a class this repo has not measured would be
a fresh false claim in a codebase being cleaned of them. When a new class is
worth adding, the UNKNOWN branch in the crash trail is what will say so, with the
raw signal it could not place.

>>> classify(ValueError("some unmapped failure"), provider="claude")
'unknown'
>>> NEXT_STRATEGY[CONTEXT_LENGTH_EXCEEDED]
'chunk-diff'
"""

from __future__ import annotations

from dataclasses import dataclass

# --- failure classes -------------------------------------------------------
#: The one measured class: the provider refused the request because the prompt
#: did not fit the model's context window.
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
#: Everything the table does not place. Not an error — an instruction to the
#: trail to record the raw signal so the class worth adding next is visible.
UNKNOWN = "unknown"

# --- declared strategy destinations (NOT implemented in this unit) ----------
#: Where a :data:`CONTEXT_LENGTH_EXCEEDED` failure should be sent: split the diff
#: and review the pieces. The name is DECLARED here so the map has a real
#: destination; the strategy itself does not exist yet (it is not in
#: ``reviewer.STRATEGIES``) and building it is unit 3. There is deliberately no
#: empty placeholder and no ``NotImplementedError`` — a string in a dict is the
#: whole declaration.
CHUNK_DIFF_STRATEGY = "chunk-diff"

#: failure class -> the next strategy to try. A plain dict, extensible by adding
#: a row — no registry, no plugins, no entry points. One entry today.
NEXT_STRATEGY: dict[str, str] = {
    CONTEXT_LENGTH_EXCEEDED: CHUNK_DIFF_STRATEGY,
}


@dataclass(frozen=True)
class WireSignal:
    """What came off the wire, read from the exception. The unit of classification.

    ``http_status`` and ``body_type`` are ``None`` when the exception carried
    none — e.g. the anthropic 1.4.0 ``ValueError`` has no status at all, which is
    exactly why it must fall to UNKNOWN rather than be forced into a class.
    ``message`` is always a string (``str(exc)`` when the exception exposes no
    ``.message``), so the trail always has something literal to show.
    """

    provider: str | None
    http_status: int | None
    body_type: str | None
    message: str
    #: What extraction could NOT resolve, one short note per field, e.g.
    #: ``"http_status: absent"`` or ``"body_type: unreadable (RuntimeError: ...)"``.
    #: This is what keeps "never raises" from meaning "swallows in silence": a
    #: field that was genuinely absent and a field whose read *raised* both end up
    #: ``None`` above, and without this they would be indistinguishable — an
    #: ``except: pass`` with good manners. Empty when every field resolved.
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Rule:
    """One table row: a conjunction of field constraints -> a failure class.

    ``None`` on a field means "don't care". ``message_contains`` matches if ANY
    of its substrings is present (case-insensitive). :meth:`matches` is the one
    generic comparison applied to every row — the table is data, the matcher is
    not per-class branching.
    """

    failure_class: str
    provider: str | None
    http_status: int | None
    body_type: str | None
    message_contains: tuple[str, ...]

    def matches(self, signal: WireSignal) -> bool:
        if self.provider is not None and signal.provider != self.provider:
            return False
        if self.http_status is not None and signal.http_status != self.http_status:
            return False
        if self.body_type is not None and signal.body_type != self.body_type:
            return False
        if self.message_contains:
            low = signal.message.lower()
            if not any(sub in low for sub in self.message_contains):
                return False
        return True


#: The classification table. ONE row, measured 2026-09-09. The message
#: substrings are anthropic's known context-overflow phrasings; a real overflow
#: whose wording is not among them falls to UNKNOWN and lands in the trail with
#: its literal message — self-correcting by design, never a silent miss.
_TABLE: tuple[_Rule, ...] = (
    _Rule(
        failure_class=CONTEXT_LENGTH_EXCEEDED,
        provider="claude",
        http_status=400,
        body_type="invalid_request_error",
        message_contains=("prompt is too long", "context window", "maximum context"),
    ),
)


# Each reader returns ``(value, note)``. ``note`` is ``None`` when the field
# resolved; otherwise it is a short string saying why it did not — distinguishing
# "absent" from "the read raised". None of them raise: classify() runs inside
# _finalize_crashed, which must not, but silence is not the price of that — the
# note is the trace.


def _read_status(exc: BaseException) -> tuple[int | None, str | None]:
    try:
        value = getattr(exc, "status_code", None)
    except Exception as read_exc:
        return None, f"http_status: unreadable ({type(read_exc).__name__}: {read_exc})"
    if isinstance(value, int):
        return value, None
    if value is None:
        return None, "http_status: absent"
    return None, f"http_status: absent (not an int: {type(value).__name__})"


def _read_body_type(exc: BaseException) -> tuple[str | None, str | None]:
    """The error body's ``type``, read the way the SDKs expose it.

    The anthropic SDK parses ``body["error"]["type"]`` onto the exception as
    ``.type`` (see ``APIStatusError.__init__``), so that is tried first. Falling
    back to digging ``body["error"]["type"]`` covers an OpenAI-compatible body,
    whose top level has no ``type``. The bare ``body["type"]`` is deliberately
    NOT used: on anthropic that is the envelope value ``"error"``, not the
    discriminating ``"invalid_request_error"``. Every branch that fails to
    resolve says which shape it saw, so the trail shows *what* was malformed.
    """
    try:
        direct = getattr(exc, "type", None)
        if isinstance(direct, str):
            return direct, None
        body = getattr(exc, "body", None)
        if body is None:
            return None, "body_type: absent (no body)"
        if not isinstance(body, dict):
            return None, f"body_type: absent (body is {type(body).__name__}, not dict)"
        error = body.get("error")
        if not isinstance(error, dict):
            return None, "body_type: absent (body has no 'error' dict)"
        found = error.get("type")
        if isinstance(found, str):
            return found, None
        return None, "body_type: absent (error dict has no 'type')"
    except Exception as read_exc:
        return None, f"body_type: unreadable ({type(read_exc).__name__}: {read_exc})"


def _read_message(exc: BaseException) -> tuple[str, str | None]:
    try:
        message = getattr(exc, "message", None)
        if isinstance(message, str) and message:
            return message, None
        return str(exc), None
    except Exception as read_exc:
        return repr(exc), f"message: degraded to repr ({type(read_exc).__name__})"


def extract_signal(exc: BaseException, provider: str | None = None) -> WireSignal:
    """Read the wire signal off ``exc``. Never raises, never swallows.

    ``provider`` is passed in rather than sniffed: ``"invalid_request_error"`` is
    shared across providers, so the exception alone cannot say whose 400 it is,
    and the unstable exception class cannot be trusted to reveal it either. The
    caller — the crash handler now, unit 3's upstream capture point later — knows
    it from ``LLMClient.provider``. It is a datum, not a handle on the caller.

    Every field it could not resolve is named in :attr:`WireSignal.unresolved`,
    so a malformed exception leaves a trace of *what* was unreadable instead of a
    silent ``None``.
    """
    status, status_note = _read_status(exc)
    body_type, body_note = _read_body_type(exc)
    message, message_note = _read_message(exc)
    unresolved = tuple(note for note in (status_note, body_note, message_note) if note)
    return WireSignal(
        provider=provider,
        http_status=status,
        body_type=body_type,
        message=message,
        unresolved=unresolved,
    )


def classify_signal(signal: WireSignal) -> str:
    """Match a :class:`WireSignal` against the table; first hit wins, else UNKNOWN."""
    for rule in _TABLE:
        if rule.matches(signal):
            return rule.failure_class
    return UNKNOWN


def classify(exc: BaseException, provider: str | None = None) -> str:
    """Exception -> failure class. Pure, isolated, never raises."""
    return classify_signal(extract_signal(exc, provider))
