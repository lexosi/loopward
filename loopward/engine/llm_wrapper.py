"""llm_wrapper.py — one client, multiple providers.

Providers
---------
- ``deepseek``  OpenAI-compatible endpoint (``base_url=https://api.deepseek.com``).
- ``claude``    Anthropic Messages API.
- ``fake``      Deterministic, offline. No network, no API key. Powers the demo
                and the test suite so everything runs green without secrets.

The real providers import their SDKs lazily, so installing them is optional
(see the ``deepseek`` / ``claude`` extras in ``pyproject.toml``). API keys are
read from the environment only — never hardcoded, never passed positionally.

Token accounting and the empty-response guard are distilled from a production
wrapper: a blank completion raises loudly instead of silently returning "".

The provider's own reason for stopping travels with the reply. Both real
providers report one — Anthropic as ``stop_reason``, OpenAI-compatible ones as
``finish_reason`` — and both used to be discarded here, which made a truncated
answer indistinguishable from a complete one anywhere upstream. It is carried
**raw and unnormalised**; see :attr:`LLMResponse.stop_reason`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}

# USD per 1M tokens, cache-miss rates. Unknown (provider, model) pairs cost 0
# and emit no charge — keeps the fake provider free and avoids guessing.
#
# The 0 is not marked as such downstream. A trail from an unpriced pair reports
# `"cost_usd": 0.0`, which reads the same as a run that genuinely cost nothing;
# only the provider/model in the same envelope tells them apart. The benchmark
# carries a `priced` flag for exactly this and the trail does not. Read a zero
# here as "not priced", not as "free".
#
# Prices are published rates and drift over time; last verified 2026-07-31.
# Re-check the providers' pricing pages before relying on the cost figures.
PRICING: dict[tuple[str, str], dict[str, float]] = {
    ("deepseek", "deepseek-v4-flash"): {"prompt": 0.14, "completion": 0.28},
    ("deepseek", "deepseek-v4-pro"): {"prompt": 0.435, "completion": 0.87},
    ("claude", "claude-haiku-4-5"): {"prompt": 1.0, "completion": 5.0},
    ("claude", "claude-sonnet-4-6"): {"prompt": 3.0, "completion": 15.0},
}

DEFAULT_MODEL: dict[str, str] = {
    "deepseek": "deepseek-v4-flash",
    "claude": "claude-haiku-4-5",
    "fake": "fake-1",
}

#: The caller's declared intent for a completion, passed OUT OF BAND: it never
#: enters the message payload, because real providers reject unknown keys. The
#: fake routes on this and must never sniff the prompt text to work out what it
#: is being asked — reword a prompt and text-sniffing fails silently, which is
#: the failure this constant exists to prevent.
TASK_REVIEW = "review"  # the Reviewer sends f"{TASK_REVIEW}:{strategy}"
TASK_VERIFY = "verify"

# A scripted fake is either a fixed list of replies (consumed in order) or a
# function mapping (messages, task) to a reply string.
FakeScript = Sequence[str] | Callable[[list[Message], str | None], str]

# Numbered findings in a verification request: "1. HIGH: ...".
_LISTED_RE = re.compile(r"^\s*(\d+)\.\s", re.MULTILINE)


@dataclass(frozen=True)
class LLMResponse:
    """Result of one completion call."""

    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    #: Why the provider stopped generating, **verbatim and unnormalised**:
    #: Anthropic's ``stop_reason`` (``end_turn``, ``max_tokens``, ``refusal``,
    #: ``model_context_window_exceeded``, ...) or an OpenAI-compatible
    #: ``finish_reason`` (``stop``, ``length``, ``content_filter``, ...).
    #:
    #: Not translated into a common vocabulary, on purpose. The two sets are
    #: not in one-to-one correspondence — ``refusal`` has no OpenAI counterpart
    #: and ``content_filter`` has no Anthropic one — so deciding that
    #: ``length`` and ``max_tokens`` are the same thing is a mapping decision,
    #: not a transport one. The raw strings happen to be disjoint between the
    #: two vocabularies, so a value identifies its own provider; :attr:`provider`
    #: is in this same object either way.
    #:
    #: ``None`` when the provider reported nothing: the ``fake`` provider always
    #: (it does not know why it stopped, and saying ``end_turn`` would be it
    #: asserting something it cannot know), and a real provider on an SDK version
    #: predating the field — see :meth:`_complete_claude`.
    #:
    #: **Nothing branches on this.** It is recorded, not consulted.
    stop_reason: str | None = None


def _estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


def _calc_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING.get((provider, model))
    if not rates:
        return 0.0
    cost = (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1_000_000
    return round(cost, 6)


class EmptyCompletionError(RuntimeError):
    """Raised when a provider returns blank content — a loud failure, not a silent ""."""


class LLMClient:
    """Provider-agnostic chat client.

    Parameters
    ----------
    provider:
        ``"deepseek"`` | ``"claude"`` | ``"fake"``.
    model:
        Model id. Defaults to a sensible per-provider model. Always
        configurable; never hardcoded at the call site.

        The promise covers ``model`` and nothing else. The rest of the request
        is fixed in the provider methods and not exposed: ``claude`` sends
        ``max_tokens=4096`` (:meth:`_complete_claude`) and ``deepseek`` sends no
        ``max_tokens`` at all (:meth:`_complete_deepseek`), so the two providers
        truncate differently and neither says so. Making those configurable is
        open work, not a claim this class currently meets.
    fake_script:
        Only for ``provider="fake"``. A list of canned replies or a function
        ``(messages, task) -> str`` — it is called with both arguments, as
        :data:`FakeScript` types it and :meth:`_complete_fake` does. If
        omitted, a built-in heuristic replies.
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        fake_script: FakeScript | None = None,
    ) -> None:
        if provider not in ("deepseek", "claude", "fake"):
            raise ValueError(f"unknown provider {provider!r} (use deepseek|claude|fake)")
        self.provider = provider
        self.model = model or DEFAULT_MODEL[provider]
        self._fake_script = fake_script
        self._fake_calls = 0
        self._client = None  # lazily constructed for real providers
        self._totals = {"prompt": 0, "completion": 0, "cost_usd": 0.0, "calls": 0}
        self._stop_reasons: list[str | None] = []

    @property
    def totals(self) -> dict[str, float]:
        """Accumulated usage across all calls: prompt, completion, cost_usd, calls."""
        return dict(self._totals)

    @property
    def stop_reasons(self) -> list[str | None]:
        """Every call's raw stop reason, in call order. A copy; never the list itself.

        Ordered rather than counted or de-duplicated, because a count is already
        a summary and a set loses how many. A reader lining this up against the
        trail's attempt events can tell *which* call truncated, not just that one
        did.

        It can be **longer** than ``totals["calls"]``, and the gap is meaningful:
        a reason is recorded before the blank-content guard in :meth:`complete`,
        while ``calls`` is incremented after it. So a provider that truncates at
        zero tokens contributes a ``max_tokens`` here and nothing there — which
        is precisely the case that must not go unrecorded, since it raises
        :class:`EmptyCompletionError` and ends the run.
        """
        return list(self._stop_reasons)

    # ---- public API -------------------------------------------------------

    def complete(self, messages: list[Message], *, task: str | None = None) -> LLMResponse:
        """Run one completion. Routes to the configured provider.

        ``task`` is the caller's declared intent (:data:`TASK_REVIEW` /
        :data:`TASK_VERIFY`). Only the fake reads it; real providers never see
        it, which is why it is a keyword argument and not a message field.
        """
        if self.provider == "fake":
            text, stop_reason = self._complete_fake(messages, task)
        elif self.provider == "deepseek":
            text, stop_reason = self._complete_deepseek(messages)
        else:
            text, stop_reason = self._complete_claude(messages)

        # Recorded BEFORE the blank-content guard, deliberately. A provider that
        # truncates at zero tokens reports `max_tokens` and returns "", and that
        # pair is the most informative thing this method ever sees — it is also
        # the one that ends the run. Recording it after the raise would lose it
        # exactly when it matters. `_totals` stays where it was, below the guard,
        # so no counter moves; `stop_reasons` documents the asymmetry.
        self._stop_reasons.append(stop_reason)

        if not text or not text.strip():
            raise EmptyCompletionError(
                f"empty content from provider={self.provider} model={self.model}"
            )

        prompt_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        completion_tokens = _estimate_tokens(text)
        cost = _calc_cost(self.provider, self.model, prompt_tokens, completion_tokens)
        self._totals["prompt"] += prompt_tokens
        self._totals["completion"] += completion_tokens
        self._totals["cost_usd"] = round(self._totals["cost_usd"] + cost, 6)
        self._totals["calls"] += 1
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            stop_reason=stop_reason,
        )

    # ---- providers --------------------------------------------------------

    def _complete_fake(self, messages: list[Message], task: str | None) -> tuple[str, None]:
        """Reply plus a stop reason that is always ``None``.

        The fake does not know why it stopped, so it says so. Reporting
        ``end_turn`` would be the offline provider asserting a fact only a real
        one can establish. Widening :data:`FakeScript` so a script *can* express
        a truncation is separate, open work; it stays ``-> str`` here.
        """
        self._fake_calls += 1
        script = self._fake_script
        if script is None:
            return _fake_heuristic(messages, task), None
        if callable(script):
            return script(messages, task), None
        # sequence: consume in order, clamp to last when exhausted
        idx = min(self._fake_calls - 1, len(script) - 1)
        return script[idx], None

    def _complete_deepseek(self, messages: list[Message]) -> tuple[str, str | None]:
        """Reply plus the OpenAI-compatible ``finish_reason``, read defensively.

        ``getattr`` rather than attribute access, for two independent reasons.
        ``pyproject`` floors the extra at ``openai>=1.40``, so the field can be
        absent on an older client object; and the value on the wire comes from
        DeepSeek's server, not from the SDK's type — ``finish_reason`` is typed
        as a required ``Literal`` in ``openai``, which states what OpenAI sends,
        not what an OpenAI-*compatible* endpoint does.

        The SDK never raises on it. ``LengthFinishReasonError`` and
        ``ContentFilterFinishReasonError`` exist, but only the ``.parse()`` and
        streaming helpers raise them; the plain ``create()`` below returns a
        truncated reply with ``finish_reason="length"`` and HTTP 200.
        """
        if self._client is None:
            from openai import OpenAI  # lazy: optional dependency

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set (provider=deepseek)")
            self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        choice = resp.choices[0]
        return choice.message.content or "", getattr(choice, "finish_reason", None)

    def _complete_claude(self, messages: list[Message]) -> tuple[str, str | None]:
        """Reply plus Anthropic's ``stop_reason``, read defensively.

        ``getattr`` because ``pyproject`` floors the extra at ``anthropic>=0.39``
        and the value set has grown since: ``model_context_window_exceeded`` is
        in the ``StopReason`` literal of the SDK this repo runs against (1.1.0)
        and arrives as **HTTP 200**, not the 400 a context-length failure is
        usually assumed to be. Nothing here matches on the value, so a member
        added by a future version travels through untouched — which is the
        point of carrying the string raw.

        Not read: ``stop_details``, the structured refusal category. It is
        populated only when ``stop_reason == "refusal"`` and is GA from Opus 4.7
        onward, while this client's default Claude model is ``claude-haiku-4-5``.
        Open work, not an omission.
        """
        if self._client is None:
            import anthropic  # lazy: optional dependency

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set (provider=claude)")
            self._client = anthropic.Anthropic(api_key=api_key)
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        convo = [m for m in messages if m.get("role") != "system"]
        resp = self._client.messages.create(
            model=self.model, max_tokens=4096, system=system or None, messages=convo
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return text, getattr(resp, "stop_reason", None)


def _fake_heuristic(messages: list[Message], task: str | None = None) -> str:
    """Default offline reply: a tiny deterministic 'code reviewer' + 'verifier'.

    Looks at the planted smells in the diff so the demo produces believable,
    repeatable findings without a network call. Which role it plays is decided
    by ``task``, never by matching the prompt's wording.
    """
    if task == TASK_VERIFY:
        # One line per listed finding: the adjudication has to cover 1..n or
        # the verifier rejects it as unparseable.
        listing = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        n = max(1, len(_LISTED_RE.findall(listing)))
        return "\n".join(f"CONFIRM {i}" for i in range(1, n + 1))

    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    findings: list[str] = []
    if "<=" in last_user and "expir" in last_user.lower():
        findings.append("HIGH: token expiry uses `<=`; expired-at-boundary tokens pass. Use `<`.")
    if "range(" in last_user and "len(" in last_user and "+ 1" in last_user:
        findings.append("HIGH: off-by-one; `range(len(x) + 1)` indexes past the end.")
    if "except" in last_user and "pass" in last_user:
        findings.append("MEDIUM: bare `except: pass` swallows errors. Handle or re-raise.")
    if not findings:
        findings.append("LOW: no blocking issues found in the provided diff.")
    return "\n".join(findings)
