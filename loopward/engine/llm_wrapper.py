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
#: is being asked -- reword a prompt and text-sniffing fails silently, which is
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
        ``(messages) -> str``. If omitted, a built-in heuristic replies.
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

    @property
    def totals(self) -> dict[str, float]:
        """Accumulated usage across all calls: prompt, completion, cost_usd, calls."""
        return dict(self._totals)

    # ---- public API -------------------------------------------------------

    def complete(self, messages: list[Message], *, task: str | None = None) -> LLMResponse:
        """Run one completion. Routes to the configured provider.

        ``task`` is the caller's declared intent (:data:`TASK_REVIEW` /
        :data:`TASK_VERIFY`). Only the fake reads it; real providers never see
        it, which is why it is a keyword argument and not a message field.
        """
        if self.provider == "fake":
            text = self._complete_fake(messages, task)
        elif self.provider == "deepseek":
            text = self._complete_deepseek(messages)
        else:
            text = self._complete_claude(messages)

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
        )

    # ---- providers --------------------------------------------------------

    def _complete_fake(self, messages: list[Message], task: str | None) -> str:
        self._fake_calls += 1
        script = self._fake_script
        if script is None:
            return _fake_heuristic(messages, task)
        if callable(script):
            return script(messages, task)
        # sequence: consume in order, clamp to last when exhausted
        idx = min(self._fake_calls - 1, len(script) - 1)
        return script[idx]

    def _complete_deepseek(self, messages: list[Message]) -> str:
        if self._client is None:
            from openai import OpenAI  # lazy: optional dependency

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set (provider=deepseek)")
            self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def _complete_claude(self, messages: list[Message]) -> str:
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
        return "".join(block.text for block in resp.content if block.type == "text")


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
