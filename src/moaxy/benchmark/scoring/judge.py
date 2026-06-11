"""LLM-as-judge scorer for the M7 benchmark harness.

The :mod:`moaxy.benchmark.scoring.judge` module owns the
:class:`LLMJudgeScorer` — the LLM-as-judge implementation that
scores the ``explain`` category's responses on a 0-10 rubric.

The judge model is the cheapest locally-available model on Ollama
(``deepseek-v4-pro:cloud``). The scorer is intentionally cheap
because the benchmark issues one judge call per ``explain``
prompt per cell (3 prompts × 8 cells = 24 judge calls per live
run). Using the cheapest local model keeps the live run inside
the user's local-Ollama compute envelope; the judge is not
expected to be perfect, just directionally useful.

The contract (VAL-BENCH-006, VAL-BENCH-007) requires:

* A known judge response (e.g. ``"The code is correct and clear.
  <SCORE> 8 </SCORE>"``) is parsed and the scorer returns
  ``8.0``.
* A known judge response with a lowercase tag
  (``"<score>10</score>"``) is parsed and returns ``10.0``.
* A malformed judge response (no ``<SCORE>`` tag, no parseable
  integer, empty string) returns ``5.0`` — the documented
  default. A single bad judge call must not break the
  benchmark.
* All scores are in ``[0, 10]`` (the scorer's public surface
  is a float in ``[0, 10]``; the report generator
  normalises to ``[0, 1]`` at aggregation time).

The scorer is async: ``async def score(prompt, model_output) ->
float``. The judge model is called via an
:class:`httpx.AsyncClient` to the local Ollama API (the same
``/v1/chat/completions`` endpoint the moaxy proxy uses). The
scorer can also be constructed with an in-process
:class:`httpx.AsyncBaseTransport` for hermetic unit tests — the
:class:`tests.test_benchmark.TestLLMJudgeScorer` class exercises
that path with canned judge responses.

The judge prompt is intentionally short: the scorer asks the
judge to evaluate the response on three criteria (correctness,
completeness, clarity) and to emit a structured ``<SCORE> N
</SCORE>`` tag on the last line of its reply. The structured
output is then parsed by a regex (with a fallback to a bare
integer line) so the scorer is robust to minor formatting drift
in the judge's response.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# The default judge model. The M7 spec fixes this as the
# cheapest model on local Ollama (``deepseek-v4-pro:cloud``).
# The ``LLMJudgeScorer`` constructor accepts an alternate model
# for tests, but the production benchmark always uses the
# default; the contract pins the model's name (not just its
# shape) so a future edit that swaps to a more expensive model
# is caught by a downstream test.
DEFAULT_JUDGE_MODEL: str = "deepseek-v4-pro:cloud"
"""The default judge model: the cheapest model on local Ollama."""

# The default base URL for the local Ollama API. The judge
# adapter uses the same ``/v1/chat/completions`` path the moaxy
# proxy uses; only the client (httpx vs. the OpenAI-shaped
# moaxy adapter) differs.
DEFAULT_JUDGE_BASE_URL: str = "http://127.0.0.1:11434"
"""The default Ollama base URL for judge calls."""

DEFAULT_JUDGE_TIMEOUT_S: float = 30.0
"""The default per-call timeout for judge requests, in seconds."""

# The default score returned when the judge's response is
# malformed (no ``<SCORE>`` tag, no parseable integer, empty
# string). The contract (VAL-BENCH-007) pins this at 5.0; a
# future edit that wants a different default should add a
# contract assertion and update this constant in lockstep.
DEFAULT_FALLBACK_SCORE: float = 5.0
"""The score returned when the judge's response cannot be parsed."""

# The judge prompt template. The scorer formats the user's
# original prompt and the model's response into the template
# and sends the result to the judge model. The template asks
# the judge to evaluate on three criteria (correctness,
# completeness, clarity) and to emit a structured ``<SCORE> N
# </SCORE>`` tag on the last line of its reply. The structured
# output is the contract-pinned shape the regex parser
# expects; a judge that drifts from the shape is parsed with
# the bare-integer fallback or, failing that, returns
# :data:`DEFAULT_FALLBACK_SCORE`.
JUDGE_PROMPT_TEMPLATE: str = (
    "You are an expert code-review judge. Evaluate the following "
    "model response on three criteria:\n"
    "\n"
    "1. Correctness — does the explanation match what the code does?\n"
    "2. Completeness — does it cover the salient points (no critical "
    "omissions)?\n"
    "3. Clarity — is it well-organized and easy to follow?\n"
    "\n"
    "After your evaluation, output exactly one structured score on "
    "the last line of your response in this exact format:\n"
    "\n"
    "  <SCORE> N </SCORE>\n"
    "\n"
    "where N is an integer in [0, 10]. Do not include any other text "
    "after the SCORE tag.\n"
    "\n"
    "--- ORIGINAL PROMPT ---\n"
    "{prompt}\n"
    "--- END ORIGINAL PROMPT ---\n"
    "\n"
    "--- MODEL RESPONSE ---\n"
    "{model_output}\n"
    "--- END MODEL RESPONSE ---\n"
)
"""The template the scorer formats the user's prompt and the model's response into."""


# The regex that parses the structured ``<SCORE> N </SCORE>`` or
# ``<score>N</score>`` tag from the judge's response. The
# pattern is case-insensitive on the tag name and tolerates
# arbitrary whitespace inside the tag (e.g. ``<SCORE>  7 </SCORE>``).
# The first capture group is the integer score. When the
# structured tag is absent, the parser falls back to a bare
# integer on a line by itself.
_SCORE_TAG_REGEX: re.Pattern[str] = re.compile(
    r"<\s*[Ss][Cc][Oo][Rr][Ee]\s*>\s*(-?\d+)\s*<\s*/\s*[Ss][Cc][Oo][Rr][Ee]\s*>",
)
"""Regex that extracts the integer from ``<SCORE> N </SCORE>`` or ``<score>N</score>``."""

# The regex that parses a bare integer line. The pattern is
# anchored to the start of a line (MULTILINE) and tolerates
# surrounding whitespace. The fallback only fires when the
# structured tag is absent.
_BARE_INTEGER_REGEX: re.Pattern[str] = re.compile(r"^\s*(-?\d+)\s*$", re.MULTILINE)
"""Regex that extracts a bare integer on a line by itself."""


def parse_judge_score(
    judge_response: str,
    *,
    fallback: float = DEFAULT_FALLBACK_SCORE,
) -> float:
    """Parse a judge response and return the integer score as a float.

    The parser is robust to the common formatting variations
    the judge model emits. It tries three strategies, in order:

    1. The structured ``<SCORE> N </SCORE>`` or
       ``<score>N</score>`` tag (case-insensitive, whitespace-
       tolerant).
    2. A bare integer on a line by itself (anchored, MULTILINE).
    3. The :data:`DEFAULT_FALLBACK_SCORE` (``5.0``) when neither
       strategy matches.

    The parsed integer is clamped to ``[0, 10]`` so a judge
    that emits ``-1`` or ``42`` does not produce an out-of-
    range score. The clamp is intentionally permissive: a
    judge that emits ``7`` is a valid ``7``; a judge that
    emits ``7.5`` (a float, not an integer) is parsed by the
    bare-integer path only if it appears as the entire line,
    but the integer parser (``\\d+``) rejects ``7.5`` (the
    regex does not match a decimal point). A judge that
    emits a decimal therefore falls through to the fallback.

    Args:
        judge_response: The raw text the judge model returned.
            May be empty; the function returns the fallback in
            that case.
        fallback: The score returned when the parser cannot
            find a structured tag or a bare integer. Defaults
            to :data:`DEFAULT_FALLBACK_SCORE` (``5.0``). The
            parameter is exposed so a caller can override the
            fallback (e.g. for tests that want a different
            "no parse" sentinel).

    Returns:
        A float in ``[0, 10]``. The clamp is applied to the
        parsed integer (or to the fallback, which is already
        in range).
    """
    if not judge_response or not judge_response.strip():
        return _clamp_score(fallback)
    # Strategy 1: the structured SCORE tag. The regex is
    # case-insensitive on the tag name (handled inline) and
    # tolerates whitespace inside the tag. ``search`` is
    # correct here — the tag may appear anywhere in the
    # judge's response.
    tag_match = _SCORE_TAG_REGEX.search(judge_response)
    if tag_match is not None:
        return _clamp_score(float(int(tag_match.group(1))))
    # Strategy 2: a bare integer on a line by itself. The
    # contract-pinned fallback path. ``search`` again — the
    # parser looks for any line that is a bare integer.
    bare_match = _BARE_INTEGER_REGEX.search(judge_response)
    if bare_match is not None:
        return _clamp_score(float(int(bare_match.group(1))))
    # Strategy 3: the fallback. The judge emitted something
    # unparseable (e.g. a verbose explanation without a
    # structured tag). The contract (VAL-BENCH-007) pins the
    # fallback at 5.0; a single bad judge call must not
    # break the benchmark.
    return _clamp_score(fallback)


def _clamp_score(score: float) -> float:
    """Clamp a parsed score into the ``[0, 10]`` interval.

    The clamp is intentionally permissive: a judge that
    emits ``-1`` is clamped to ``0.0``; a judge that emits
    ``42`` is clamped to ``10.0``. The clamp guards against
    out-of-range parses without surfacing a hard error to
    the benchmark — a single bad judge call must not break
    the run.

    Args:
        score: The score to clamp. May be a float or an
            integer (the cast is safe for both).

    Returns:
        ``max(0.0, min(10.0, score))``. Floats are preserved
        as floats; integers are returned as floats for a
        stable type contract.
    """
    if score < 0.0:
        return 0.0
    if score > 10.0:
        return 10.0
    return float(score)


class LLMJudgeScorer:
    """LLM-as-judge scorer for the ``explain`` category.

    The scorer formats the user's prompt and the model's
    response into the :data:`JUDGE_PROMPT_TEMPLATE`, sends the
    result to the judge model via an :class:`httpx.AsyncClient`,
    parses the judge's response with :func:`parse_judge_score`,
    and returns the parsed float in ``[0, 10]``.

    The scorer is async: ``await scorer.score(prompt,
    model_output)``. The contract (the feature description)
    pins the async signature; the report generator and the
    live benchmark CLI both ``await`` the scorer.

    The judge model is configurable: the constructor accepts
    a ``judge_model`` argument that overrides the default
    (``deepseek-v4-pro:cloud``). The hermetic tests use the
    default model name with a fake transport; the live
    benchmark uses the default model against the real local
    Ollama. A future edit that wants a different default
    judge model should add a contract assertion and update
    :data:`DEFAULT_JUDGE_MODEL` in lockstep.

    The constructor also accepts a ``base_url`` and ``timeout``
    for the judge client, and a ``_transport`` for hermetic
    tests (mirroring the
    :class:`moaxy.adapters.openrouter.OpenRouterAdapter` and
    :class:`moaxy.adapters.ollama.OllamaAdapter` test hooks).
    The transport is ``None`` in production, which makes the
    :class:`httpx.AsyncClient` use a real network client.

    Attributes:
        judge_model: The name of the judge model (e.g.
            ``"deepseek-v4-pro:cloud"``). The scorer sends
            this as the ``model`` field of the chat-completion
            request.
        base_url: The base URL of the local Ollama API.
            Defaults to ``http://127.0.0.1:11434``.
        timeout: The per-call timeout in seconds. Defaults to
            30 seconds (Ollama can be slow on the first
            request to a cold model).
        default_fallback_score: The score returned when the
            judge's response cannot be parsed. Defaults to
            :data:`DEFAULT_FALLBACK_SCORE` (``5.0``). The
            parameter is exposed so tests can pin a different
            fallback (the production benchmark keeps the
            default).
    """

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        *,
        base_url: str = DEFAULT_JUDGE_BASE_URL,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_S,
        default_fallback_score: float = DEFAULT_FALLBACK_SCORE,
        prompt_template: str = JUDGE_PROMPT_TEMPLATE,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.judge_model = judge_model
        self.base_url = base_url.rstrip("/") if base_url else DEFAULT_JUDGE_BASE_URL
        self.timeout = float(timeout)
        self.default_fallback_score = float(default_fallback_score)
        self.prompt_template = prompt_template
        self._transport = _transport
        self._client: httpx.AsyncClient | None = None

    @property
    def endpoint(self) -> str:
        """The full URL of the ``/v1/chat/completions`` endpoint."""
        return f"{self.base_url}/v1/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        """Return the lazily-initialised ``httpx.AsyncClient``."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` and release its socket."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_messages(
        self, prompt: str, model_output: str
    ) -> list[dict[str, Any]]:
        """Build the chat-completion messages list for the judge call.

        The judge is asked to evaluate the model's response on
        three criteria; the user message is the formatted
        :data:`JUDGE_PROMPT_TEMPLATE`. No system message is
        sent — the prompt template is self-contained.

        Args:
            prompt: The user's original prompt (the coding
                task the model was asked to solve).
            model_output: The raw text the model returned.

        Returns:
            A list of OpenAI-shaped messages, ready to be
            passed to the ``messages`` field of the
            chat-completion payload.
        """
        user_text = self.prompt_template.format(
            prompt=prompt,
            model_output=model_output,
        )
        return [{"role": "user", "content": user_text}]

    async def _call_judge(self, messages: list[dict[str, Any]]) -> str:
        """Send the judge call and return the assistant's content.

        The judge is called with ``stream=False`` and a small
        ``max_tokens`` budget (the judge is asked to emit a
        short evaluation and a structured score tag; 512
        tokens is plenty). The function returns the
        ``choices[0].message.content`` of the response, or
        an empty string when the response is malformed.

        Network failures (timeouts, connection errors) are
        caught and logged; the function returns an empty
        string in that case. The caller
        (:meth:`score`) treats an empty string as a
        malformed response and returns
        :attr:`default_fallback_score`. The
        contract (VAL-BENCH-007) requires the scorer to
        be robust to any kind of judge failure, including
        network failures.

        Args:
            messages: The messages to send to the judge.

        Returns:
            The raw ``content`` of the judge's response
            message, or an empty string on any failure.
        """
        payload: dict[str, Any] = {
            "model": self.judge_model,
            "messages": messages,
            "stream": False,
            "max_tokens": 512,
        }
        client = self._get_client()
        try:
            response = await client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning(
                "LLMJudgeScorer timeout for model=%s: %s", self.judge_model, exc
            )
            return ""
        except httpx.ConnectError as exc:
            logger.warning(
                "LLMJudgeScorer connect error for model=%s: %s",
                self.judge_model,
                exc,
            )
            return ""
        except httpx.RequestError as exc:
            logger.warning(
                "LLMJudgeScorer request error for model=%s: %s",
                self.judge_model,
                exc,
            )
            return ""
        if response.status_code >= 400:
            logger.warning(
                "LLMJudgeScorer upstream %d for model=%s",
                response.status_code,
                self.judge_model,
            )
            return ""
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLMJudgeScorer failed to decode response for model=%s: %s",
                self.judge_model,
                exc,
            )
            return ""
        choices = data.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content")
        if content is None:
            return ""
        return str(content)

    async def score(self, prompt: str, model_output: str) -> float:
        """Score ``model_output`` against ``prompt`` using the judge model.

        The function is the scorer's public surface. It:

        1. Builds the judge messages with :meth:`_build_messages`.
        2. Calls the judge model with :meth:`_call_judge`.
        3. Parses the judge's response with
           :func:`parse_judge_score`, falling back to
           :attr:`default_fallback_score` on any failure.

        The function never raises: a judge timeout,
        connection error, malformed response, or unparseable
        score all degrade gracefully to
        :attr:`default_fallback_score`. The contract
        (VAL-BENCH-007) requires the scorer to be robust to
        any kind of judge failure; the report generator and
        the live benchmark CLI rely on this property.

        Args:
            prompt: The user's original prompt (the coding
                task the model was asked to solve).
            model_output: The raw text the model returned.

        Returns:
            A float in ``[0, 10]``. When the judge's
            response is well-formed, the value is the parsed
            integer (clamped to ``[0, 10]``). When the
            response is malformed, the value is
            :attr:`default_fallback_score` (``5.0`` by
            default).
        """
        messages = self._build_messages(prompt, model_output)
        judge_response = await self._call_judge(messages)
        return parse_judge_score(
            judge_response, fallback=self.default_fallback_score
        )


# A type alias for the ``score`` callable. Exposed so callers
# (the harness, the report generator, the live benchmark CLI)
# can type-annotate a "scorer" parameter without binding to
# the :class:`LLMJudgeScorer` class directly. The alias is a
# :class:`Callable` rather than the method itself so the
# scorer is reusable across multiple ``score()`` calls.
JudgeScoreCallable = Callable[[str, str], Awaitable[float]]
"""Async callable that scores a (prompt, model_output) pair and returns a float in [0, 10]."""


# A function-form factory: returns a coroutine that scores
# ``(prompt, model_output)`` using a default-configured
# :class:`LLMJudgeScorer`. The factory is the convenience
# surface for the report generator and the live benchmark CLI,
# which want a one-line "judge this" hook without managing a
# scorer instance. The factory is the canonical
# ``LLMJudgeScorer.score`` equivalent for callers that do not
# need to configure the scorer.
async def score(prompt: str, model_output: str) -> float:
    """Score ``model_output`` against ``prompt`` with a default scorer.

    The function is a thin wrapper around
    :meth:`LLMJudgeScorer.score` that constructs a default
    scorer, calls it, and closes the scorer's underlying
    :class:`httpx.AsyncClient` after the call. Use this
    when the caller does not need to keep the scorer alive
    (e.g. a one-shot CLI invocation). Use a long-lived
    :class:`LLMJudgeScorer` instance when the caller wants
    to amortise the client setup across many calls.

    The function is the contract-pinned surface for the
    ``score(prompt, model_output)`` callable referenced in
    the feature description. The benchmark harness imports
    this name from :mod:`moaxy.benchmark.scoring.judge`.

    Args:
        prompt: The user's original prompt.
        model_output: The raw text the model returned.

    Returns:
        A float in ``[0, 10]``. See :meth:`LLMJudgeScorer.score`
        for the parsing and fallback semantics.
    """
    scorer = LLMJudgeScorer()
    try:
        return await scorer.score(prompt, model_output)
    finally:
        await scorer.close()


__all__ = [
    "DEFAULT_FALLBACK_SCORE",
    "DEFAULT_JUDGE_BASE_URL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_TIMEOUT_S",
    "JUDGE_PROMPT_TEMPLATE",
    "LLMJudgeScorer",
    "parse_judge_score",
    "score",
]
