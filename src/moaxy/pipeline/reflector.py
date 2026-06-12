"""Self-reflection step for the moaxy pipeline.

The reflection stage runs once per configured turn. For each turn the
reflector:

1. Builds the critique message list with
   :func:`moaxy.pipeline.message_builders.build_reflection_messages`.
2. Calls the configured adapter with the requested model.
3. Parses the model's confidence from the response text using
   :func:`parse_confidence` (regex
   ``^REFLECT_CONFIDENCE:\\s*([0-9.]+)\\s*$``). The contract pins the
   exact regex (VAL-PIPE-010); see :data:`_CONFIDENCE_RE`.
4. Parses the integer score from the response text using
   :func:`parse_score` (regex ``^SCORE:\\s*(\\d+)\\s*$``,
   VAL-PIPE-EXTRA-039).
5. Computes the weighted early-exit signal with
   :func:`parse_weighted_signal` (VAL-PIPE-EXTRA-042), the
   ``trust_verbal * confidence + trust_score * (score / 10)``
   formula. The "score missing" path falls back to ``confidence``
   (v1 invariant).
6. Runs the configured ``REFLECTOR`` plugins via
   :meth:`moaxy.plugins.manager.PluginManager.run` so user-supplied
   plugins can observe (or transform) the critique.
7. Returns ``(critique_text, confidence)`` so the orchestrator can
   decide whether to short-circuit (early-exit) or call the revision
   step.

The function deliberately makes a single LLM call against a single
model. The orchestrator (M2+) wraps :func:`reflect_turn` in
:func:`moaxy.pipeline.fallback.call_with_fallbacks` so the reflection
step inherits the same retry/fallback policy as the initial generation.
Splitting "make a critique LLM call" from "walk the fallback list"
keeps the reflector easy to test in isolation and the fallback walker
backend-agnostic.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from moaxy.adapters.base import Adapter
from moaxy.pipeline.message_builders import build_reflection_messages
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)

# Anchored regex that extracts the float from a critique's last line.
# The contract pins the exact pattern
# ``^REFLECT_CONFIDENCE:\\s*([0-9.]+)\\s*$`` (VAL-PIPE-010); the
# ``re.MULTILINE`` flag makes ``^`` and ``$`` match line boundaries
# inside multi-line critique text. ``[0-9.]+`` allows both
# ``"0.92"`` and ``"1.0"`` (and integer forms like ``"0"``) while
# rejecting non-numeric suffixes; the float conversion in
# :func:`parse_confidence` rejects ``"...."`` (which matches the
# pattern but is not a valid float).
_CONFIDENCE_RE: re.Pattern[str] = re.compile(
    r"^REFLECT_CONFIDENCE:\s*([0-9.]+)\s*$",
    re.MULTILINE,
)

# Anchored regex that extracts the integer from a critique's last
# ``SCORE:`` line. The contract pins the exact pattern
# ``^SCORE:\\s*(\\d+)\\s*$`` (VAL-PIPE-EXTRA-039, DELTA 5/6). It is
# integer-only by design: scores are 0..10 and the model is expected
# to emit integers; non-integer or non-numeric forms (e.g.
# ``"SCORE: 7.5"``, ``"SCORE: seven"``) are rejected.
_SCORE_RE: re.Pattern[str] = re.compile(
    r"^SCORE:\s*(\d+)\s*$",
    re.MULTILINE,
)


def parse_confidence(text: str | None) -> float:
    """Return the float from the last ``REFLECT_CONFIDENCE: <float>`` line.

    The helper uses the regex
    ``^REFLECT_CONFIDENCE:\\s*([0-9.]+)\\s*$`` (case-sensitive,
    anchored per line). When the line is missing, the value is not a
    valid float, or ``text`` is not a string, the helper returns
    ``0.0`` so the orchestrator can treat the missing-confidence case
    as "no early exit".

    Args:
        text: The critique text returned by the model. May be ``None``
            or a non-string; in that case the helper returns ``0.0``
            rather than raising. The contract only passes strings.

    Returns:
        The float parsed from the *last* matching line in ``text``,
        or ``0.0`` when the regex finds nothing or the matched value
        is not a valid float.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    matches = _CONFIDENCE_RE.findall(text)
    if not matches:
        return 0.0
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return 0.0


def parse_score(text: str | None) -> int | None:
    """Return the integer from the last ``SCORE: <int>`` line.

    The helper uses the regex ``^SCORE:\\s*(\\d+)\\s*$`` (anchored
    per line, MULTILINE, integer-only). When the line is missing or
    the value is not a valid integer, the helper returns ``None``
    so the orchestrator can distinguish a missing-score case from a
    parsed-but-malformed case.

    Args:
        text: The critique text returned by the model. May be
            ``None`` or a non-string; in that case the helper
            returns ``None`` rather than raising. The contract only
            passes strings.

    Returns:
        The integer parsed from the *last* matching line in
        ``text``, or ``None`` when the regex finds nothing. The
        integer-only constraint rejects ``"SCORE: 7.5"`` and
        ``"SCORE: seven"``. Out-of-range values (e.g. ``"SCORE: 11"``)
        are recorded as-is; the parser does not clamp.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


def parse_weighted_signal(
    text: str | None,
    *,
    trust_verbal: float,
    trust_score: float,
) -> tuple[float, float, int | None]:
    """Return ``(combined, confidence, score)`` for the weighted early-exit check.

    The function extracts the last ``REFLECT_CONFIDENCE:`` and
    ``SCORE:`` values from ``text`` (using the same parsers the
    orchestrator calls directly), then computes the combined
    early-exit signal:

    * When a ``SCORE:`` value is present:
      ``combined = trust_verbal * confidence + trust_score * (score / 10.0)``.
    * When ``SCORE:`` is missing: ``combined = confidence`` (the
      v1 behavior). The "score missing" path does NOT multiply
      confidence by ``trust_verbal``; the v1 threshold check
      ``confidence >= threshold`` is preserved verbatim. This is
      DELTA 5's documented exception (VAL-PIPE-EXTRA-035,
      VAL-PIPE-EXTRA-042).

    Args:
        text: The critique text returned by the model. May be
            ``None`` or a non-string; the helper handles it
            defensively.
        trust_verbal: Weight applied to the verbal ``REFLECT_CONFIDENCE:``
            signal. Caller must supply a non-negative float.
        trust_score: Weight applied to the integer ``SCORE:`` signal
            (after dividing by 10). Caller must supply a
            non-negative float.

    Returns:
        A 3-tuple ``(combined, confidence, score)``.

        * ``combined`` is the early-exit signal the orchestrator
          compares against the route's ``threshold``. ``confidence``
          when ``score`` is ``None``; the weighted formula above
          when both are present.
        * ``confidence`` is the float parsed from
          ``REFLECT_CONFIDENCE:`` (or ``0.0`` when the line is
          missing or malformed).
        * ``score`` is the integer parsed from ``SCORE:`` (or
          ``None`` when the line is missing or malformed).
    """
    confidence = parse_confidence(text)
    score = parse_score(text)
    if score is None:
        return confidence, confidence, score
    combined = trust_verbal * confidence + trust_score * (score / 10.0)
    return combined, confidence, score


async def reflect_turn(
    ctx: dict[str, Any],
    model: str,
    history: list[dict[str, Any]],
    current_answer: str,
    *,
    system_prompt: str | None = None,
    fresh_context: bool = False,
) -> tuple[str, float]:
    """Run one self-reflection turn and return ``(critique_text, confidence)``.

    The function builds the critique message list with
    :func:`moaxy.pipeline.message_builders.build_reflection_messages`,
    calls ``ctx["adapter"].chat(model=model, messages=...)`` once, and
    parses the confidence off the response. The configured
    ``REFLECTOR`` plugins are then invoked through
    :meth:`moaxy.plugins.manager.PluginManager.run` (when a plugin
    manager is present in the context); the plugin context receives
    the parsed ``critique_text`` and ``confidence`` so plugins can
    inspect the model's self-assessment.

    Args:
        ctx: A dict carrying the per-request state. The reflector
            reads ``ctx["adapter"]`` (a :class:`moaxy.adapters.base.Adapter`
            instance, required) and ``ctx["plugin_manager"]`` (a
            :class:`moaxy.plugins.manager.PluginManager` instance,
            optional — when missing, plugin dispatch is skipped).
            Any other fields are passed through to the plugin manager
            unchanged so plugins can read request metadata (request
            id, route, etc.).
        model: The model identifier to pass to the adapter. The
            orchestrator resolves aliases before calling reflect_turn,
            so the value here is the real model name the adapter
            understands.
        history: The conversation history to include in the critique
            message list. The list is not mutated; the builder deep
            copies it before forwarding. *Ignored* when
            ``fresh_context: true`` — the M8 "type 2 reflection"
            contract is that the critique is built from the candidate
            answer and a cold-grading rubric only.
        current_answer: The model's previous answer to critique. The
            builder embeds it verbatim in the trailing user-role
            message.
        system_prompt: Optional reflector system prompt. When
            ``None`` or empty, no system message is prepended; the
            default used by the orchestrator is
            :data:`moaxy.pipeline.prompts.DEFAULT_REFLECT_PROMPT`.
            *Ignored* when ``fresh_context: true`` — the
            :data:`_FRESH_CONTEXT_RUBRIC` is used instead so the
            critique is genuinely isolated from the original
            request context.
        fresh_context: M8 "type 2 reflection" toggle. When ``True``,
            the critique message list excludes the client's system
            prompt, the chat history, and the user request. The list
            contains only a cold-grading rubric (system) and the
            candidate answer (user). When ``False`` (default), the
            existing M1-M7 prompt construction is used unchanged.

    Returns:
        A ``(critique_text, confidence)`` tuple. ``critique_text`` is
        the full assistant content returned by the adapter (the
        entire critique, including the ``REFLECT_CONFIDENCE:`` line
        when present). ``confidence`` is the float parsed by
        :func:`parse_confidence` (``0.0`` when the line is missing
        or malformed).

    Raises:
        KeyError: ``ctx`` does not contain an ``"adapter"`` key.
        UpstreamError: The adapter returned a 4xx/5xx or its response
            could not be decoded. Errors bubble up unchanged so the
            orchestrator's fallback walker can retry / advance.
    """
    adapter: Adapter = ctx["adapter"]
    plugin_manager: PluginManager | None = ctx.get("plugin_manager")

    messages = build_reflection_messages(
        history=history,
        answer=current_answer,
        system_prompt=system_prompt,
        fresh_context=fresh_context,
    )
    response = await adapter.chat(model=model, messages=messages)
    text = response.message.content
    confidence = parse_confidence(text)

    logger.debug(
        "reflect_turn: model=%s confidence=%s len(text)=%d",
        model,
        confidence,
        len(text),
    )

    # Expose the result on the context so REFLECTOR plugins can read it.
    ctx["critique_text"] = text
    ctx["confidence"] = confidence
    ctx["reflect_model"] = model

    if plugin_manager is not None:
        await plugin_manager.run(ctx, plugin_types=[PluginType.REFLECTOR])

    return text, confidence


__all__ = [
    "parse_confidence",
    "parse_score",
    "parse_weighted_signal",
    "reflect_turn",
]
