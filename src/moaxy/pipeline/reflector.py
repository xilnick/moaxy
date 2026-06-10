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
4. Runs the configured ``REFLECTOR`` plugins via
   :meth:`moaxy.plugins.manager.PluginManager.run` so user-supplied
   plugins can observe (or transform) the critique.
5. Returns ``(critique_text, confidence)`` so the orchestrator can
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


async def reflect_turn(
    ctx: dict[str, Any],
    model: str,
    history: list[dict[str, Any]],
    current_answer: str,
    *,
    system_prompt: str | None = None,
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
            copies it before forwarding.
        current_answer: The model's previous answer to critique. The
            builder embeds it verbatim in the trailing user-role
            message.
        system_prompt: Optional reflector system prompt. When
            ``None`` or empty, no system message is prepended; the
            default used by the orchestrator is
            :data:`moaxy.pipeline.prompts.DEFAULT_REFLECT_PROMPT`.

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


__all__ = ["parse_confidence", "reflect_turn"]
