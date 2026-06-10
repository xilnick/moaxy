"""Advisor stage for the moaxy pipeline.

The advisor runs a single pass after the reflection loop. It calls a
second model (the "advisor") over the post-reflection answer and
interprets the response to either approve the previous answer or
produce a revised one. When the advisor revises, the orchestrator
follows up with a primary-model call to materialise the final
revised answer (see :func:`moaxy.pipeline.orchestrator.Orchestrator.run`).

Algorithm
---------

1. Build the advisor message list with
   :func:`moaxy.pipeline.message_builders.build_advisor_messages`.
2. Call the configured adapter with the requested advisor model.
3. Parse the assistant content with :func:`parse_advisor_response`:
   the parser returns a ``(decision, text)`` tuple where
   ``decision`` is one of ``"approve"`` or ``"revise"`` and ``text``
   is the revised answer (or ``None`` on approve).
4. Run the configured :class:`moaxy.plugins.types.PluginType.ADVISOR`
   plugins via :meth:`moaxy.plugins.manager.PluginManager.run`. The
   plugin context receives ``advisor_decision`` and
   ``advisor_text`` (the full assistant content) so plugins can
   inspect or transform the advisor's verdict.

Self-advise
-----------

When ``advisor_model`` equals the primary model (self-advise), the
function still makes a separate advisor LLM call. Self-advise is
useful for double-checking a model's own answer from a fresh prompt
context. The two LLM calls are observable as distinct events in the
pipeline log.
"""

from __future__ import annotations

import logging
from typing import Any

from moaxy.adapters.base import Adapter
from moaxy.pipeline.message_builders import build_advisor_messages
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)

_APPROVE_MARKER = "ADVISOR_APPROVE"
_REVISE_MARKER = "ADVISOR_REVISE:"


def parse_advisor_response(text: str | None) -> tuple[str, str | None]:
    """Return ``(decision, revised_text)`` for the advisor's reply.

    The helper inspects the assistant content returned by the advisor
    LLM call. It supports two markers:

    * ``ADVISOR_APPROVE`` — anywhere in the text. The decision is
      ``"approve"`` and ``revised_text`` is ``None``. Extra text
      alongside the marker is ignored.
    * ``ADVISOR_REVISE: <text>`` — the marker followed by the revised
      answer. The decision is ``"revise"`` and ``revised_text`` is
      the trimmed text after the marker.

    When neither marker is present, the helper defaults to
    ``("revise", text)`` so the orchestrator treats the entire
    reply as the revised answer. This is the conservative
    fallback: a model that "intended" to revise but forgot the
    prefix still feeds a revision to the primary model.

    Args:
        text: The assistant content returned by the advisor. May
            be ``None`` or a non-string; in that case the helper
            returns ``("revise", "")`` rather than raising.

    Returns:
        A ``(decision, revised_text)`` tuple. ``decision`` is the
        string ``"approve"`` or ``"revise"``; ``revised_text`` is
        ``None`` on approve, the trimmed revised text on revise,
        or the original ``text`` when no marker is present.
    """
    if not isinstance(text, str) or not text:
        return "revise", text if isinstance(text, str) else ""

    approve_idx = text.find(_APPROVE_MARKER)
    if approve_idx >= 0:
        return "approve", None

    revise_idx = text.find(_REVISE_MARKER)
    if revise_idx >= 0:
        after = text[revise_idx + len(_REVISE_MARKER):]
        return "revise", after.strip()

    return "revise", text.strip()


async def advisor_turn(
    ctx: dict[str, Any],
    advisor_model: str,
    history: list[dict[str, Any]],
    current_answer: str,
    *,
    system_prompt: str | None = None,
) -> tuple[str, str | None]:
    """Run one advisor pass and return ``(response, decision, revised_text)``.

    The function builds the advisor message list with
    :func:`moaxy.pipeline.message_builders.build_advisor_messages`,
    calls ``ctx["adapter"].chat(model=advisor_model, messages=...)``
    once, parses the response with :func:`parse_advisor_response`,
    and runs the configured
    :class:`moaxy.plugins.types.PluginType.ADVISOR` plugins through
    :meth:`moaxy.plugins.manager.PluginManager.run` (when a plugin
    manager is present in the context). The plugin context receives
    ``advisor_decision`` (``"approve"`` or ``"revise"``) and
    ``advisor_text`` (the full assistant content) so plugins can
    inspect the advisor's verdict.

    Args:
        ctx: A dict carrying the per-request state. The advisor reads
            ``ctx["adapter"]`` (a
            :class:`moaxy.adapters.base.Adapter` instance, required)
            and ``ctx["plugin_manager"]`` (a
            :class:`moaxy.plugins.manager.PluginManager` instance,
            optional — when missing, plugin dispatch is skipped).
            Any other fields are passed through to the plugin
            manager unchanged.
        advisor_model: The model identifier to pass to the adapter.
            The orchestrator resolves aliases before calling
            advisor_turn, so the value here is the real model name
            the adapter understands.
        history: The conversation history to include in the advisor
            message list. The list is not mutated; the builder deep
            copies it before forwarding.
        current_answer: The post-reflection answer to review. The
            builder embeds it verbatim in the trailing user-role
            message.
        system_prompt: Optional advisor system prompt. When
            ``None`` or empty, no system message is prepended; the
            default used by the orchestrator is
            :data:`moaxy.pipeline.prompts.DEFAULT_ADVISOR_PROMPT`.

    Returns:
        A ``(decision, revised_text)`` tuple. ``decision`` is the
        string ``"approve"`` or ``"revise"``. ``revised_text`` is
        ``None`` on approve; on revise it is the trimmed text
        returned by :func:`parse_advisor_response` (the entire
        assistant content when the ``ADVISOR_REVISE:`` marker is
        missing).

    Raises:
        KeyError: ``ctx`` does not contain an ``"adapter"`` key.
        UpstreamError: The adapter returned a 4xx/5xx or its
            response could not be decoded. Errors bubble up
            unchanged so the orchestrator's fallback walker can
            retry / advance.
    """
    adapter: Adapter = ctx["adapter"]
    plugin_manager: PluginManager | None = ctx.get("plugin_manager")

    messages = build_advisor_messages(
        history=history,
        answer=current_answer,
        system_prompt=system_prompt,
    )
    response = await adapter.chat(model=advisor_model, messages=messages)
    text = response.message.content
    decision, revised_text = parse_advisor_response(text)

    logger.debug(
        "advisor_turn: model=%s decision=%s len(text)=%d",
        advisor_model,
        decision,
        len(text),
    )

    # Expose the parsed verdict on the context so ADVISOR plugins can read it.
    ctx["advisor_decision"] = decision
    ctx["advisor_text"] = text
    ctx["advisor_revised_text"] = revised_text
    ctx["advisor_model"] = advisor_model
    ctx["advisor_response"] = response

    if plugin_manager is not None:
        await plugin_manager.run(ctx, plugin_types=[PluginType.ADVISOR])

    return decision, revised_text


__all__ = ["advisor_turn", "parse_advisor_response"]
