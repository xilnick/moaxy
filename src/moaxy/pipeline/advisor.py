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
   the parser returns a ``(decision, revised_text, score, issues)``
   4-tuple. ``decision`` is one of ``"approve"`` or ``"revise"``;
   ``revised_text`` is the revised answer (or ``None`` on
   approve); ``score`` is the parsed integer from the last
   ``ADVISOR_SCORE: <int>`` line (``None`` when missing); and
   ``issues`` is the bulleted list extracted from the
   ``ADVISOR_ISSUES:`` block (``[]`` when missing).
4. The cross-critique fields (decision / score / issues) are
   extracted additively on top of the legacy
   ``ADVISOR_APPROVE`` / ``ADVISOR_REVISE:`` substring parsers.
   When the new ``ADVISOR_DECISION: APPROVE|REVISE`` line is
   present it takes precedence over the legacy
   ``ADVISOR_APPROVE`` substring; otherwise the legacy parser
   continues to be the source of truth for the decision. The
   score and issues are always extracted (independent of the
   decision path) so callers see the full cross-critique
   context regardless of which decision marker the model used.
5. Run the configured :class:`moaxy.plugins.types.PluginType.ADVISOR`
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
import re
from typing import Any

from moaxy.adapters.base import Adapter
from moaxy.pipeline.message_builders import build_advisor_messages
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)

_APPROVE_MARKER = "ADVISOR_APPROVE"
_REVISE_MARKER = "ADVISOR_REVISE:"

# Anchored regex that extracts the integer from an advisor's last
# ``ADVISOR_SCORE:`` line. The contract pins the exact pattern
# ``^ADVISOR_SCORE:\\s*(\\d+)\\s*$`` (VAL-PIPE-EXTRA-040). Integer-
# only by design; non-integer or non-numeric forms are rejected.
_ADVISOR_SCORE_RE: re.Pattern[str] = re.compile(
    r"^ADVISOR_SCORE:\s*(\d+)\s*$",
    re.MULTILINE,
)

# Anchored regex that captures the leading ``ADVISOR_ISSUES:`` line
# (case-sensitive, anchored at line start). The bullet bodies are
# extracted line-by-line in :func:`parse_advisor_issues` because the
# regex needs to tolerate multiple bullet markers and arbitrary
# whitespace.
_ADVISOR_ISSUES_HEADER_RE: re.Pattern[str] = re.compile(
    r"^ADVISOR_ISSUES:\s*$",
    re.MULTILINE,
)

# Anchored regex that captures the cross-critique decision line
# ``ADVISOR_DECISION: APPROVE`` or ``ADVISOR_DECISION: REVISE``
# (case-sensitive, anchored per line, MULTILINE). The captured
# group is the decision keyword, trimmed and uppercased. When
# the line is missing the regex returns ``None`` and the parser
# falls back to the legacy ``ADVISOR_APPROVE`` / ``ADVISOR_REVISE:``
# substring parsers for the decision.
_ADVISOR_DECISION_RE: re.Pattern[str] = re.compile(
    r"^ADVISOR_DECISION:\s*(APPROVE|REVISE)\s*$",
    re.MULTILINE,
)


def parse_advisor_response(
    text: str | None,
) -> tuple[str, str | None, int | None, list[str]]:
    """Return ``(decision, revised_text, score, issues)`` for the advisor's reply.

    The helper inspects the assistant content returned by the
    advisor LLM call and returns a 4-tuple covering the legacy
    decision path AND the cross-critique fields. The four
    elements are:

    * ``decision`` — the string ``"approve"`` or ``"revise"``.
    * ``revised_text`` — ``None`` on approve; the trimmed
      revised text on revise; or the entire input ``text`` when
      no marker is present (the conservative fallback for a
      model that intended to revise but forgot the prefix).
    * ``score`` — the integer parsed from the last
      ``ADVISOR_SCORE: <int>`` line, or ``None`` when the line
      is missing / malformed.
    * ``issues`` — the bulleted list extracted from the
      ``ADVISOR_ISSUES:`` block, or ``[]`` when the block is
      missing or empty.

    Decision resolution
    -------------------

    The parser applies a small precedence ladder so a model can
    emit either the legacy substring markers or the new
    ``ADVISOR_DECISION:`` line (or both) without ambiguity:

    1. When the ``ADVISOR_DECISION: APPROVE|REVISE`` line is
       present, the captured value decides ``decision`` and
       ``ADVISOR_REVISE: <text>`` (if present) supplies
       ``revised_text``; otherwise the whole text is the
       revised text.
    2. When no ``ADVISOR_DECISION:`` line is present but
       ``ADVISOR_APPROVE`` is a substring, the decision is
       ``"approve"`` and ``revised_text`` is ``None`` (matches
       the v1-v4 contract; substring matching is preserved for
       backward compatibility).
    3. When ``ADVISOR_REVISE: <text>`` is present (and step 1/2
       did not match), the decision is ``"revise"`` and
       ``revised_text`` is the trimmed text after the marker.
    4. Otherwise the decision is ``"revise"`` (conservative
       fallback) and ``revised_text`` is the trimmed input.

    Score and issues are always parsed from the input via
    :func:`parse_advisor_score` and :func:`parse_advisor_issues`
    regardless of which decision path was taken, so a model
    that emits the cross-critique fields alongside a legacy
    decision marker (e.g. ``ADVISOR_APPROVE\nADVISOR_SCORE: 8``)
    still gets the score surfaced.

    Args:
        text: The assistant content returned by the advisor. May
            be ``None`` or a non-string; in that case the helper
            returns ``("revise", "", None, [])`` rather than
            raising.

    Returns:
        A 4-tuple ``(decision, revised_text, score, issues)``.
        ``decision`` is one of ``"approve"`` / ``"revise"``;
        ``revised_text`` is ``None`` on approve or the trimmed
        revised text on revise; ``score`` is the integer
        ``ADVISOR_SCORE:`` value or ``None``; ``issues`` is the
        parsed bullet list (``[]`` when missing).
    """
    if not isinstance(text, str) or not text:
        return (
            "revise",
            text if isinstance(text, str) else "",
            None,
            [],
        )

    # Cross-critique fields: always extracted (additive to the
    # decision). The helpers are pure and tolerant of missing /
    # malformed input — see :func:`parse_advisor_score` and
    # :func:`parse_advisor_issues` for the exact semantics.
    score = parse_advisor_score(text)
    issues = parse_advisor_issues(text)

    # Step 1: ``ADVISOR_DECISION:`` line takes precedence.
    decision_match = _ADVISOR_DECISION_RE.search(text)
    if decision_match is not None:
        decision_value = decision_match.group(1).upper()
        if decision_value == "APPROVE":
            return "approve", None, score, issues
        # ``REVISE``: prefer the ``ADVISOR_REVISE: <text>`` marker
        # for the revised text; fall back to the whole input when
        # the marker is absent (mirrors the v1-v4 conservative
        # fallback for missing marker).
        revise_idx = text.find(_REVISE_MARKER)
        if revise_idx >= 0:
            after = text[revise_idx + len(_REVISE_MARKER):]
            return "revise", after.strip(), score, issues
        return "revise", text.strip(), score, issues

    # Step 2: legacy ``ADVISOR_APPROVE`` substring.
    approve_idx = text.find(_APPROVE_MARKER)
    if approve_idx >= 0:
        return "approve", None, score, issues

    # Step 3: legacy ``ADVISOR_REVISE: <text>`` marker.
    revise_idx = text.find(_REVISE_MARKER)
    if revise_idx >= 0:
        after = text[revise_idx + len(_REVISE_MARKER):]
        return "revise", after.strip(), score, issues

    # Step 4: conservative fallback — treat the whole input as a revise.
    return "revise", text.strip(), score, issues


def parse_advisor_score(text: str | None) -> int | None:
    """Return the integer from the last ``ADVISOR_SCORE: <int>`` line.

    The helper uses the regex
    ``^ADVISOR_SCORE:\\s*(\\d+)\\s*$`` (anchored per line,
    MULTILINE, integer-only). When the line is missing or the value
    is not a valid integer, the helper returns ``None`` so the
    orchestrator can distinguish a missing-score case from a
    parsed-but-malformed case.

    Args:
        text: The advisor's reply text. May be ``None`` or a
            non-string; in that case the helper returns ``None``
            rather than raising. The contract only passes strings.

    Returns:
        The integer parsed from the *last* matching line in
        ``text``, or ``None`` when the regex finds nothing.
        Integer-only: ``"ADVISOR_SCORE: 8.5"`` and
        ``"ADVISOR_SCORE: eight"`` return ``None``.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = _ADVISOR_SCORE_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


# Bullet markers accepted by :func:`parse_advisor_issues`. The set
# of allowed markers is closed: ``"-"`` (ASCII hyphen), ``"*"`` (ASCII
# asterisk), and ``"•"`` (Unicode bullet, U+2022). All markers must
# be followed by one or more whitespace characters (space or tab) so
# that text like ``"-"`` mid-sentence is not misread as a bullet.
_ADVISOR_ISSUES_MARKERS: tuple[str, ...] = ("-", "*", "\u2022")
_ADVISOR_ISSUES_MARKER_SET: frozenset[str] = frozenset(_ADVISOR_ISSUES_MARKERS)


def _parse_issues_block(block: str) -> list[str]:
    """Extract a list of trimmed bullet strings from a body block.

    The body is the text that follows the ``ADVISOR_ISSUES:`` header
    line, up to the next blank line or end of input. Each line that
    starts with one of ``-``, ``*``, or ``•`` followed by one or
    more whitespace characters is treated as a bullet; the bullet
    marker and the trailing whitespace are stripped and the
    remainder is trimmed. Lines that don't start with a recognised
    marker (or that are blank) are skipped. Order is preserved.

    Args:
        block: A chunk of text containing zero or more bullet
            lines. May be empty.

    Returns:
        A list of trimmed strings, one per recognised bullet line.
        Empty bullets (``-`` with nothing after) are skipped.
    """
    issues: list[str] = []
    if not block:
        return issues
    # Iterate line-by-line. ``str.splitlines()`` collapses \r\n, \r
    # and \n into a flat list and skips the final empty entry.
    for line in block.splitlines():
        # Fast-path: empty / whitespace-only lines.
        if not line.strip():
            continue
        first = line[0]
        if first not in _ADVISOR_ISSUES_MARKER_SET:
            continue
        # The character after the marker must be whitespace.
        if len(line) < 2 or line[1] not in (" ", "\t"):
            continue
        body = line[2:].strip()
        if not body:
            # Skip empty bullets.
            continue
        issues.append(body)
    return issues


def parse_advisor_issues(text: str | None) -> list[str]:
    """Parse the ``ADVISOR_ISSUES:`` block into a list of issue strings.

    The helper locates the *last* ``ADVISOR_ISSUES:`` header line in
    ``text`` (case-sensitive, anchored at the start of a line,
    MULTILINE) and extracts the bullet lines that follow it, up to
    the next blank line. Each line is accepted when it starts with
    one of ``-``, ``*``, or ``•`` (U+2022) followed by one or more
    whitespace characters. The bullet marker and trailing
    whitespace are stripped; the remainder is trimmed. Empty
    bullets are skipped; order is preserved.

    Args:
        text: The advisor's reply text. May be ``None`` or a
            non-string; in that case the helper returns ``[]``
            rather than raising. The contract only passes strings.

    Returns:
        A list of trimmed issue strings. Returns ``[]`` when the
        header is missing, the body is empty, or every bullet is
        blank. The legacy ``ADVISOR_APPROVE`` / ``ADVISOR_REVISE:``
        markers (substring parsers) are unchanged; this helper is
        additive.
    """
    if not isinstance(text, str) or not text:
        return []
    matches = list(_ADVISOR_ISSUES_HEADER_RE.finditer(text))
    if not matches:
        return []
    # Use the *last* header so a model that emits multiple
    # ``ADVISOR_ISSUES:`` blocks resolves to the final one (mirrors
    # the "last match wins" pattern used elsewhere).
    header_match = matches[-1]
    # The block starts immediately after the header line. Locate
    # the next newline (or the end of the string) and then scan
    # line-by-line until a blank line is encountered.
    start = header_match.end()
    # Advance to the start of the next line.
    nl = text.find("\n", start)
    if nl == -1:
        return []
    block = text[nl + 1:]
    # Truncate the block at the first blank line.
    blank_line = block.find("\n\n")
    if blank_line != -1:
        block = block[:blank_line]
    return _parse_issues_block(block)


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
    inspect the advisor's verdict. The M5 cross-critique fields
    (``advisor_score`` and ``advisor_issues``) are also seeded on
    the context so plugins and observers can read them.

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
        A ``(decision, revised_text)`` 2-tuple. ``decision`` is
        the string ``"approve"`` or ``"revise"``. ``revised_text``
        is ``None`` on approve; on revise it is the trimmed text
        returned by :func:`parse_advisor_response` (the entire
        assistant content when the ``ADVISOR_REVISE:`` marker is
        missing). The function does NOT return the score or the
        issues; those are surfaced on the context dict
        (``ctx["advisor_score"]`` and ``ctx["advisor_issues"]``)
        so plugins and orchestrator observers can read them
        without the function signature carrying four elements
        (the orchestrator's caller unpacks the 2-tuple). The
        full cross-critique fields remain available via
        :func:`parse_advisor_response`'s 4-tuple return.

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
    decision, revised_text, advisor_score, advisor_issues = (
        parse_advisor_response(text)
    )

    logger.debug(
        "advisor_turn: model=%s decision=%s score=%s issues=%d len(text)=%d",
        advisor_model,
        decision,
        advisor_score,
        len(advisor_issues),
        len(text),
    )

    # Expose the parsed verdict on the context so ADVISOR plugins can read it.
    ctx["advisor_decision"] = decision
    ctx["advisor_text"] = text
    ctx["advisor_revised_text"] = revised_text
    ctx["advisor_model"] = advisor_model
    ctx["advisor_response"] = response
    # M5 cross-critique fields: ``advisor_score`` and ``advisor_issues``
    # are surfaced on the context so plugins and orchestrator
    # observers can read them. The values come straight from the
    # 4-tuple return of :func:`parse_advisor_response`.
    ctx["advisor_score"] = advisor_score
    ctx["advisor_issues"] = advisor_issues

    if plugin_manager is not None:
        await plugin_manager.run(ctx, plugin_types=[PluginType.ADVISOR])

    return decision, revised_text


__all__ = [
    "advisor_turn",
    "parse_advisor_issues",
    "parse_advisor_response",
    "parse_advisor_score",
]
