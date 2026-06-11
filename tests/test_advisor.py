"""Tests for :mod:`moaxy.pipeline.advisor`.

The advisor module is the dedicated entry point for the M3 advisor
stage. It exposes two public symbols:

* :func:`parse_advisor_response` — pure helper that interprets the
  advisor's reply. It returns a ``(decision, text)`` tuple where
  ``decision`` is one of ``"approve"`` or ``"revise"`` and ``text``
  is the revised answer (or ``None`` on approve).
* :func:`advisor_turn` — coroutine that performs one advisor pass
  over the post-reflection answer. It builds the advisor message
  list with :func:`moaxy.pipeline.message_builders.build_advisor_messages`,
  dispatches one LLM call through the configured adapter, parses the
  response, and runs the configured :class:`PluginType.ADVISOR`
  plugins through :meth:`moaxy.plugins.manager.PluginManager.run`.
  The function returns ``(decision, text)`` so the orchestrator can
  decide whether to short-circuit (approve) or call the primary
  model once more with :func:`moaxy.pipeline.message_builders.build_advisor_revision_messages`.

The tests are hermetic: a hand-rolled :class:`ScriptedAdapter` records
every call and returns scripted responses, mirroring the pattern in
``tests/test_reflector.py`` and ``tests/test_orchestrator.py``. The
plugin manager is constructed in-memory and populated with a single
recorder plugin; no discovery on disk.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    Usage,
)
from moaxy.pipeline.advisor import (
    advisor_turn,
    parse_advisor_issues,
    parse_advisor_response,
    parse_advisor_score,
)
from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_advisor_revision_messages,
)
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.pipeline.prompts import DEFAULT_ADVISOR_PROMPT
from moaxy.plugins.base import Plugin
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR_FOR_TESTS = PROJECT_ROOT / "plugins"


# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter — hermetic, in-process
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    The script is a list of either:

    * a :class:`ChatResponse` to return on success, or
    * an :class:`Exception` instance to raise.

    Calls are matched to script entries in order. The class records
    every call (model + kwargs) in ``self.calls``, indexed by the
    order in which ``chat`` was invoked.
    """

    name = "scripted_advisor"

    def __init__(self, script: list[Any]) -> None:
        self._script: list[Any] = list(script)
        self._index: int = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self._index >= len(self._script):
            raise AssertionError(
                f"ScriptedAdapter: no more scripted responses "
                f"(call #{self._index + 1} for model={model})"
            )
        entry = self._script[self._index]
        self._index += 1
        if isinstance(entry, BaseException):
            raise entry
        if not isinstance(entry, ChatResponse):
            raise AssertionError(
                f"ScriptedAdapter: script entry must be ChatResponse or "
                f"Exception, got {type(entry).__name__}"
            )
        return entry

    async def stream(  # pragma: no cover - not exercised by advisor tests
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        if False:
            yield ""

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


def _advisor_response(
    content: str,
    *,
    model: str = "deepseek-v4-pro:cloud",
    prompt_tokens: int = 4,
    completion_tokens: int = 6,
) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-advisor",
        model=model,
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        finish_reason="stop",
    )


# ────────────────────────────────────────────────────────────────────
# Module exports
# ────────────────────────────────────────────────────────────────────


class TestAdvisorModuleExports:
    """The :mod:`moaxy.pipeline.advisor` module exports the documented names."""

    def test_parse_advisor_response_is_callable(self):
        assert callable(parse_advisor_response)

    def test_advisor_turn_is_callable(self):
        assert callable(advisor_turn)

    def test_advisor_turn_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(advisor_turn)

    def test_parse_advisor_response_is_not_a_coroutine(self):
        # parse_advisor_response is a pure synchronous helper.
        assert not inspect.iscoroutinefunction(parse_advisor_response)

    def test_pipeline_package_re_exports_advisor(self):
        from moaxy.pipeline import (
            advisor_turn as PackageAdvisorTurn,
        )
        from moaxy.pipeline import (
            parse_advisor_response as PackageParseAdvisorResponse,
        )

        assert PackageAdvisorTurn is advisor_turn
        assert PackageParseAdvisorResponse is parse_advisor_response


# ────────────────────────────────────────────────────────────────────
# parse_advisor_response
# ────────────────────────────────────────────────────────────────────


class TestParseAdvisorApprove:
    """The parser recognises ``ADVISOR_APPROVE`` (optionally with extra text)."""

    def test_plain_approve(self):
        decision, text, _score, _issues = parse_advisor_response("ADVISOR_APPROVE")
        assert decision == "approve"
        assert text is None

    def test_approve_with_trailing_whitespace(self):
        decision, text, _score, _issues = parse_advisor_response("ADVISOR_APPROVE   \n")
        assert decision == "approve"
        assert text is None

    def test_approve_with_explanatory_text(self):
        # The advisor may include explanation; any text alongside the
        # marker is treated as approval.
        decision, text, _score, _issues = parse_advisor_response(
            "The previous answer is good.\nADVISOR_APPROVE"
        )
        assert decision == "approve"
        assert text is None

    def test_approve_uppercase_only(self):
        # The marker is case-sensitive; lowercase variants are not approve.
        assert parse_advisor_response("advisor_approve") == ("revise", "advisor_approve", None, [])
        assert parse_advisor_response("Advisor_Approve") == ("revise", "Advisor_Approve", None, [])

    def test_approve_substring_is_treated_as_approve(self):
        # Substring matching is the historical contract (the parser uses
        # ``ADVISOR_APPROVE in text``). A substring occurrence anywhere
        # in the text is treated as approval. This matches the existing
        # orchestrator behaviour for backwards compatibility.
        assert parse_advisor_response("XADVISOR_APPROVE") == ("approve", None, None, [])


class TestParseAdvisorRevise:
    """The parser returns the text after ``ADVISOR_REVISE:`` on a revise."""

    def test_revise_with_text(self):
        decision, text, _score, _issues = parse_advisor_response("ADVISOR_REVISE: better answer")
        assert decision == "revise"
        assert text == "better answer"

    def test_revise_with_leading_newline(self):
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_REVISE:\nthis is the new answer"
        )
        assert decision == "revise"
        assert text == "this is the new answer"

    def test_revise_with_multiline_text(self):
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_REVISE: line1\nline2\nline3"
        )
        assert decision == "revise"
        assert text == "line1\nline2\nline3"

    def test_revise_with_explanatory_text_before_marker(self):
        # The advisor may explain its reasoning before the marker; only
        # the text after the marker is treated as the revised answer.
        decision, text, _score, _issues = parse_advisor_response(
            "Here is my critique of the previous answer.\n"
            "ADVISOR_REVISE: the final improved answer"
        )
        assert decision == "revise"
        assert text == "the final improved answer"

    def test_revise_text_is_stripped(self):
        # Leading and trailing whitespace on the revised text is removed.
        decision, text, _score, _issues = parse_advisor_response("ADVISOR_REVISE:    \n answer \n  ")
        assert decision == "revise"
        assert text == "answer"

    def test_revise_with_empty_text_after_marker(self):
        # The marker may be present but the text is empty; the decision
        # is still "revise" and the text is an empty string.
        decision, text, _score, _issues = parse_advisor_response("ADVISOR_REVISE:")
        assert decision == "revise"
        assert text == ""


class TestParseAdvisorMissing:
    """When neither marker is present, the helper treats the text as a revise."""

    def test_plain_text_treated_as_revise(self):
        decision, text, _score, _issues = parse_advisor_response("just a plain response")
        assert decision == "revise"
        assert text == "just a plain response"

    def test_empty_string_treated_as_revise(self):
        decision, text, _score, _issues = parse_advisor_response("")
        assert decision == "revise"
        assert text == ""

    def test_only_whitespace_treated_as_revise(self):
        decision, text, _score, _issues = parse_advisor_response("   \n\n  ")
        assert decision == "revise"
        assert text == ""

    def test_non_string_input_treated_as_revise(self):
        # The helper is defensive against non-string input.
        decision, text, _score, _issues = parse_advisor_response(None)  # type: ignore[arg-type]
        assert decision == "revise"
        assert text is None or text == ""


class TestParseAdvisorReturnsTuple:
    """The helper always returns a 4-tuple ``(decision, text, score, issues)``."""

    def test_returns_tuple_of_str_and_optional(self):
        result = parse_advisor_response("ADVISOR_APPROVE")
        assert isinstance(result, tuple)
        assert len(result) == 4
        decision, text, score, issues = result
        assert isinstance(decision, str)
        assert decision in {"approve", "revise"}

    def test_approve_text_is_none(self):
        _decision, text, _score, _issues = parse_advisor_response("ADVISOR_APPROVE")
        assert text is None

    def test_revise_text_is_string(self):
        _decision, text, _score, _issues = parse_advisor_response("ADVISOR_REVISE: x")
        assert isinstance(text, str)

    def test_legacy_approve_score_is_none(self):
        # The cross-critique fields default to ``None`` / ``[]`` when
        # the model emits only the legacy ``ADVISOR_APPROVE`` marker.
        _decision, _text, score, issues = parse_advisor_response(
            "ADVISOR_APPROVE"
        )
        assert score is None
        assert issues == []

    def test_legacy_revise_score_is_none(self):
        # The cross-critique fields default to ``None`` / ``[]`` when
        # the model emits only the legacy ``ADVISOR_REVISE:`` marker.
        _decision, _text, score, issues = parse_advisor_response(
            "ADVISOR_REVISE: better answer"
        )
        assert score is None
        assert issues == []


class TestParseAdvisorPrecedence:
    """When both markers appear, the parser picks the first one in the text."""

    def test_approve_first_wins(self):
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_APPROVE\nADVISOR_REVISE: ignored"
        )
        assert decision == "approve"
        assert text is None

    def test_revise_only_is_revise(self):
        # When the revise marker appears but no approve marker does, the
        # decision is revise. (The reverse case ``revise then approve``
        # resolves to approve because the parser checks the approve
        # marker first; that's the documented order.)
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_REVISE: this one"
        )
        assert decision == "revise"
        assert text == "this one"


# ────────────────────────────────────────────────────────────────────
# Cross-critique parser (DELTA 2 / VAL-PIPE-EXTRA-004, 005, 006)
# ────────────────────────────────────────────────────────────────────


class TestParseAdvisorCrossCritiqueDecision:
    """The ``ADVISOR_DECISION:`` line drives the decision when present.

    The cross-critique prompt format uses
    ``ADVISOR_DECISION: APPROVE|REVISE`` alongside the legacy
    ``ADVISOR_APPROVE`` / ``ADVISOR_REVISE:`` markers. When the
    new line is present, the captured value wins for the
    decision. The legacy ``ADVISOR_REVISE: <text>`` marker (when
    also present) supplies the revised text.
    """

    def test_decision_revise_with_score_and_revise_marker(self):
        # The expected-behavior case from the feature description.
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_SCORE: 7\n"
            "ADVISOR_REVISE: new answer"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "new answer",
            7,
            [],
        )

    def test_decision_approve_with_legacy_substring_ignored(self):
        # ``ADVISOR_DECISION: APPROVE`` wins; the legacy
        # ``ADVISOR_APPROVE`` substring (also present) is
        # consistent, not contradictory.
        text = "ADVISOR_DECISION: APPROVE\nADVISOR_APPROVE"
        assert parse_advisor_response(text) == (
            "approve",
            None,
            None,
            [],
        )

    def test_decision_approve_legacy_substring_only(self):
        # The legacy path (no ``ADVISOR_DECISION:`` line) returns
        # ``("approve", None, None, [])`` — the contract case from
        # VAL-PIPE-EXTRA-005.
        assert parse_advisor_response("ADVISOR_APPROVE") == (
            "approve",
            None,
            None,
            [],
        )

    def test_decision_revise_with_issues(self):
        # The expected-behavior case from the feature description:
        # ``ADVISOR_ISSUES:`` block supplies the bullet list;
        # ``ADVISOR_REVISE:`` supplies the revised text; no
        # ``ADVISOR_SCORE:`` line is emitted, so the score is
        # ``None``.
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "- issue 2\n"
            "ADVISOR_REVISE: improved"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "improved",
            None,
            ["issue 1", "issue 2"],
        )

    def test_decision_revise_with_asterisk_bullets(self):
        # ``*`` bullet markers are accepted (mirrors the
        # ``parse_advisor_issues`` behaviour).
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_ISSUES:\n"
            "* bullet one\n"
            "* bullet two\n"
            "ADVISOR_REVISE: new"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "new",
            None,
            ["bullet one", "bullet two"],
        )

    def test_decision_revise_with_score_and_issues(self):
        # All three cross-critique fields flow through in one call.
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_SCORE: 9\n"
            "ADVISOR_ISSUES:\n"
            "- a\n"
            "- b\n"
            "- c\n"
            "ADVISOR_REVISE: final"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "final",
            9,
            ["a", "b", "c"],
        )

    def test_decision_approve_with_score_only(self):
        # ``ADVISOR_DECISION: APPROVE`` with a non-revise path
        # still surfaces the score on the 4-tuple (the cross-
        # critique fields are extracted regardless of decision).
        text = "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 8"
        assert parse_advisor_response(text) == (
            "approve",
            None,
            8,
            [],
        )

    def test_decision_revise_without_revise_marker(self):
        # When ``ADVISOR_DECISION: REVISE`` is present but no
        # ``ADVISOR_REVISE:`` marker, the whole text is the
        # revised text (conservative fallback for missing marker).
        text = "ADVISOR_DECISION: REVISE\nADVISOR_SCORE: 5"
        assert parse_advisor_response(text) == (
            "revise",
            "ADVISOR_DECISION: REVISE\nADVISOR_SCORE: 5",
            5,
            [],
        )

    def test_decision_approve_without_approve_marker(self):
        # When ``ADVISOR_DECISION: APPROVE`` is present and the
        # legacy ``ADVISOR_APPROVE`` substring is absent, the
        # decision is still approve. The ``ADVISOR_REVISE:``
        # substring (if any) is ignored because the
        # ``ADVISOR_DECISION:`` line won the decision.
        text = (
            "Looks fine.\n"
            "ADVISOR_DECISION: APPROVE\n"
            "ADVISOR_REVISE: should be ignored"
        )
        assert parse_advisor_response(text) == (
            "approve",
            None,
            None,
            [],
        )

    def test_decision_is_case_sensitive(self):
        # The decision keyword is uppercase-only (matches the
        # cross-critique spec). Lowercase variants are NOT
        # parsed as the new marker; the legacy substring path
        # is the fallback (the input here has no legacy
        # markers, so the whole text is a revise).
        text = "advisor_decision: revise\nadvisor_score: 5"
        decision, _text, score, _issues = parse_advisor_response(text)
        assert decision == "revise"
        # The lowercase ``advisor_score:`` line is NOT the
        # ``ADVISOR_SCORE:`` marker, so the parsed score is
        # ``None``.
        assert score is None

    def test_decision_revise_lowercase_value_falls_through(self):
        # ``ADVISOR_DECISION:`` line present but with a
        # non-APPROVE/REVISE value: the regex does not match
        # (only APPROVE|REVISE are captured). The parser
        # falls through to the legacy substring parsers. No
        # legacy marker is present, so the decision is
        # ``revise`` with the whole text as the revised
        # answer. Score and issues are still extracted
        # additively.
        text = (
            "ADVISOR_DECISION: MAYBE\n"
            "ADVISOR_SCORE: 6\n"
            "ADVISOR_ISSUES:\n"
            "- a\n"
            "ADVISOR_REVISE: ignored-because-no-decision"
        )
        # Decision resolves to revise via the legacy fallback
        # (no ADVISOR_APPROVE substring, no ADVISOR_REVISE:
        # substring because the regex never matched the
        # ``ADVISOR_DECISION:`` line, but the
        # ``ADVISOR_REVISE:`` substring IS present in the
        # text — so the legacy revise path wins and
        # ``revised_text`` is the text after the marker).
        decision, _text, score, issues = parse_advisor_response(text)
        assert decision == "revise"
        # The legacy ``ADVISOR_REVISE:`` substring IS present
        # in the input, so the legacy revise path picks it
        # up and ``revised_text`` is the text after the
        # marker.
        assert "ignored-because-no-decision" in _text
        assert score == 6
        assert issues == ["a"]


class TestParseAdvisorCrossCritiqueBackwardCompat:
    """The cross-critique extension is backward compatible with v1-v4.

    Models that emit ONLY the legacy ``ADVISOR_APPROVE`` /
    ``ADVISOR_REVISE:`` markers still parse correctly: the
    decision is ``"approve"`` or ``"revise"``; the score is
    ``None``; the issues list is ``[]``. The v1-v4 behaviour
    is unchanged.
    """

    def test_legacy_approve_score_none_issues_empty(self):
        # ``ADVISOR_APPROVE`` only.
        _decision, _text, score, issues = parse_advisor_response(
            "ADVISOR_APPROVE"
        )
        assert score is None
        assert issues == []

    def test_legacy_revise_score_none_issues_empty(self):
        # ``ADVISOR_REVISE: <text>`` only.
        _decision, _text, score, issues = parse_advisor_response(
            "ADVISOR_REVISE: better answer"
        )
        assert score is None
        assert issues == []

    def test_legacy_approve_with_explanation(self):
        # ``ADVISOR_APPROVE`` with surrounding text. Score
        # and issues are still ``None`` / ``[]`` (no cross-
        # critique fields were emitted).
        _decision, _text, score, issues = parse_advisor_response(
            "Looks fine to me.\nADVISOR_APPROVE"
        )
        assert score is None
        assert issues == []

    def test_legacy_revise_with_explanation(self):
        # ``ADVISOR_REVISE:`` with explanation before the
        # marker.
        _decision, _text, score, issues = parse_advisor_response(
            "My critique:\nADVISOR_REVISE: improved answer"
        )
        assert score is None
        assert issues == []


class TestParseAdvisorCrossCritiqueScoreAndIssues:
    """Score and issues are extracted regardless of the decision path.

    The M5 cross-critique fields (score, issues) are
    additive to the decision; they are extracted from the
    full input text on every call. The examples below pin
    the order in which the helpers are called (score first,
    issues second) and confirm the cross-critique fields
    flow through on each decision path.
    """

    def test_score_extracted_on_legacy_approve(self):
        # A model that emits the legacy approve marker AND a
        # cross-critique score line gets both: the decision
        # is approve, the score is parsed.
        text = "The answer is correct.\nADVISOR_APPROVE\nADVISOR_SCORE: 9"
        assert parse_advisor_response(text) == (
            "approve",
            None,
            9,
            [],
        )

    def test_score_extracted_on_legacy_revise(self):
        # Same pattern on the revise path. The legacy
        # ``ADVISOR_REVISE:`` marker takes the post-marker
        # text verbatim as the revised answer; the
        # ``ADVISOR_SCORE:`` line in the post-marker text is
        # part of the revised answer in the v1-v4 contract.
        # The score is still extracted additively (via the
        # cross-critique helper), so a model that interleaves
        # the markers still surfaces the score on the
        # 4-tuple.
        text = (
            "Critique here.\n"
            "ADVISOR_REVISE: improved\n"
            "ADVISOR_SCORE: 4"
        )
        decision, revised, score, issues = parse_advisor_response(text)
        assert decision == "revise"
        assert revised.startswith("improved")
        assert "ADVISOR_SCORE: 4" in revised
        assert score == 4
        assert issues == []

    def test_issues_extracted_on_legacy_approve(self):
        # A model that emits the legacy approve marker AND a
        # cross-critique issues block gets both: the decision
        # is approve, the issues list is parsed.
        text = (
            "ADVISOR_APPROVE\n"
            "ADVISOR_ISSUES:\n"
            "- minor nit\n"
            "- second nit"
        )
        assert parse_advisor_response(text) == (
            "approve",
            None,
            None,
            ["minor nit", "second nit"],
        )

    def test_last_score_wins(self):
        # When the model emits two ``ADVISOR_SCORE:`` lines,
        # the last one wins (mirrors the existing
        # ``parse_advisor_score`` behaviour).
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_SCORE: 4\n"
            "ADVISOR_SCORE: 8\n"
            "ADVISOR_REVISE: final"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "final",
            8,
            [],
        )

    def test_last_issues_block_wins(self):
        # When the model emits two ``ADVISOR_ISSUES:``
        # blocks, the last one wins (mirrors the existing
        # ``parse_advisor_issues`` behaviour).
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_ISSUES:\n"
            "- first\n"
            "ADVISOR_ISSUES:\n"
            "- second\n"
            "- third\n"
            "ADVISOR_REVISE: final"
        )
        assert parse_advisor_response(text) == (
            "revise",
            "final",
            None,
            ["second", "third"],
        )


# ────────────────────────────────────────────────────────────────────
# advisor_turn
# ────────────────────────────────────────────────────────────────────


class TestAdvisorTurnCallsAdapter:
    """advisor_turn calls the configured adapter with the advisor prompt."""

    @pytest.mark.asyncio
    async def test_uses_model_from_argument(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="deepseek-v4-pro:cloud",
            history=[{"role": "user", "content": "hi"}],
            current_answer="the answer to review",
        )
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_messages_use_advisor_builder(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        history = [{"role": "user", "content": "hi"}]
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=history,
            current_answer="the answer to review",
        )
        sent = adapter.calls[0]["messages"]
        # The last message should be a user-role advisor request that
        # includes both the canonical prefix and the answer text.
        last = sent[-1]
        assert last["role"] == "user"
        assert "advise on this" in last["content"]
        assert "the answer to review" in last["content"]

    @pytest.mark.asyncio
    async def test_history_is_passed_through(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        history = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "what is 2+2?"},
        ]
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=history,
            current_answer="4",
        )
        sent = adapter.calls[0]["messages"]
        for entry in history:
            assert entry in sent

    @pytest.mark.asyncio
    async def test_explicit_default_system_prompt_is_attached(self):
        """When the caller passes the default system prompt, it is prepended."""
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
            system_prompt=DEFAULT_ADVISOR_PROMPT,
        )
        sent = adapter.calls[0]["messages"]
        first = sent[0]
        assert first["role"] == "system"
        assert first["content"] == DEFAULT_ADVISOR_PROMPT

    @pytest.mark.asyncio
    async def test_custom_system_prompt_is_attached(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        custom = "You are a strict advisor. Be terse."
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
            system_prompt=custom,
        )
        sent = adapter.calls[0]["messages"]
        first = sent[0]
        assert first["role"] == "system"
        assert first["content"] == custom

    @pytest.mark.asyncio
    async def test_no_system_message_when_prompt_empty(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[{"role": "user", "content": "hi"}],
            current_answer="x",
            system_prompt="",
        )
        sent = adapter.calls[0]["messages"]
        assert all(m["role"] != "system" for m in sent)

    @pytest.mark.asyncio
    async def test_history_not_mutated(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        original = deepcopy(history)
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=history,
            current_answer="x",
        )
        assert history == original


class TestAdvisorTurnReturnsDecision:
    """advisor_turn returns ``(decision, text)`` based on the parsed response."""

    @pytest.mark.asyncio
    async def test_approve_returns_approve_none(self):
        adapter = ScriptedAdapter(
            [_advisor_response("Looks good to me.\nADVISOR_APPROVE")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert decision == "approve"
        assert text is None

    @pytest.mark.asyncio
    async def test_revise_returns_revise_with_text(self):
        adapter = ScriptedAdapter(
            [_advisor_response("ADVISOR_REVISE: better answer")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert decision == "revise"
        assert text == "better answer"

    @pytest.mark.asyncio
    async def test_revise_with_multiline_text(self):
        adapter = ScriptedAdapter(
            [_advisor_response("ADVISOR_REVISE: line1\nline2\nline3")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert decision == "revise"
        assert text == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_plain_text_treated_as_revise(self):
        adapter = ScriptedAdapter(
            [_advisor_response("the previous answer is fine")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        # The marker is missing, so the helper falls back to "revise"
        # with the entire content as the revised text.
        assert decision == "revise"
        assert text == "the previous answer is fine"

    @pytest.mark.asyncio
    async def test_returns_a_tuple_of_str_and_optional(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        result = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        decision, text = result
        assert isinstance(decision, str)
        assert decision in {"approve", "revise"}


class TestAdvisorTurnCrossCritiqueContext:
    """``advisor_turn`` surfaces the cross-critique fields on the context.

    The M5 cross-critique format extends ``advisor_turn`` to
    expose ``ctx["advisor_score"]`` (the parsed
    ``ADVISOR_SCORE:`` value) and ``ctx["advisor_issues"]``
    (the parsed ``ADVISOR_ISSUES:`` bullet list) in addition
    to the existing ``ctx["advisor_decision"]`` and
    ``ctx["advisor_revised_text"]`` keys. ADVISOR plugins and
    orchestrator observers read these keys to get the full
    cross-critique context.
    """

    @pytest.mark.asyncio
    async def test_score_surfaced_on_context(self):
        adapter = ScriptedAdapter([
            _advisor_response(
                "ADVISOR_DECISION: REVISE\n"
                "ADVISOR_SCORE: 7\n"
                "ADVISOR_REVISE: new answer"
            )
        ])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert ctx["advisor_score"] == 7

    @pytest.mark.asyncio
    async def test_issues_surfaced_on_context(self):
        adapter = ScriptedAdapter([
            _advisor_response(
                "ADVISOR_DECISION: REVISE\n"
                "ADVISOR_ISSUES:\n"
                "- issue 1\n"
                "- issue 2\n"
                "ADVISOR_REVISE: improved"
            )
        ])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert ctx["advisor_issues"] == ["issue 1", "issue 2"]

    @pytest.mark.asyncio
    async def test_score_and_issues_surfaced_on_legacy_approve(self):
        # The cross-critique fields are surfaced even on a
        # legacy ``ADVISOR_APPROVE`` path (the helpers are
        # additive).
        adapter = ScriptedAdapter([
            _advisor_response(
                "The answer is correct.\n"
                "ADVISOR_APPROVE\n"
                "ADVISOR_SCORE: 9\n"
                "ADVISOR_ISSUES:\n"
                "- nit"
            )
        ])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert ctx["advisor_score"] == 9
        assert ctx["advisor_issues"] == ["nit"]

    @pytest.mark.asyncio
    async def test_score_none_when_missing(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert ctx["advisor_score"] is None
        assert ctx["advisor_issues"] == []


class TestAdvisorTurnErrorPropagation:
    """Adapter errors bubble out of advisor_turn unchanged."""

    @pytest.mark.asyncio
    async def test_upstream_error_is_raised(self):
        adapter = ScriptedAdapter(
            [UpstreamError("boom", status_code=500, body="server error")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        with pytest.raises(UpstreamError) as exc_info:
            await advisor_turn(
                ctx,
                advisor_model="m",
                history=[],
                current_answer="x",
            )
        assert exc_info.value.status_code == 500


# ────────────────────────────────────────────────────────────────────
# advisor_turn + PluginManager integration
# ────────────────────────────────────────────────────────────────────


class _RecorderAdvisorPlugin(Plugin):
    """An ADVISOR plugin that records every process_async call."""

    name = "recorder_advisor"
    version = "1.0.0"
    plugin_type = PluginType.ADVISOR

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.contexts: list[dict[str, Any]] = []

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    async def process_async(self, context: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.contexts.append(context)
        return context


def _build_manager_with(plugin: Plugin) -> PluginManager:
    """Build a PluginManager with ``plugin`` already registered."""
    mgr = PluginManager(plugins_dir=PLUGINS_DIR_FOR_TESTS)
    mgr._plugins[plugin.name] = plugin
    mgr._loaded = True
    return mgr


class TestAdvisorTurnPluginDispatch:
    """advisor_turn invokes the configured ADVISOR plugins per turn."""

    @pytest.mark.asyncio
    async def test_runs_advisor_plugin_once(self):
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        plugin = _RecorderAdvisorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert plugin.call_count == 1

    @pytest.mark.asyncio
    async def test_no_plugin_manager_is_ok(self):
        """When the context has no plugin manager, the advisor_turn still runs."""
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert decision == "approve"
        assert text is None

    @pytest.mark.asyncio
    async def test_plugin_receives_context(self):
        """The plugin sees the context dict advisor_turn was given."""
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        plugin = _RecorderAdvisorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {
            "adapter": adapter,
            "plugin_manager": mgr,
            "request_id": "req-abc",
            "route": "reflective-coder",
        }
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert plugin.call_count == 1
        ctx_seen = plugin.contexts[0]
        assert ctx_seen["request_id"] == "req-abc"
        assert ctx_seen["route"] == "reflective-coder"

    @pytest.mark.asyncio
    async def test_plugin_receives_advisor_text_and_decision(self):
        """The plugin can read the parsed decision and advisor text from the ctx."""
        adapter = ScriptedAdapter(
            [_advisor_response("ADVISOR_REVISE: better answer")]
        )
        plugin = _RecorderAdvisorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        ctx_seen = plugin.contexts[0]
        # The plugin sees the decision and the advisor text (the full
        # assistant content, including the marker).
        assert ctx_seen["advisor_decision"] == "revise"
        assert ctx_seen["advisor_text"] == "ADVISOR_REVISE: better answer"
        # The plugin sees the parsed revised text (without the marker).
        assert ctx_seen["advisor_revised_text"] == "better answer"
        # The plugin sees the raw ChatResponse too.
        assert "advisor_response" in ctx_seen
        assert ctx_seen["advisor_model"] == "m"

    @pytest.mark.asyncio
    async def test_non_advisor_plugins_not_invoked(self):
        """PluginManager.run with ADVISOR only touches ADVISOR plugins."""

        class _RecorderTransformer(Plugin):
            name = "recorder_transformer_for_advisor"
            version = "1.0.0"
            plugin_type = PluginType.TRANSFORMER

            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def process(self, context):
                self.call_count += 1
                return context

            async def process_async(self, context):
                return context

        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        advisor = _RecorderAdvisorPlugin()
        transformer = _RecorderTransformer()
        mgr = PluginManager(plugins_dir=PLUGINS_DIR_FOR_TESTS)
        mgr._plugins[advisor.name] = advisor
        mgr._plugins[transformer.name] = transformer
        mgr._loaded = True

        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=[],
            current_answer="x",
        )
        assert advisor.call_count == 1
        assert transformer.call_count == 0


# ────────────────────────────────────────────────────────────────────
# Integration with build_advisor_messages
# ────────────────────────────────────────────────────────────────────


class TestAdvisorTurnWithMessageBuilders:
    """advisor_turn composes cleanly with build_advisor_messages."""

    @pytest.mark.asyncio
    async def test_uses_build_advisor_messages(self):
        """The same messages build_advisor_messages produces are forwarded to the adapter."""
        adapter = ScriptedAdapter([_advisor_response("ADVISOR_APPROVE")])
        ctx: dict[str, Any] = {"adapter": adapter}
        history = [{"role": "user", "content": "hi"}]
        answer = "the answer"
        await advisor_turn(
            ctx,
            advisor_model="m",
            history=history,
            current_answer=answer,
            system_prompt=DEFAULT_ADVISOR_PROMPT,
        )
        expected = build_advisor_messages(
            history=history, answer=answer, system_prompt=DEFAULT_ADVISOR_PROMPT
        )
        assert adapter.calls[0]["messages"] == expected


# ────────────────────────────────────────────────────────────────────
# Self-advise (advisor.model == primary model)
# ────────────────────────────────────────────────────────────────────


class TestSelfAdvise:
    """Self-advise (advisor.model == primary.model) still triggers a separate call."""

    @pytest.mark.asyncio
    async def test_self_advise_uses_same_model_name(self):
        """When advisor.model == primary, the advisor call is a separate LLM call."""
        adapter = ScriptedAdapter(
            [
                _advisor_response("ADVISOR_APPROVE", model="minimax-m3:cloud"),
            ]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        decision, text = await advisor_turn(
            ctx,
            advisor_model="minimax-m3:cloud",
            history=[],
            current_answer="x",
        )
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"
        assert decision == "approve"
        assert text is None


# ────────────────────────────────────────────────────────────────────
# Orchestrator integration: ADVISOR plugin dispatch
# ────────────────────────────────────────────────────────────────────


class TestBuildAdvisorRevisionMessages:
    """``build_advisor_revision_messages`` materialises the primary revision call."""

    def test_prepends_system_message(self):
        msgs = build_advisor_revision_messages(
            history=[],
            answer="post-reflection answer",
            advisor_feedback="fix X",
            system_prompt="reflector system prompt",
        )
        assert msgs[0] == {
            "role": "system",
            "content": "reflector system prompt",
        }

    def test_includes_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        msgs = build_advisor_revision_messages(
            history=history,
            answer="x",
            advisor_feedback="y",
            system_prompt=None,
        )
        for entry in history:
            assert entry in msgs

    def test_uses_critique_then_answer_then_advisor_feedback(self):
        msgs = build_advisor_revision_messages(
            history=[],
            answer="the previous answer",
            advisor_feedback="the advisor's feedback",
            system_prompt=None,
        )
        # The last message is the user-role feedback request.
        last = msgs[-1]
        assert last["role"] == "user"
        assert "incorporate the advisor's feedback" in last["content"]
        assert "the advisor's feedback" in last["content"]
        # The previous-assistant message holds the previous answer.
        assistant_msg = msgs[-2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "the previous answer"

    def test_history_not_mutated(self):
        history = [{"role": "user", "content": "hi"}]
        snapshot = [{"role": "user", "content": "hi"}]
        build_advisor_revision_messages(
            history=history,
            answer="x",
            advisor_feedback="y",
            system_prompt=None,
        )
        assert history == snapshot

    def test_no_system_message_when_prompt_empty(self):
        msgs = build_advisor_revision_messages(
            history=[],
            answer="x",
            advisor_feedback="y",
            system_prompt="",
        )
        assert all(m["role"] != "system" for m in msgs)


class TestOrchestratorAdvisorPluginDispatch:
    """The orchestrator invokes :class:`PluginType.ADVISOR` plugins per advisor pass.

    These tests pin the M3 contract: when the orchestrator runs the
    advisor stage, the configured :class:`PluginType.ADVISOR` plugins
    are dispatched through ``plugin_manager.run`` with the
    ``PluginType.ADVISOR`` type. The plugins see a context dict
    carrying the parsed decision and the advisor text.
    """

    @pytest.mark.asyncio
    async def test_advisor_plugin_invoked_by_orchestrator(self):
        """An ADVISOR plugin registered on the manager is invoked per advisor pass."""
        from moaxy.adapters.base import (
            ChatResponse,
            Message,
            Usage,
        )
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import (
            RouteMatch as ConfigRouteMatch,
        )
        from moaxy.pipeline.context import PipelineContext
        from moaxy.pipeline.orchestrator import Orchestrator
        from moaxy.routing.matcher import RouteMatch

        class _RecorderAdvisor(Plugin):
            name = "recorder_orchestrator_advisor"
            version = "1.0.0"
            plugin_type = PluginType.ADVISOR

            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0
                self.contexts: list[dict[str, Any]] = []

            def process(self, context):
                return context

            async def process_async(self, context):
                self.call_count += 1
                self.contexts.append(context)
                return context

        def _advisor_response(
            content: str, *, model: str = "deepseek-v4-pro:cloud"
        ) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-advisor",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=4, completion_tokens=6, total_tokens=10
                ),
                finish_reason="stop",
            )

        def _initial_response() -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-init",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content="initial"),
                usage=Usage(prompt_tokens=4, completion_tokens=4, total_tokens=8),
                finish_reason="stop",
            )

        adapter = ScriptedAdapter(
            [
                _initial_response(),
                _advisor_response("ADVISOR_APPROVE"),
            ]
        )
        plugin = _RecorderAdvisor()
        mgr = PluginManager(plugins_dir=PLUGINS_DIR_FOR_TESTS)
        mgr._plugins[plugin.name] = plugin
        mgr._loaded = True

        config_route = RouteConfig(
            name="test-advisor-plugin-route",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=0),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-1",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )

        orchestrator = Orchestrator(adapter, plugin_manager=mgr)
        await orchestrator.run(ctx)

        # The plugin was invoked once for the single advisor pass.
        assert plugin.call_count == 1
        ctx_seen = plugin.contexts[0]
        # The plugin context carries the parsed verdict.
        assert ctx_seen["advisor_decision"] == "approve"
        assert ctx_seen["advisor_text"] == "ADVISOR_APPROVE"


# ────────────────────────────────────────────────────────────────────
# Cross-advise (M3): primary and advisor use different model names
# ────────────────────────────────────────────────────────────────────


class TestCrossAdviseOrchestrator:
    """Cross-advise: the advisor call uses a different model name than the primary.

    M3 cross-advise is the case where ``advisor.model`` differs from the
    primary model. The orchestrator must issue the advisor call to the
    configured advisor model name (not the primary's), and the
    :func:`build_response_headers` helper must surface that name in the
    ``x-moaxy-advisor-model`` response header.
    """

    def _make_response(
        self,
        content: str,
        *,
        model: str,
        prompt_tokens: int = 4,
        completion_tokens: int = 4,
    ) -> ChatResponse:
        return ChatResponse(
            id=f"chatcmpl-{model}",
            model=model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    def _build_route(
        self,
        *,
        primary_model: str = "minimax-m3:cloud",
        advisor_model: str | None = "deepseek-v4-pro:cloud",
        reflection_turns: int = 0,
        advisor_turns: int = 1,
    ) -> Any:
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch

        config_route = RouteConfig(
            name="cross-advise-route",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": primary_model},
            reflection=ReflectionConfig(turns=reflection_turns),
            advisor=AdvisorConfig(model=advisor_model, turns=advisor_turns),
        )
        # Use the matcher.Routematch (the dataclass) since it accepts
        # the per-route typed config. The Pydantic ``RouteMatch`` model
        # is a different type (the glob matcher config) and rejects
        # these fields.
        from moaxy.routing.matcher import RouteMatch as MatcherRouteMatch
        rm = MatcherRouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model=primary_model,
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        return rm

    def _build_context(
        self,
        route: Any,
        *,
        request_id: str = "req-1",
    ) -> Any:
        from moaxy.pipeline.context import PipelineContext
        return PipelineContext(
            request_id=request_id,
            request={
                "model": route.original_model,
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )

    @pytest.mark.asyncio
    async def test_advisor_call_uses_advisor_model(self):
        """The advisor LLM call carries the configured ``advisor.model`` name."""
        adapter = ScriptedAdapter(
            [
                self._make_response("initial", model="minimax-m3:cloud"),
                self._make_response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        route = self._build_route()
        ctx = self._build_context(route)
        await Orchestrator(adapter).run(ctx)

        # Two calls: initial on the primary, advisor on the advisor model.
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "deepseek-v4-pro:cloud",
        ]

    @pytest.mark.asyncio
    async def test_advisor_model_header_reports_advisor_name(self):
        """The ``x-moaxy-advisor-model`` header equals ``advisor.model``."""
        adapter = ScriptedAdapter(
            [
                self._make_response("initial", model="minimax-m3:cloud"),
                self._make_response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        route = self._build_route()
        ctx = self._build_context(route, request_id="req-cross")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The advisor model header reports the configured advisor.
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_cross_advise_revise_uses_advisor_then_primary(self):
        """ADVISOR_REVISE → a follow-up primary call; the final response
        is the primary's revision, not the advisor's text."""
        adapter = ScriptedAdapter(
            [
                self._make_response("initial", model="minimax-m3:cloud"),
                self._make_response(
                    "ADVISOR_REVISE: better answer",
                    model="deepseek-v4-pro:cloud",
                ),
                # The follow-up primary revision.
                self._make_response(
                    "primary-revised", model="minimax-m3:cloud"
                ),
            ]
        )
        route = self._build_route()
        ctx = self._build_context(route, request_id="req-revise")
        await Orchestrator(adapter).run(ctx)
        # The final response carries the primary's revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "primary-revised"
        # Three calls: initial, advisor (to deepseek), primary-revise.
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "deepseek-v4-pro:cloud",
            "minimax-m3:cloud",
        ]
        # The header still reports the advisor model.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# Self-advise (M3): advisor.model equals the primary
# ────────────────────────────────────────────────────────────────────


class TestSelfAdviseOrchestrator:
    """Self-advise: advisor.model == primary model; both calls go to the same name."""

    def _make_response(
        self,
        content: str,
        *,
        model: str,
        prompt_tokens: int = 4,
        completion_tokens: int = 4,
    ) -> ChatResponse:
        return ChatResponse(
            id=f"chatcmpl-{model}",
            model=model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    @pytest.mark.asyncio
    async def test_self_advise_makes_two_separate_llm_calls(self):
        """Self-advise issues a separate advisor LLM call, even though the
        model name is the same as the primary."""
        adapter = ScriptedAdapter(
            [
                self._make_response("initial", model="minimax-m3:cloud"),
                self._make_response(
                    "ADVISOR_APPROVE", model="minimax-m3:cloud"
                ),
            ]
        )
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.routing.matcher import RouteMatch

        config_route = RouteConfig(
            name="self-advise-route",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=0),
            advisor=AdvisorConfig(
                model="minimax-m3:cloud", turns=1
            ),
        )
        route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-self",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )
        await Orchestrator(adapter).run(ctx)
        # Two calls: both to the same model name.
        assert len(adapter.calls) == 2
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "minimax-m3:cloud",
        ]
        # The advisor header reports the (same) advisor model.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-model"] == "minimax-m3:cloud"


# ────────────────────────────────────────────────────────────────────
# Advisor runs AFTER reflection
# ────────────────────────────────────────────────────────────────────


class TestAdvisorAfterReflection:
    """The advisor stage runs after the reflection loop in the default sequential path."""

    def _make_response(
        self,
        content: str,
        *,
        model: str = "minimax-m3:cloud",
        prompt_tokens: int = 4,
        completion_tokens: int = 4,
    ) -> ChatResponse:
        return ChatResponse(
            id=f"chatcmpl-{id(self)}-{prompt_tokens}",
            model=model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    @pytest.mark.asyncio
    async def test_advisor_called_after_reflection_revision(self):
        """The advisor LLM call is the last call; the revision's content
        is the input to the advisor (not the initial answer)."""
        adapter = ScriptedAdapter(
            [
                self._make_response("initial-answer", prompt_tokens=5, completion_tokens=3),
                # Critique.
                self._make_response("critique\nREFLECT_CONFIDENCE: 0.5"),
                # Revision.
                self._make_response("revised-answer", prompt_tokens=4, completion_tokens=4),
                # Advisor call.
                self._make_response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.routing.matcher import RouteMatch

        config_route = RouteConfig(
            name="reflective-advisor-route",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=1, early_exit=False),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-arf",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )
        await Orchestrator(adapter).run(ctx)
        # 4 LLM calls: initial, critique, revision, advisor.
        assert len(adapter.calls) == 4
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",  # initial
            "minimax-m3:cloud",  # critique
            "minimax-m3:cloud",  # revision
            "deepseek-v4-pro:cloud",  # advisor
        ]
        # Event ordering: initial → critique → revised → advisor → advisor_approve.
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "advisor",
            "advisor_approve",
        ]
        # The advisor's input is the post-reflection answer.
        advisor_call = adapter.calls[3]
        # The last user-role message in the advisor call contains the revised answer.
        last_user_msg = None
        for msg in advisor_call["messages"]:
            if msg.get("role") == "user":
                last_user_msg = msg
        assert last_user_msg is not None
        assert "revised-answer" in last_user_msg["content"]

    @pytest.mark.asyncio
    async def test_advisor_sees_post_reflection_answer_not_initial(self):
        """The advisor call's messages list references the REVISED answer,
        not the initial one. (Concrete check on input payload.)"""
        adapter = ScriptedAdapter(
            [
                self._make_response("INITIAL_ANSWER", prompt_tokens=5, completion_tokens=3),
                self._make_response("critique\nREFLECT_CONFIDENCE: 0.5"),
                self._make_response("REVISED_ANSWER", prompt_tokens=4, completion_tokens=4),
                self._make_response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.routing.matcher import RouteMatch

        config_route = RouteConfig(
            name="advisor-sees-revised",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=1, early_exit=False),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-input",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )
        await Orchestrator(adapter).run(ctx)
        # The advisor call's messages reference the REVISED answer.
        advisor_call = adapter.calls[3]
        advisor_messages = advisor_call["messages"]
        advisor_blob = " ".join(m.get("content", "") for m in advisor_messages)
        # The revised answer is in the advisor call's messages; the
        # initial answer is NOT in any of the advisor's input messages.
        assert "REVISED_ANSWER" in advisor_blob
        assert "INITIAL_ANSWER" not in advisor_blob


# ────────────────────────────────────────────────────────────────────
# Parallel correctness (advisor.parallel=true)
# ────────────────────────────────────────────────────────────────────


class TestAdvisorParallelCorrectness:
    """``advisor.parallel: true`` engages the M4 parallel path.

    The M4 contract is that when ``reflection.parallel: true`` AND
    ``advisor.parallel: true`` are both set, the orchestrator runs
    the final advisor revision concurrently with a self-reflection
    on the original answer, taking whichever finishes last. The
    contract pins content equivalence to the sequential path
    (VAL-PIPE-021); no strict timing assertion is made.

    The scripted FakeAdapter tests below supply responses for both
    concurrent paths (advisor and self-reflection); the orchestrator
    may consume them in either order.
    """

    def _make_response(
        self,
        content: str,
        *,
        model: str = "minimax-m3:cloud",
        prompt_tokens: int = 4,
        completion_tokens: int = 4,
    ) -> ChatResponse:
        return ChatResponse(
            id=f"chatcmpl-{id(self)}-{prompt_tokens}",
            model=model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )

    @pytest.mark.asyncio
    async def test_advisor_parallel_true_runs_equivalent_to_sequential(self):
        """``advisor.parallel=true`` (with ``reflection.parallel=true``)
        runs the advisor and a self-reflection on the initial
        answer concurrently via ``asyncio.gather``. The scripted
        response queue supplies responses for both paths; the
        orchestrator consumes them in either order, but the final
        answer is content-equivalent to the sequential path.
        """
        # Script: 2 for the advisor path (advisor call, no primary
        # revision because ADVISOR_APPROVE), 2 for the self-
        # reflection on the initial answer.
        adapter = ScriptedAdapter(
            [
                self._make_response("initial"),
                self._make_response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
                self._make_response(
                    "self_critique", model="minimax-m3:cloud"
                ),
                self._make_response(
                    "self_revised", model="minimax-m3:cloud"
                ),
            ]
        )
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.routing.matcher import RouteMatch

        config_route = RouteConfig(
            name="advisor-parallel",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=0, parallel=True),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud",
                turns=1,
                parallel=True,  # M4 path; engages when reflection.parallel is also true.
            ),
        )
        route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-parallel",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved=route.resolved_model,
            target_backend=route.backend,
            original_model=route.original_model,
        )
        await Orchestrator(adapter).run(ctx)
        # The M4 path runs the advisor and a (noop) self-reflection
        # concurrently. With reflection.turns=0 the self-reflection
        # is a no-op, so the only LLM calls are the initial and the
        # advisor.
        assert len(adapter.calls) == 2
        # The advisor header is present.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# Concurrent request isolation (M3: per-context state)
# ────────────────────────────────────────────────────────────────────


class TestAdvisorConcurrentRequestIsolation:
    """Concurrent requests with overlapping config do not cross-contaminate.

    The :class:`Orchestrator` and the :func:`advisor_turn` helper are
    stateless from request to request: every coroutine receives its
    own :class:`PipelineContext` (or context dict) and mutates only
    that object. Two concurrent coroutines must each see their own
    request_id, route, adapter call log, and final response.
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_advisor_turns_isolated(self):
        """Two concurrent ``advisor_turn`` coroutines each see their own state."""
        adapter = ScriptedAdapter(
            [
                _advisor_response("ADVISOR_APPROVE"),  # request A
                _advisor_response("ADVISOR_REVISE: better"),  # request B
            ]
        )
        ctx_a: dict[str, Any] = {
            "adapter": adapter,
            "request_id": "req-A",
        }
        ctx_b: dict[str, Any] = {
            "adapter": adapter,
            "request_id": "req-B",
        }
        # Run both advisor passes concurrently.
        results = await asyncio.gather(
            advisor_turn(
                ctx_a,
                advisor_model="model-A",
                history=[{"role": "user", "content": "A"}],
                current_answer="answer A",
            ),
            advisor_turn(
                ctx_b,
                advisor_model="model-B",
                history=[{"role": "user", "content": "B"}],
                current_answer="answer B",
            ),
        )
        decision_a, text_a = results[0]
        decision_b, text_b = results[1]
        # Each request's decision is independent.
        assert decision_a == "approve"
        assert text_a is None
        assert decision_b == "revise"
        assert text_b == "better"
        # The two advisor calls carried the correct model names.
        assert adapter.calls[0]["model"] == "model-A"
        assert adapter.calls[1]["model"] == "model-B"
        # Each context dict carries its own request_id.
        assert ctx_a["request_id"] == "req-A"
        assert ctx_b["request_id"] == "req-B"

    @pytest.mark.asyncio
    async def test_two_concurrent_orchestrator_runs_isolated(self):
        """Two concurrent :meth:`Orchestrator.run` invocations do not share state."""
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.routing.matcher import RouteMatch

        def _make_resp(content: str, *, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-x",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=4, completion_tokens=4, total_tokens=8
                ),
                finish_reason="stop",
            )

        def _build_context(
            request_id: str, original_answer: str
        ) -> PipelineContext:
            config_route = RouteConfig(
                name="concurrent-route",
                match=ConfigRouteMatch(
                    model="*", path="/v1/chat/completions"
                ),
                backend="ollama-local",
                aliases={"coder-pro": "m1"},
                reflection=ReflectionConfig(turns=0),
                advisor=AdvisorConfig(model="m2", turns=1),
            )
            route = RouteMatch(
                route=config_route,
                original_model="coder-pro",
                resolved_model="m1",
                backend="ollama-local",
                path="/v1/chat/completions",
                reflection=config_route.reflection,
                advisor=config_route.advisor,
                fallbacks=[],
                retry=0,
                aliases=dict(config_route.aliases),
            )
            return PipelineContext(
                request_id=request_id,
                request={
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": original_answer}],
                },
                route=route,
                model_alias_resolved=route.resolved_model,
                target_backend=route.backend,
                original_model=route.original_model,
            )

        # Script: request A's calls come first (initial + advisor), then
        # request B's calls (initial + advisor + primary-revise).
        adapter = ScriptedAdapter(
            [
                _make_resp("initial-A", model="m1"),
                _advisor_response("ADVISOR_APPROVE", model="m2"),
                _make_resp("initial-B", model="m1"),
                _advisor_response("ADVISOR_REVISE: better B", model="m2"),
                _make_resp("primary-revised-B", model="m1"),
            ]
        )
        ctx_a = _build_context("req-conc-A", "answer-A")
        ctx_b = _build_context("req-conc-B", "answer-B")
        await asyncio.gather(
            Orchestrator(adapter).run(ctx_a),
            Orchestrator(adapter).run(ctx_b),
        )
        # Each context has the response matching its own request.
        assert ctx_a.upstream_response is not None
        assert ctx_b.upstream_response is not None
        # Context A's response is the initial answer (advisor approved).
        assert ctx_a.upstream_response.message.content == "initial-A"
        # Context B's response is the primary revision after ADVISOR_REVISE.
        assert ctx_b.upstream_response.message.content == "primary-revised-B"
        # The contexts' request_ids are unchanged.
        assert ctx_a.request_id == "req-conc-A"
        assert ctx_b.request_id == "req-conc-B"
        # Each context has its own events list.
        assert len(ctx_a.events) == 3  # initial, advisor, advisor_approve
        assert len(ctx_b.events) >= 3  # initial, advisor, advisor_revised, advisor_revision


# ────────────────────────────────────────────────────────────────────
# parse_advisor_score (DELTA 5/6: integer-only 0-10 score from advisor)
# ────────────────────────────────────────────────────────────────────


class TestParseAdvisorScoreHappyPath:
    """The regex matches the documented ``ADVISOR_SCORE: <int>`` line."""

    def test_simple_integer(self):
        assert parse_advisor_score("ADVISOR_SCORE: 8") == 8

    def test_zero(self):
        assert parse_advisor_score("ADVISOR_SCORE: 0") == 0

    def test_ten_upper_bound(self):
        assert parse_advisor_score("ADVISOR_SCORE: 10") == 10

    def test_mid_value(self):
        assert parse_advisor_score("ADVISOR_SCORE: 5") == 5

    def test_out_of_range_value_recorded_as_is(self):
        # The parser does not clamp; "11" is recorded as-is.
        assert parse_advisor_score("ADVISOR_SCORE: 11") == 11


class TestParseAdvisorScoreInAdvisorText:
    """The line can be embedded anywhere in the advisor's text."""

    def test_line_appears_with_other_markers(self):
        text = (
            "ADVISOR_DECISION: REVISE\n"
            "ADVISOR_SCORE: 7\n"
            "ADVISOR_REVISE: better answer"
        )
        assert parse_advisor_score(text) == 7

    def test_takes_last_match_when_multiple_lines(self):
        # "Last match wins" mirrors parse_score and parse_confidence.
        text = (
            "ADVISOR_SCORE: 3\n"
            "ADVISOR_SCORE: 8"
        )
        assert parse_advisor_score(text) == 8

    def test_no_space_after_colon(self):
        # The regex allows zero-or-more whitespace after the colon.
        assert parse_advisor_score("ADVISOR_SCORE:8") == 8


class TestParseAdvisorScoreWhitespaceHandling:
    """The regex tolerates whitespace around the marker and the number."""

    def test_multiple_spaces_after_colon(self):
        assert parse_advisor_score("ADVISOR_SCORE:    7") == 7

    def test_tab_after_colon(self):
        assert parse_advisor_score("ADVISOR_SCORE:\t7") == 7

    def test_trailing_whitespace(self):
        assert parse_advisor_score("ADVISOR_SCORE: 7   ") == 7


class TestParseAdvisorScoreMissing:
    """When the line is missing, the helper returns None."""

    def test_no_line_at_all(self):
        assert parse_advisor_score("no score line") is None

    def test_empty_string(self):
        assert parse_advisor_score("") is None

    def test_similar_but_wrong_prefix(self):
        # The marker must be ADVISOR_SCORE: exactly (case-sensitive).
        assert parse_advisor_score("ADVISOR SCORE: 7") is None
        assert parse_advisor_score("advisor_score: 7") is None
        # The reflector-only SCORE: marker is not the advisor marker.
        assert parse_advisor_score("SCORE: 7") is None

    def test_substring_match_fails(self):
        # The regex is anchored; substring matches are not accepted.
        assert parse_advisor_score("XADVISOR_SCORE: 7") is None
        assert parse_advisor_score("ADVISOR_SCORE: 7Y") is None

    def test_leading_spaces_breaks_anchored_match(self):
        # Anchored at line start; leading whitespace breaks the match.
        assert parse_advisor_score("    ADVISOR_SCORE: 7") is None


class TestParseAdvisorScoreMalformed:
    """When the line is present but the value is malformed, return None."""

    def test_empty_after_colon(self):
        assert parse_advisor_score("ADVISOR_SCORE:") is None

    def test_only_spaces_after_colon(self):
        assert parse_advisor_score("ADVISOR_SCORE:   ") is None

    def test_non_numeric_after_colon(self):
        assert parse_advisor_score("ADVISOR_SCORE: eight") is None

    def test_float_is_rejected(self):
        # Integer-only by design; floats are rejected.
        assert parse_advisor_score("ADVISOR_SCORE: 8.5") is None
        assert parse_advisor_score("ADVISOR_SCORE: 0.92") is None

    def test_negative_integer_rejected(self):
        # The regex does not allow a leading sign.
        assert parse_advisor_score("ADVISOR_SCORE: -5") is None


class TestParseAdvisorScoreType:
    """Return type and edge cases for the helper."""

    def test_non_string_input_returns_none(self):
        # Defensive against non-string input.
        assert parse_advisor_score(None) is None  # type: ignore[arg-type]
        assert parse_advisor_score(8) is None  # type: ignore[arg-type]

    def test_returned_value_is_int_or_none(self):
        assert isinstance(parse_advisor_score("ADVISOR_SCORE: 8"), int)
        assert parse_advisor_score("not a line") is None
        assert parse_advisor_score("") is None


# ────────────────────────────────────────────────────────────────────
# parse_advisor_issues (DELTA 2: bullet list from ADVISOR_ISSUES: block)
# ────────────────────────────────────────────────────────────────────


class TestParseAdvisorIssuesHappyPath:
    """The parser recognises the ``ADVISOR_ISSUES:`` block."""

    def test_simple_dash_bullets(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "- issue 2"
        )
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]

    def test_mixed_bullet_markers(self):
        # The parser tolerates "-", "*", and "•" markers; the order
        # is preserved.
        text = (
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "* issue 2\n"
            "\u2022 issue 3"
        )
        assert parse_advisor_issues(text) == ["issue 1", "issue 2", "issue 3"]

    def test_documented_example(self):
        # The M5 contract example: ADVISOR_ISSUES: followed by "- ",
        # "* ", and "\u2022 " bullet markers.
        text = (
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "* issue 2\n"
            "\u2022 issue 3"
        )
        assert parse_advisor_issues(text) == ["issue 1", "issue 2", "issue 3"]

    def test_multiple_bullets_with_dash(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- first\n"
            "- second\n"
            "- third"
        )
        assert parse_advisor_issues(text) == ["first", "second", "third"]


class TestParseAdvisorIssuesEmpty:
    """Empty input returns an empty list."""

    def test_empty_string(self):
        assert parse_advisor_issues("") == []

    def test_non_string_input_returns_empty_list(self):
        assert parse_advisor_issues(None) == []  # type: ignore[arg-type]

    def test_no_header_at_all(self):
        # When the ADVISOR_ISSUES: header is missing, the parser
        # returns an empty list.
        assert parse_advisor_issues("- issue 1\n- issue 2") == []

    def test_header_only_no_bullets(self):
        # The header is present but the body is empty (header is the
        # last line of the text).
        assert parse_advisor_issues("ADVISOR_ISSUES:") == []

    def test_header_followed_by_blank_line(self):
        # A blank line after the header terminates the body.
        assert parse_advisor_issues("ADVISOR_ISSUES:\n\n") == []


class TestParseAdvisorIssuesEmptyBullets:
    """Empty bullets are skipped."""

    def test_dash_only_is_skipped(self):
        text = "ADVISOR_ISSUES:\n- \n- real issue"
        assert parse_advisor_issues(text) == ["real issue"]

    def test_star_only_is_skipped(self):
        text = "ADVISOR_ISSUES:\n* \n* real issue"
        assert parse_advisor_issues(text) == ["real issue"]

    def test_unicode_bullet_only_is_skipped(self):
        text = "ADVISOR_ISSUES:\n\u2022 \n\u2022 real issue"
        assert parse_advisor_issues(text) == ["real issue"]

    def test_mix_of_empty_and_real_bullets(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- \n"
            "- real issue 1\n"
            "* \n"
            "- real issue 2"
        )
        assert parse_advisor_issues(text) == ["real issue 1", "real issue 2"]


class TestParseAdvisorIssuesWhitespace:
    """The parser trims surrounding whitespace."""

    def test_trims_leading_whitespace_on_issue(self):
        text = "ADVISOR_ISSUES:\n-    lots of leading spaces\n- normal"
        assert parse_advisor_issues(text) == [
            "lots of leading spaces",
            "normal",
        ]

    def test_trims_trailing_whitespace(self):
        text = "ADVISOR_ISSUES:\n- issue 1   \n- issue 2\t"
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]

    def test_tabs_as_whitespace(self):
        text = "ADVISOR_ISSUES:\n-\tissue 1\n-\tissue 2"
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]


class TestParseAdvisorIssuesRejectsNonBullets:
    """Non-bullet lines are skipped (not errors)."""

    def test_text_without_marker_is_skipped(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- real issue\n"
            "this is plain text, not a bullet\n"
            "- another real issue"
        )
        assert parse_advisor_issues(text) == [
            "real issue",
            "another real issue",
        ]

    def test_dash_without_whitespace_is_not_a_bullet(self):
        # "-word" (no whitespace after the marker) is not a bullet.
        text = "ADVISOR_ISSUES:\n-word\n- real issue"
        assert parse_advisor_issues(text) == ["real issue"]


class TestParseAdvisorIssuesBlockTermination:
    """The block ends at the first blank line."""

    def test_blank_line_terminates_block(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "- issue 2\n"
            "\n"
            "Some unrelated text after the blank line."
        )
        # The unrelated text is NOT part of the issues block; it is
        # excluded by the blank-line terminator.
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]

    def test_continues_until_end_of_string_without_blank_line(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- issue 1\n"
            "- issue 2\n"
            "no blank line at the end"
        )
        # "no blank line at the end" is not a bullet and is skipped.
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]


class TestParseAdvisorIssuesMultipleBlocks:
    """When multiple ``ADVISOR_ISSUES:`` blocks exist, the LAST one wins."""

    def test_last_block_wins(self):
        text = (
            "ADVISOR_ISSUES:\n"
            "- first issue\n"
            "ADVISOR_ISSUES:\n"
            "- second issue"
        )
        assert parse_advisor_issues(text) == ["second issue"]


class TestParseAdvisorIssuesType:
    """Return type and edge cases for the helper."""

    def test_returns_list(self):
        result = parse_advisor_issues("ADVISOR_ISSUES:\n- x")
        assert isinstance(result, list)

    def test_returns_list_of_strings(self):
        result = parse_advisor_issues("ADVISOR_ISSUES:\n- x\n- y")
        for item in result:
            assert isinstance(item, str)


class TestParseAdvisorIssuesPipelinePackageExport:
    """The helper is re-exported from the :mod:`moaxy.pipeline` package."""

    def test_pipeline_package_re_exports_parse_advisor_score(self):
        from moaxy.pipeline import (
            parse_advisor_score as PkgParseAdvisorScore,
        )

        assert PkgParseAdvisorScore is parse_advisor_score

    def test_pipeline_package_re_exports_parse_advisor_issues(self):
        from moaxy.pipeline import (
            parse_advisor_issues as PkgParseAdvisorIssues,
        )

        assert PkgParseAdvisorIssues is parse_advisor_issues


