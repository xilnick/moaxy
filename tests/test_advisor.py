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
from moaxy.pipeline.advisor import advisor_turn, parse_advisor_response
from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_advisor_revision_messages,
)
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
        decision, text = parse_advisor_response("ADVISOR_APPROVE")
        assert decision == "approve"
        assert text is None

    def test_approve_with_trailing_whitespace(self):
        decision, text = parse_advisor_response("ADVISOR_APPROVE   \n")
        assert decision == "approve"
        assert text is None

    def test_approve_with_explanatory_text(self):
        # The advisor may include explanation; any text alongside the
        # marker is treated as approval.
        decision, text = parse_advisor_response(
            "The previous answer is good.\nADVISOR_APPROVE"
        )
        assert decision == "approve"
        assert text is None

    def test_approve_uppercase_only(self):
        # The marker is case-sensitive; lowercase variants are not approve.
        assert parse_advisor_response("advisor_approve") == ("revise", "advisor_approve")
        assert parse_advisor_response("Advisor_Approve") == ("revise", "Advisor_Approve")

    def test_approve_substring_is_treated_as_approve(self):
        # Substring matching is the historical contract (the parser uses
        # ``ADVISOR_APPROVE in text``). A substring occurrence anywhere
        # in the text is treated as approval. This matches the existing
        # orchestrator behaviour for backwards compatibility.
        assert parse_advisor_response("XADVISOR_APPROVE") == ("approve", None)


class TestParseAdvisorRevise:
    """The parser returns the text after ``ADVISOR_REVISE:`` on a revise."""

    def test_revise_with_text(self):
        decision, text = parse_advisor_response("ADVISOR_REVISE: better answer")
        assert decision == "revise"
        assert text == "better answer"

    def test_revise_with_leading_newline(self):
        decision, text = parse_advisor_response(
            "ADVISOR_REVISE:\nthis is the new answer"
        )
        assert decision == "revise"
        assert text == "this is the new answer"

    def test_revise_with_multiline_text(self):
        decision, text = parse_advisor_response(
            "ADVISOR_REVISE: line1\nline2\nline3"
        )
        assert decision == "revise"
        assert text == "line1\nline2\nline3"

    def test_revise_with_explanatory_text_before_marker(self):
        # The advisor may explain its reasoning before the marker; only
        # the text after the marker is treated as the revised answer.
        decision, text = parse_advisor_response(
            "Here is my critique of the previous answer.\n"
            "ADVISOR_REVISE: the final improved answer"
        )
        assert decision == "revise"
        assert text == "the final improved answer"

    def test_revise_text_is_stripped(self):
        # Leading and trailing whitespace on the revised text is removed.
        decision, text = parse_advisor_response("ADVISOR_REVISE:    \n answer \n  ")
        assert decision == "revise"
        assert text == "answer"

    def test_revise_with_empty_text_after_marker(self):
        # The marker may be present but the text is empty; the decision
        # is still "revise" and the text is an empty string.
        decision, text = parse_advisor_response("ADVISOR_REVISE:")
        assert decision == "revise"
        assert text == ""


class TestParseAdvisorMissing:
    """When neither marker is present, the helper treats the text as a revise."""

    def test_plain_text_treated_as_revise(self):
        decision, text = parse_advisor_response("just a plain response")
        assert decision == "revise"
        assert text == "just a plain response"

    def test_empty_string_treated_as_revise(self):
        decision, text = parse_advisor_response("")
        assert decision == "revise"
        assert text == ""

    def test_only_whitespace_treated_as_revise(self):
        decision, text = parse_advisor_response("   \n\n  ")
        assert decision == "revise"
        assert text == ""

    def test_non_string_input_treated_as_revise(self):
        # The helper is defensive against non-string input.
        decision, text = parse_advisor_response(None)  # type: ignore[arg-type]
        assert decision == "revise"
        assert text is None or text == ""


class TestParseAdvisorReturnsTuple:
    """The helper always returns a 2-tuple ``(decision, text)``."""

    def test_returns_tuple_of_str_and_optional(self):
        result = parse_advisor_response("ADVISOR_APPROVE")
        assert isinstance(result, tuple)
        assert len(result) == 2
        decision, text = result
        assert isinstance(decision, str)
        assert decision in {"approve", "revise"}

    def test_approve_text_is_none(self):
        _decision, text = parse_advisor_response("ADVISOR_APPROVE")
        assert text is None

    def test_revise_text_is_string(self):
        _decision, text = parse_advisor_response("ADVISOR_REVISE: x")
        assert isinstance(text, str)


class TestParseAdvisorPrecedence:
    """When both markers appear, the parser picks the first one in the text."""

    def test_approve_first_wins(self):
        decision, text = parse_advisor_response(
            "ADVISOR_APPROVE\nADVISOR_REVISE: ignored"
        )
        assert decision == "approve"
        assert text is None

    def test_revise_only_is_revise(self):
        # When the revise marker appears but no approve marker does, the
        # decision is revise. (The reverse case ``revise then approve``
        # resolves to approve because the parser checks the approve
        # marker first; that's the documented order.)
        decision, text = parse_advisor_response(
            "ADVISOR_REVISE: this one"
        )
        assert decision == "revise"
        assert text == "this one"


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
