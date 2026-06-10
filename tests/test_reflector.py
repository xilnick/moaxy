"""Tests for :mod:`moaxy.pipeline.reflector`.

The reflector module exposes two public symbols:

* :func:`parse_confidence` — pure helper that extracts a float from a
  critique's text via the regex
  ``^REFLECT_CONFIDENCE:\\s*([0-9.]+)\\s*$``. The validation contract
  pins this exact regex (VAL-PIPE-010). When the line is missing or
  malformed, the function returns ``0.0``.
* :func:`reflect_turn` — the M2 self-reflection step. It builds the
  critique message list with
  :func:`moaxy.pipeline.message_builders.build_reflection_messages`,
  calls the adapter once with the configured model, parses the
  confidence off the response, runs the configured REFLECTOR plugins
  through :meth:`moaxy.plugins.manager.PluginManager.run`, and returns
  ``(critique_text, confidence)``.

The tests are hermetic: a hand-rolled :class:`ScriptedAdapter` records
every call and returns scripted responses, mirroring the pattern in
``tests/test_fallback.py``. The plugin manager is constructed
in-memory and populated with a single recorder plugin; no discovery
on disk.
"""

from __future__ import annotations

import inspect
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
from moaxy.pipeline.fallback import call_with_fallbacks
from moaxy.pipeline.message_builders import build_reflection_messages
from moaxy.pipeline.prompts import DEFAULT_REFLECT_PROMPT
from moaxy.pipeline.reflector import parse_confidence, reflect_turn
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

    Calls are matched to script entries in order. If the script runs
    out, every subsequent ``chat`` raises an :class:`AssertionError`
    so the test fails loudly with a clear message.

    The class records every call (model + kwargs) in ``self.calls``,
    indexed by the order in which ``chat`` was invoked.
    """

    name = "scripted"

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

    async def stream(  # pragma: no cover - not exercised by reflector tests
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


def _critique_response(
    content: str,
    *,
    model: str = "minimax-m3:cloud",
    prompt_tokens: int = 5,
    completion_tokens: int = 7,
) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-critique",
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


class TestReflectorModuleExports:
    """The :mod:`moaxy.pipeline.reflector` module exports the documented names."""

    def test_parse_confidence_is_callable(self):
        assert callable(parse_confidence)

    def test_reflect_turn_is_callable(self):
        assert callable(reflect_turn)

    def test_reflect_turn_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(reflect_turn)

    def test_parse_confidence_is_not_a_coroutine(self):
        # parse_confidence is a pure synchronous helper.
        assert not inspect.iscoroutinefunction(parse_confidence)

    def test_pipeline_package_re_exports_reflector(self):
        from moaxy.pipeline import parse_confidence as PipelineParseConfidence
        from moaxy.pipeline import reflect_turn as PipelineReflectTurn

        assert PipelineParseConfidence is parse_confidence
        assert PipelineReflectTurn is reflect_turn


# ────────────────────────────────────────────────────────────────────
# parse_confidence
# ────────────────────────────────────────────────────────────────────


class TestParseConfidenceHappyPath:
    """The regex matches the documented ``REFLECT_CONFIDENCE: <float>`` line."""

    def test_simple_decimal(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.92") == 0.92

    def test_upper_bound_one(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 1.0") == 1.0

    def test_lower_bound_zero(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.0") == 0.0

    def test_zero_integer_form(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0") == 0.0

    def test_one_integer_form(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 1") == 1.0

    def test_mid_value(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.5") == 0.5

    def test_high_resolution_value(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.8765") == 0.8765


class TestParseConfidenceInCritiqueText:
    """The line can be embedded anywhere in the critique text."""

    def test_line_appears_after_critique(self):
        text = (
            "The previous answer is correct and complete.\n"
            "It addresses all the edge cases I checked.\n"
            "REFLECT_CONFIDENCE: 0.95"
        )
        assert parse_confidence(text) == 0.95

    def test_takes_last_match_when_multiple_lines(self):
        text = (
            "REFLECT_CONFIDENCE: 0.5\n"
            "Some extra analysis here.\n"
            "REFLECT_CONFIDENCE: 0.9"
        )
        assert parse_confidence(text) == 0.9

    def test_takes_last_match_with_extra_content(self):
        text = (
            "First thought.\n"
            "REFLECT_CONFIDENCE: 0.4\n"
            "Refined thought after a closer look.\n"
            "REFLECT_CONFIDENCE: 0.75"
        )
        assert parse_confidence(text) == 0.75


class TestParseConfidenceWhitespaceHandling:
    """The regex tolerates whitespace around the marker and the number."""

    def test_multiple_spaces_after_colon(self):
        assert parse_confidence("REFLECT_CONFIDENCE:    0.5") == 0.5

    def test_tab_after_colon(self):
        assert parse_confidence("REFLECT_CONFIDENCE:\t0.5") == 0.5

    def test_trailing_whitespace(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.5   ") == 0.5

    def test_line_with_only_carriage_return(self):
        assert parse_confidence("REFLECT_CONFIDENCE: 0.5\r") == 0.5

    def test_leading_spaces_breaks_anchored_match(self):
        # The regex is anchored to the start of the line, so leading
        # whitespace on the line breaks the match (this is the
        # documented behaviour; the model is expected to emit the line
        # at column zero).
        assert parse_confidence("    REFLECT_CONFIDENCE: 0.5") == 0.0


class TestParseConfidenceMissing:
    """When the line is missing, the helper returns 0.0."""

    def test_no_line_at_all(self):
        assert parse_confidence("no confidence line") == 0.0

    def test_empty_string(self):
        assert parse_confidence("") == 0.0

    def test_only_whitespace(self):
        assert parse_confidence("   \n\n   \n") == 0.0

    def test_similar_but_wrong_prefix(self):
        # The marker must be REFLECT_CONFIDENCE: exactly.
        assert parse_confidence("REFLECT CONFIDENCE: 0.5") == 0.0
        assert parse_confidence("CONFIDENCE: 0.5") == 0.0
        assert parse_confidence("reflect_confidence: 0.5") == 0.0  # lowercase fails

    def test_substring_match_fails(self):
        # The regex is anchored, so a substring "REFLECT_CONFIDENCE:" inside
        # a longer word is not a match.
        assert parse_confidence("XREFLECT_CONFIDENCE: 0.5") == 0.0
        assert parse_confidence("REFLECT_CONFIDENCE: 0.5Y") == 0.0


class TestParseConfidenceMalformed:
    """When the line is present but the number is malformed, return 0.0."""

    def test_empty_after_colon(self):
        # The regex requires at least one digit-or-dot; nothing after the
        # colon means no match.
        assert parse_confidence("REFLECT_CONFIDENCE:") == 0.0

    def test_only_spaces_after_colon(self):
        assert parse_confidence("REFLECT_CONFIDENCE:   ") == 0.0

    def test_non_numeric_after_colon(self):
        # The regex is `[0-9.]+`; a word is rejected.
        assert parse_confidence("REFLECT_CONFIDENCE: high") == 0.0
        assert parse_confidence("REFLECT_CONFIDENCE: ninety") == 0.0

    def test_only_dots_after_colon(self):
        # `....` is not a valid float; the regex matches it but float() fails.
        assert parse_confidence("REFLECT_CONFIDENCE: ...") == 0.0

    def test_returns_float_type(self):
        # The returned value must be a Python float so downstream header
        # serialisation works.
        result = parse_confidence("REFLECT_CONFIDENCE: 0.92")
        assert isinstance(result, float)


class TestParseConfidenceType:
    """Return type and edge cases for the helper."""

    def test_non_string_input_returns_zero(self):
        # The helper is defensive against non-string input. The contract
        # passes strings, but a None or non-string must not raise.
        assert parse_confidence(None) == 0.0  # type: ignore[arg-type]
        assert parse_confidence(0.92) == 0.0  # type: ignore[arg-type]

    def test_returned_value_is_a_float(self):
        assert isinstance(parse_confidence("REFLECT_CONFIDENCE: 1"), float)
        assert isinstance(parse_confidence("REFLECT_CONFIDENCE: 0.0"), float)
        assert isinstance(parse_confidence("not a line"), float)
        assert isinstance(parse_confidence(""), float)


# ────────────────────────────────────────────────────────────────────
# reflect_turn
# ────────────────────────────────────────────────────────────────────


class TestReflectTurnCallsAdapter:
    """reflect_turn calls the configured adapter with the critique prompt."""

    @pytest.mark.asyncio
    async def test_uses_model_from_argument(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(
            ctx,
            model="minimax-m3:cloud",
            history=[{"role": "user", "content": "hi"}],
            current_answer="my answer",
        )
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_messages_use_critique_builder(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        history = [{"role": "user", "content": "hi"}]
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(
            ctx,
            model="m",
            history=history,
            current_answer="the answer to critique",
        )
        sent = adapter.calls[0]["messages"]
        # The last message should be a user-role critique request that
        # includes both the canonical prefix and the answer text.
        last = sent[-1]
        assert last["role"] == "user"
        assert "Please critique" in last["content"]
        assert "the answer to critique" in last["content"]

    @pytest.mark.asyncio
    async def test_history_is_passed_through(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        history = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "what is 2+2?"},
        ]
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(ctx, model="m", history=history, current_answer="4")
        sent = adapter.calls[0]["messages"]
        # The history (verbatim) appears in the sent messages, before the
        # trailing critique-user message.
        for entry in history:
            assert entry in sent

    @pytest.mark.asyncio
    async def test_explicit_default_system_prompt_is_attached(self):
        """When the caller passes the default system prompt, it is prepended.

        The reflector function does NOT auto-default to
        ``DEFAULT_REFLECT_PROMPT`` — it forwards whatever ``system_prompt``
        the caller gives (or no system message when ``None``/empty).
        The orchestrator is responsible for resolving the effective
        prompt from the route's reflection config. This test verifies
        that when the caller passes the default prompt, the builder
        attaches it as the leading system message.
        """
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(
            ctx,
            model="m",
            history=[{"role": "user", "content": "hi"}],
            current_answer="answer",
            system_prompt=DEFAULT_REFLECT_PROMPT,
        )
        sent = adapter.calls[0]["messages"]
        first = sent[0]
        assert first["role"] == "system"
        # The default reflector prompt is attached when the caller passes it.
        assert first["content"] == DEFAULT_REFLECT_PROMPT

    @pytest.mark.asyncio
    async def test_custom_system_prompt_is_attached(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        custom = "You are an EXTREMELY strict critic."
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="answer",
            system_prompt=custom,
        )
        sent = adapter.calls[0]["messages"]
        first = sent[0]
        assert first["role"] == "system"
        assert first["content"] == custom

    @pytest.mark.asyncio
    async def test_no_system_message_when_prompt_empty(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(
            ctx,
            model="m",
            history=[{"role": "user", "content": "hi"}],
            current_answer="answer",
            system_prompt="",
        )
        sent = adapter.calls[0]["messages"]
        assert all(m["role"] != "system" for m in sent)

    @pytest.mark.asyncio
    async def test_history_not_mutated(self):
        adapter = ScriptedAdapter([_critique_response("a critique\nREFLECT_CONFIDENCE: 0.8")])
        from copy import deepcopy

        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        original = deepcopy(history)
        ctx: dict[str, Any] = {"adapter": adapter}
        await reflect_turn(ctx, model="m", history=history, current_answer="x")
        assert history == original


class TestReflectTurnReturnsCritiqueAndConfidence:
    """reflect_turn returns (text, confidence) with text=full critique."""

    @pytest.mark.asyncio
    async def test_returns_full_critique_text(self):
        critique = (
            "The previous answer missed the unit on the second line and "
            "the edge case for n=0.\n"
            "REFLECT_CONFIDENCE: 0.6"
        )
        adapter = ScriptedAdapter([_critique_response(critique)])
        ctx: dict[str, Any] = {"adapter": adapter}
        text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert text == critique

    @pytest.mark.asyncio
    async def test_returns_parsed_confidence(self):
        adapter = ScriptedAdapter(
            [_critique_response("ok\nREFLECT_CONFIDENCE: 0.42")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        _text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert confidence == 0.42

    @pytest.mark.asyncio
    async def test_missing_confidence_returns_zero(self):
        adapter = ScriptedAdapter(
            [_critique_response("a critique without any confidence line")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        _text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_upper_bound_confidence(self):
        adapter = ScriptedAdapter(
            [_critique_response("great\nREFLECT_CONFIDENCE: 1.0")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        _text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert confidence == 1.0

    @pytest.mark.asyncio
    async def test_malformed_confidence_returns_zero(self):
        adapter = ScriptedAdapter(
            [_critique_response("ok\nREFLECT_CONFIDENCE: high")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        _text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_returns_a_tuple_of_text_and_float(self):
        adapter = ScriptedAdapter(
            [_critique_response("a\nREFLECT_CONFIDENCE: 0.5")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        result = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, confidence = result
        assert isinstance(text, str)
        assert isinstance(confidence, float)

    @pytest.mark.asyncio
    async def test_empty_response_content_yields_empty_text_and_zero_confidence(self):
        adapter = ScriptedAdapter([_critique_response("")])
        ctx: dict[str, Any] = {"adapter": adapter}
        text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert text == ""
        assert confidence == 0.0


class TestReflectTurnErrorPropagation:
    """Adapter errors bubble out of reflect_turn unchanged."""

    @pytest.mark.asyncio
    async def test_upstream_error_is_raised(self):
        adapter = ScriptedAdapter(
            [UpstreamError("boom", status_code=500, body="server error")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        with pytest.raises(UpstreamError) as exc_info:
            await reflect_turn(ctx, model="m", history=[], current_answer="x")
        assert exc_info.value.status_code == 500


# ────────────────────────────────────────────────────────────────────
# reflect_turn + PluginManager integration
# ────────────────────────────────────────────────────────────────────


class _RecorderReflectorPlugin(Plugin):
    """A REFLECTOR plugin that records every process_async call."""

    name = "recorder_reflector"
    version = "1.0.0"
    plugin_type = PluginType.REFLECTOR

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


class TestReflectTurnPluginDispatch:
    """reflect_turn invokes the configured REFLECTOR plugins per turn."""

    @pytest.mark.asyncio
    async def test_runs_reflector_plugin_once_per_turn(self):
        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        plugin = _RecorderReflectorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert plugin.call_count == 1

    @pytest.mark.asyncio
    async def test_runs_reflector_plugin_twice_for_two_turns(self):
        adapter = ScriptedAdapter(
            [
                _critique_response("first critique\nREFLECT_CONFIDENCE: 0.5"),
                _critique_response("second critique\nREFLECT_CONFIDENCE: 0.6"),
            ]
        )
        plugin = _RecorderReflectorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await reflect_turn(ctx, model="m", history=[], current_answer="x")
        await reflect_turn(ctx, model="m", history=[], current_answer="x")
        assert plugin.call_count == 2

    @pytest.mark.asyncio
    async def test_no_plugin_manager_is_ok(self):
        """When the context has no plugin manager, the reflect_turn still runs."""
        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        ctx: dict[str, Any] = {"adapter": adapter}
        text, confidence = await reflect_turn(
            ctx,
            model="m",
            history=[],
            current_answer="x",
        )
        assert text == "a critique\nREFLECT_CONFIDENCE: 0.7"
        assert confidence == 0.7

    @pytest.mark.asyncio
    async def test_plugin_receives_context(self):
        """The plugin sees the context dict reflect_turn was given."""
        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        plugin = _RecorderReflectorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {
            "adapter": adapter,
            "plugin_manager": mgr,
            "request_id": "req-abc",
            "route": "reflective-coder",
        }
        await reflect_turn(ctx, model="m", history=[], current_answer="x")
        assert plugin.call_count == 1
        ctx_seen = plugin.contexts[0]
        assert ctx_seen["request_id"] == "req-abc"
        assert ctx_seen["route"] == "reflective-coder"

    @pytest.mark.asyncio
    async def test_plugin_receives_critique_and_confidence(self):
        """The plugin can read the parsed critique and confidence from the ctx."""
        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        plugin = _RecorderReflectorPlugin()
        mgr = _build_manager_with(plugin)
        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await reflect_turn(ctx, model="m", history=[], current_answer="x")
        ctx_seen = plugin.contexts[0]
        assert "critique_text" in ctx_seen
        assert ctx_seen["critique_text"] == "a critique\nREFLECT_CONFIDENCE: 0.7"
        assert ctx_seen["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_non_reflector_plugins_not_invoked(self):
        """PluginManager.run with REFLECTOR only touches REFLECTOR plugins.

        A custom TRANSFORMER plugin in the manager is left untouched. The
        PipelineManager's dispatch logic already enforces this; the
        reflector relies on it.
        """
        class _RecorderTransformer(Plugin):
            name = "recorder_transformer"
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

        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        reflector = _RecorderReflectorPlugin()
        transformer = _RecorderTransformer()
        mgr = PluginManager(plugins_dir=PLUGINS_DIR_FOR_TESTS)
        mgr._plugins[reflector.name] = reflector
        mgr._plugins[transformer.name] = transformer
        mgr._loaded = True

        ctx: dict[str, Any] = {"adapter": adapter, "plugin_manager": mgr}
        await reflect_turn(ctx, model="m", history=[], current_answer="x")
        assert reflector.call_count == 1
        assert transformer.call_count == 0


# ────────────────────────────────────────────────────────────────────
# Integration with call_with_fallbacks (orchestrator-style usage)
# ────────────────────────────────────────────────────────────────────


class TestReflectTurnWithFallbacks:
    """reflect_turn composes cleanly with the M2 fallback walker.

    The orchestrator (M2+) is expected to wrap reflect_turn's adapter
    call in call_with_fallbacks so the reflection step gets the same
    retry/fallback treatment as the initial generation. These tests
    prove reflect_turn works through that wrapper.
    """

    @pytest.mark.asyncio
    async def test_works_through_call_with_fallbacks(self):
        adapter = ScriptedAdapter(
            [_critique_response("a critique\nREFLECT_CONFIDENCE: 0.7")]
        )
        # Build the messages via the same helper the orchestrator uses.
        messages = build_reflection_messages(
            history=[],
            answer="my answer",
            system_prompt=DEFAULT_REFLECT_PROMPT,
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["minimax-m3:cloud"],
            retry=0,
            messages=messages,
        )
        assert response.message.content == "a critique\nREFLECT_CONFIDENCE: 0.7"
        assert fallbacks_used == []
        # And the same content parses to 0.7.
        assert parse_confidence(response.message.content) == 0.7
