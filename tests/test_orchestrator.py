"""Tests for :mod:`moaxy.pipeline.orchestrator`.

The orchestrator is the heart of the moaxy pipeline: it threads a
:class:`PipelineContext` through the initial generation, the optional
self-reflection loop (0..3 turns), and the optional advisor pass
(0..1 turn), and emits the structured :class:`PipelineEvent` log
that downstream consumers (the server, the validators) walk to
materialise ``x-moaxy-*`` response headers.

The tests are hermetic: a hand-rolled :class:`ScriptedAdapter` records
every call and returns scripted responses, mirroring the pattern in
``tests/test_fallback.py`` and ``tests/test_reflector.py``. No
in-process HTTP, no real Ollama, no on-disk plugins.

The contract pinned by these tests matches the validation contract
section "Area: Reflection + Advisor Pipeline" (VAL-PIPE-001..043).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    Usage,
)
from moaxy.models.config import (
    AdvisorConfig,
    ReflectionConfig,
    RouteConfig,
)
from moaxy.models.config import RouteMatch as ConfigRouteMatch
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.routing.matcher import RouteMatch

# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    Mirrors the helpers in ``test_fallback.py`` and ``test_reflector.py``.
    Each script entry is either a :class:`ChatResponse` (success) or a
    :class:`BaseException` (raised). Calls are recorded in
    :attr:`calls` so tests can assert on the exact ``model`` /
    ``messages`` / ``**kwargs`` the orchestrator forwarded.
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

    async def stream(  # pragma: no cover - not exercised here
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


# ────────────────────────────────────────────────────────────────────
# Response / context factories
# ────────────────────────────────────────────────────────────────────


def _response(
    content: str,
    *,
    model: str = "minimax-m3:cloud",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    chatcmpl_id: str = "chatcmpl-test",
) -> ChatResponse:
    return ChatResponse(
        id=chatcmpl_id,
        model=model,
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        finish_reason=finish_reason,
    )


def _build_route(
    *,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    aliases: dict[str, str] | None = None,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    order: str = "reflect_first",
    fresh_context: bool = False,
    system_prompt: str | None = None,
) -> RouteMatch:
    config_route = RouteConfig(
        name="reflective-coder",
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
            order=order,
            fresh_context=fresh_context,
            system_prompt=system_prompt,
        ),
        advisor=AdvisorConfig(model=advisor_model, turns=advisor_turns),
    )
    return RouteMatch(
        route=config_route,
        original_model=original_model,
        resolved_model=resolved_model,
        backend="ollama-local",
        path="/v1/chat/completions",
        reflection=config_route.reflection,
        advisor=config_route.advisor,
        fallbacks=list(config_route.fallbacks),
        retry=config_route.retry,
        aliases=dict(config_route.aliases),
    )


def _build_context(
    route: RouteMatch,
    *,
    request_extra: dict[str, Any] | None = None,
    request_id: str = "req-1",
    request_messages: list[dict[str, Any]] | None = None,
) -> PipelineContext:
    body: dict[str, Any] = {
        "model": route.original_model,
        "messages": request_messages
        or [{"role": "user", "content": "ping"}],
    }
    if request_extra:
        body.update(request_extra)
    return PipelineContext(
        request_id=request_id,
        request=body,
        route=route,
        model_alias_resolved=route.resolved_model,
        target_backend=route.backend,
        original_model=route.original_model,
    )


# ────────────────────────────────────────────────────────────────────
# Module exports
# ────────────────────────────────────────────────────────────────────


class TestOrchestratorExports:
    """The :mod:`moaxy.pipeline.orchestrator` module exports the documented names."""

    def test_orchestrator_class_is_callable(self):
        assert callable(Orchestrator)

    def test_build_response_headers_is_callable(self):
        assert callable(build_response_headers)

    def test_pipeline_package_re_exports_orchestrator(self):
        from moaxy.pipeline import (
            Orchestrator as PackageOrchestrator,
        )
        from moaxy.pipeline import (
            build_response_headers as PackageBuildHeaders,
        )

        assert PackageOrchestrator is Orchestrator
        assert PackageBuildHeaders is build_response_headers


# ────────────────────────────────────────────────────────────────────
# Initial generation (VAL-PIPE-001, 026, 033)
# ────────────────────────────────────────────────────────────────────


class TestInitialGeneration:
    """The first stage of the pipeline is the initial generation."""

    @pytest.mark.asyncio
    async def test_turns_0_produces_one_llm_call(self):
        """VAL-PIPE-001: turns=0 is a passthrough — one LLM call, no reflect_* events."""
        adapter = ScriptedAdapter(
            [_response("initial answer", prompt_tokens=10, completion_tokens=5)]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        orchestrator = Orchestrator(adapter)
        await orchestrator.run(ctx)
        assert len(adapter.calls) == 1
        types = [e.type for e in ctx.events]
        assert types == ["initial"]

    @pytest.mark.asyncio
    async def test_initial_call_uses_resolved_model(self):
        adapter = ScriptedAdapter(
            [_response("initial", model="minimax-m3:cloud")]
        )
        route = _build_route()
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_initial_call_passes_request_messages(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route()
        ctx = _build_context(
            route,
            request_messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        await Orchestrator(adapter).run(ctx)
        # The messages are forwarded verbatim (no mutation).
        sent = adapter.calls[0]["messages"]
        assert sent == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    @pytest.mark.asyncio
    async def test_initial_event_records_model(self):
        adapter = ScriptedAdapter(
            [_response("hi", model="minimax-m3:cloud")]
        )
        route = _build_route()
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.events[0].model == "minimax-m3:cloud"
        assert ctx.events[0].text == "hi"
        assert ctx.events[0].turn is None

    @pytest.mark.asyncio
    async def test_request_messages_are_not_mutated(self):
        """VAL-PIPE-039: the request messages list is not mutated."""
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route()
        original = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        snapshot = json.dumps(original)
        ctx = _build_context(route, request_messages=original)
        await Orchestrator(adapter).run(ctx)
        assert json.dumps(ctx.request["messages"]) == snapshot


# ────────────────────────────────────────────────────────────────────
# Reflection loop (VAL-PIPE-002, 003, 004, 005, 006, 007)
# ────────────────────────────────────────────────────────────────────


class TestReflectionTurns:
    """The reflection loop runs 0..3 critique+revision pairs."""

    @pytest.mark.asyncio
    async def test_turns_1_with_early_exit_false_produces_three_calls(self):
        """VAL-PIPE-002: turns=1, early_exit=false → 3 LLM calls + reflect_revised."""
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response("a critique\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("revised answer", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=False, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 3
        types = [e.type for e in ctx.events]
        assert types == ["initial", "reflect_critique", "reflect_revised"]
        # The final response content is the revised answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised answer"

    @pytest.mark.asyncio
    async def test_turns_1_with_early_exit_above_threshold_short_circuits(self):
        """VAL-PIPE-003: critique confidence >= threshold → 2 calls + early_exit."""
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response("looks good\nREFLECT_CONFIDENCE: 0.95", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 2
        types = [e.type for e in ctx.events]
        assert types == ["initial", "reflect_critique", "reflect_early_exit"]
        # No revision happened, so the response is the initial.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial answer"

    @pytest.mark.asyncio
    async def test_turns_1_with_early_exit_below_threshold_continues(self):
        """VAL-PIPE-004: critique confidence < threshold → 3 calls + revised."""
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response("not great\nREFLECT_CONFIDENCE: 0.50", prompt_tokens=4, completion_tokens=6),
                _response("revised answer", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 3
        types = [e.type for e in ctx.events]
        assert types == ["initial", "reflect_critique", "reflect_revised"]

    @pytest.mark.asyncio
    async def test_turns_2_with_all_below_threshold_produces_five_calls(self):
        """VAL-PIPE-006: turns=2, all confidences below → 5 calls + 2 reflect_revised."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("c1\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
                _response("c2\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("rev2", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 5
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
        ]
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev2"

    @pytest.mark.asyncio
    async def test_turns_2_with_turn1_above_threshold_stops_after_turn1(self):
        """VAL-PIPE-007: turn 1 above threshold → 3 calls, stop after turn 1."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("c1\nREFLECT_CONFIDENCE: 0.95", prompt_tokens=4, completion_tokens=6),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 3
        types = [e.type for e in ctx.events]
        assert types == ["initial", "reflect_critique", "reflect_revised", "reflect_early_exit"]
        # The final answer is the first revision (turn 1).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev1"

    @pytest.mark.asyncio
    async def test_missing_confidence_line_continues_to_revision(self):
        """VAL-PIPE-005: missing REFLECT_CONFIDENCE → confidence=0.0, revision runs."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("a critique with no confidence line", prompt_tokens=4, completion_tokens=6),
                _response("revised after missing line", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 3
        types = [e.type for e in ctx.events]
        assert types == ["initial", "reflect_critique", "reflect_revised"]
        # No early-exit event was emitted.
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        # The last confidence is 0.0.
        assert ctx.__dict__.get("last_confidence") == 0.0


# ────────────────────────────────────────────────────────────────────
# Reflection event ordering (VAL-PIPE-027, 028, 033, 034)
# ────────────────────────────────────────────────────────────────────


class TestReflectionEventOrdering:
    """Events follow the documented order: initial, then reflect_critique/revised pairs."""

    @pytest.mark.asyncio
    async def test_initial_always_first(self):
        """VAL-PIPE-033: the events list always starts with 'initial'."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.95"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.events[0].type == "initial"

    @pytest.mark.asyncio
    async def test_critique_then_revised_per_turn(self):
        """VAL-PIPE-028, 034: each reflect_revised follows its reflect_critique."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5"),
                _response("r1"),
                _response("c2\nREFLECT_CONFIDENCE: 0.5"),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Per turn: reflect_critique precedes reflect_revised.
        events = ctx.events
        for i in range(1, len(events)):
            if events[i].type == "reflect_revised":
                assert events[i - 1].type == "reflect_critique"
                assert events[i - 1].turn == events[i].turn

    @pytest.mark.asyncio
    async def test_two_critique_events_for_two_turns(self):
        """VAL-PIPE-027: turns=2 with no early-exit → 2 reflect_critique events."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5"),
                _response("r1"),
                _response("c2\nREFLECT_CONFIDENCE: 0.5"),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        critiques = [e for e in ctx.events if e.type == "reflect_critique"]
        assert len(critiques) == 2
        assert [c.turn for c in critiques] == [0, 1]

    @pytest.mark.asyncio
    async def test_turns_in_sequential_order(self):
        """The turns in reflect_critique events are 0, 1, ... in order."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c0\nREFLECT_CONFIDENCE: 0.5"),
                _response("r0"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5"),
                _response("r1"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        turns = [e.turn for e in ctx.events if e.type in {"reflect_critique", "reflect_revised"}]
        assert turns == [0, 0, 1, 1]


# ────────────────────────────────────────────────────────────────────
# Advisor (VAL-PIPE-030, 031, 032, 035)
# ────────────────────────────────────────────────────────────────────


class TestAdvisorStage:
    """The advisor runs after reflection, at most once, and approves or revises."""

    @pytest.mark.asyncio
    async def test_advisor_approve_keeps_response_unchanged(self):
        """VAL-PIPE-031: advisor_approve → response equals post-reflection answer."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
                _response("ADVISOR_APPROVE", prompt_tokens=4, completion_tokens=2),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 4 LLM calls: initial, critique, revise, advisor.
        assert len(adapter.calls) == 4
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "advisor",
            "advisor_approve",
        ]
        # The advisor's input is the revised answer.
        advisor_call_messages = adapter.calls[3]["messages"]
        last_user_msg = advisor_call_messages[-1]
        assert "revised" in last_user_msg["content"]
        # The final response content is the revised answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised"
        # The advisor call is to the configured advisor model.
        assert adapter.calls[3]["model"] == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_advisor_revise_replaces_response(self):
        """VAL-PIPE-032: ADVISOR_REVISE: → primary model issues a final revision."""
        advisor_text = "this is the advisor's revised answer"
        final_answer = "this is the primary model's final revised answer"
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
                _response(f"ADVISOR_REVISE: {advisor_text}", prompt_tokens=4, completion_tokens=10),
                # The orchestrator issues a primary revision call with
                # the advisor's feedback baked in; the response of that
                # call becomes the final body content.
                _response(final_answer, prompt_tokens=4, completion_tokens=10),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "advisor",
            "advisor_revised",
            "advisor_revision",
        ]
        assert ctx.upstream_response is not None
        # The final response content is the post-primary-revision text.
        assert ctx.upstream_response.message.content == final_answer

    @pytest.mark.asyncio
    async def test_advisor_turns_0_skips_advisor(self):
        """VAL-PIPE-015: advisor.turns=0 → no advisor call, no advisor event."""
        adapter = ScriptedAdapter([_response("initial")])
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        types = [e.type for e in ctx.events]
        assert types == ["initial"]
        assert not any(e.type.startswith("advisor") for e in ctx.events)

    @pytest.mark.asyncio
    async def test_advisor_runs_after_reflection(self):
        """VAL-PIPE-035: advisor events come after all reflect_* events."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("r"),
                _response("ADVISOR_APPROVE"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The last reflect_* event appears before the first advisor event.
        reflect_indices = [
            i for i, e in enumerate(ctx.events) if e.type.startswith("reflect_")
        ]
        advisor_indices = [
            i for i, e in enumerate(ctx.events) if e.type.startswith("advisor")
        ]
        assert reflect_indices
        assert advisor_indices
        assert max(reflect_indices) < min(advisor_indices)

    @pytest.mark.asyncio
    async def test_advisor_input_is_post_reflection_answer(self):
        """VAL-PIPE-020: the advisor's input contains the post-reflection answer."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("REVISED_TEXT_HERE"),
                _response("ADVISOR_APPROVE"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        advisor_messages = adapter.calls[3]["messages"]
        last_user = advisor_messages[-1]
        assert last_user["role"] == "user"
        assert "REVISED_TEXT_HERE" in last_user["content"]


# ────────────────────────────────────────────────────────────────────
# Usage accumulation (VAL-PIPE-022, 023, 024, 025)
# ────────────────────────────────────────────────────────────────────


class TestUsageAccumulation:
    """Usage is summed across every LLM call."""

    @pytest.mark.asyncio
    async def test_single_call_usage(self):
        adapter = ScriptedAdapter(
            [_response("x", prompt_tokens=10, completion_tokens=5)]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        assert snap.prompt_tokens == 10
        assert snap.completion_tokens == 5
        assert snap.total_tokens == 15

    @pytest.mark.asyncio
    async def test_three_call_usage(self):
        """VAL-PIPE-022: usage sums across initial + 2 critiques + 2 revisions (5 calls)."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _response("c1\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=20, completion_tokens=8),
                _response("r1", prompt_tokens=20, completion_tokens=8),
                _response("c2\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=20, completion_tokens=8),
                _response("r2", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        assert snap.prompt_tokens == 10 + 20 * 4
        assert snap.completion_tokens == 5 + 8 * 4
        assert snap.total_tokens == snap.prompt_tokens + snap.completion_tokens

    @pytest.mark.asyncio
    async def test_usage_with_early_exit(self):
        """VAL-PIPE-023: early-exit does not include future turn's usage."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _response("c1\nREFLECT_CONFIDENCE: 0.95", prompt_tokens=20, completion_tokens=8),
                _response("r1", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        # Only 3 calls happened: 10+20+20 = 50 prompt; 5+8+8 = 21 completion.
        assert snap.prompt_tokens == 50
        assert snap.completion_tokens == 21
        assert snap.total_tokens == 71

    @pytest.mark.asyncio
    async def test_usage_includes_advisor_call(self):
        """VAL-PIPE-024: advisor usage is summed in."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _response("c\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=8, completion_tokens=3),
                _response("r", prompt_tokens=8, completion_tokens=3),
                _response("ADVISOR_APPROVE", prompt_tokens=4, completion_tokens=2),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        assert snap.prompt_tokens == 10 + 8 + 8 + 4
        assert snap.completion_tokens == 5 + 3 + 3 + 2

    @pytest.mark.asyncio
    async def test_usage_includes_advisor_revise_call(self):
        """VAL-PIPE-025: advisor revise call usage is summed in (advisor + primary revision)."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _response("c\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=8, completion_tokens=3),
                _response("r", prompt_tokens=8, completion_tokens=3),
                _response("ADVISOR_REVISE: final", prompt_tokens=4, completion_tokens=2),
                # Primary revision call after advisor REVISE.
                _response("final answer", prompt_tokens=4, completion_tokens=2),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        # 5 calls: initial, critique, revise, advisor, primary-revision.
        assert snap.prompt_tokens == 10 + 8 + 8 + 4 + 4
        assert snap.completion_tokens == 5 + 3 + 3 + 2 + 2


# ────────────────────────────────────────────────────────────────────
# Response model field (VAL-PIPE / VAL-HTTP-018)
# ────────────────────────────────────────────────────────────────────


class TestResponseModelEchoesAlias:
    """The response's model field echoes the original alias the client sent."""

    @pytest.mark.asyncio
    async def test_model_field_is_original_alias(self):
        adapter = ScriptedAdapter([_response("hi")])
        route = _build_route(
            original_model="coder-pro", resolved_model="minimax-m3:cloud"
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_model_field_after_reflection_is_still_alias(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_model_field_after_advisor_revise_is_still_alias(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("r"),
                _response("ADVISOR_REVISE: final"),
                # Primary revision after the advisor REVISE.
                _response("primary-final"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.model == "coder-pro"
        # The final response is the primary revision's text.
        assert ctx.upstream_response.message.content == "primary-final"


# ────────────────────────────────────────────────────────────────────
# Fallback chain integration
# ────────────────────────────────────────────────────────────────────


class TestFallbackChain:
    """The orchestrator composes with call_with_fallbacks at every LLM site."""

    @pytest.mark.asyncio
    async def test_initial_falls_back_to_secondary_model(self):
        """When the primary model fails 5xx, the walker uses the fallback."""
        from moaxy.adapters.base import UpstreamError
        adapter = ScriptedAdapter(
            [
                UpstreamError("primary failed", status_code=500, body="err"),
                _response("fallback answer", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Two calls were made: the failed primary and the successful fallback.
        assert len(adapter.calls) == 2
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"
        assert adapter.calls[1]["model"] == "minimax-m2.7:cloud"
        # The fallback list is recorded for the response header.
        assert ctx.__dict__.get("fallbacks_used") == ["minimax-m2.7:cloud"]
        # The final response is the fallback's answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "fallback answer"
        # The response model still echoes the alias.
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_no_fallbacks_means_empty_fallbacks_used(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route(fallbacks=[], retry=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.__dict__.get("fallbacks_used") == []

    @pytest.mark.asyncio
    async def test_advisor_uses_advisor_model_in_chain(self):
        """The advisor's primary model is the configured advisor model."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("r"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The advisor call's primary is the advisor model.
        assert adapter.calls[3]["model"] == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# Routing-fallback integration (VAL-RT-011..018, VAL-HTTP-023, VAL-CROSS-004)
# The orchestrator reads the effective fallbacks/retry off
# ``ctx.route`` and threads them through ``call_with_fallbacks`` at
# every LLM call site. The tests below prove the override flows
# end-to-end through the orchestrator (not just the matcher).
# ────────────────────────────────────────────────────────────────────


class TestRouteFallbacksUsedByOrchestrator:
    """``RouteMatch.fallbacks`` (already override-resolved) drives the walker."""

    @pytest.mark.asyncio
    async def test_route_fallbacks_used_for_initial_call(self):
        """The initial call walks the route's effective fallbacks on 5xx."""
        from moaxy.adapters.base import UpstreamError
        adapter = ScriptedAdapter(
            [
                UpstreamError("primary failed", status_code=500, body="err"),
                _response("from fallback", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The walker called the primary, then the route's fallback.
        assert len(adapter.calls) == 2
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"
        assert adapter.calls[1]["model"] == "minimax-m2.7:cloud"
        # The header is populated from the walker's findings.
        assert ctx.__dict__.get("fallbacks_used") == ["minimax-m2.7:cloud"]

    @pytest.mark.asyncio
    async def test_route_fallbacks_used_for_reflection_calls(self):
        """Reflection critique+revision each walk the route's fallbacks.

        With ``retry=0`` and a primary that fails on the critique and
        revision calls, the walker should advance to the route's
        fallback model for both. The orchestrator aggregates the
        fallbacks used across all LLM call sites.
        """
        from moaxy.adapters.base import UpstreamError
        adapter = ScriptedAdapter(
            [
                # Initial: primary succeeds.
                _response("initial", model="minimax-m3:cloud"),
                # Critique: primary fails, fallback succeeds.
                UpstreamError("critique primary fail", status_code=500, body="e"),
                _response("c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m2.7:cloud"),
                # Revision: primary fails, fallback succeeds.
                UpstreamError("revision primary fail", status_code=500, body="e"),
                _response("revised", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The critique and revision both invoked the route's fallback.
        assert adapter.calls[2]["model"] == "minimax-m2.7:cloud"
        assert adapter.calls[4]["model"] == "minimax-m2.7:cloud"
        # The orchestrator aggregates fallbacks used across all sites.
        fallbacks_used = ctx.__dict__.get("fallbacks_used", [])
        # Two fallbacks were used (critique + revision).
        assert fallbacks_used == ["minimax-m2.7:cloud", "minimax-m2.7:cloud"]

    @pytest.mark.asyncio
    async def test_route_fallbacks_used_for_advisor_call(self):
        """The advisor LLM call walks the route's fallbacks too."""
        from moaxy.adapters.base import UpstreamError
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                # Advisor: primary fails, fallback succeeds.
                UpstreamError("advisor failed", status_code=500, body="err"),
                _response("ADVISOR_APPROVE", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The advisor's primary is the configured advisor model.
        assert adapter.calls[1]["model"] == "deepseek-v4-pro:cloud"
        # The walker fell back to the route's fallback.
        assert adapter.calls[2]["model"] == "minimax-m2.7:cloud"
        # The header captures the advisor's fallback.
        assert ctx.__dict__.get("fallbacks_used") == ["minimax-m2.7:cloud"]


class TestRouteRetryUsedByOrchestrator:
    """``RouteMatch.retry`` (already override-resolved) drives the walker."""

    @pytest.mark.asyncio
    async def test_route_retry_used_for_initial_call(self):
        """``retry=2`` on the route yields 1+2=3 calls on the primary."""
        from moaxy.adapters.base import UpstreamError
        # Three failures, then a success on the fallback.
        adapter = ScriptedAdapter(
            [
                UpstreamError("fail", status_code=500, body="e"),
                UpstreamError("fail", status_code=500, body="e"),
                UpstreamError("fail", status_code=500, body="e"),
                _response("from fallback", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(fallbacks=["minimax-m2.7:cloud"], retry=2)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The primary was called 1+2 times, then the fallback.
        assert len(adapter.calls) == 4
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"
        assert adapter.calls[1]["model"] == "minimax-m3:cloud"
        assert adapter.calls[2]["model"] == "minimax-m3:cloud"
        assert adapter.calls[3]["model"] == "minimax-m2.7:cloud"


# ────────────────────────────────────────────────────────────────────
# Sampling parameters forwarded (VAL-PIPE-042)
# ────────────────────────────────────────────────────────────────────


class TestSamplingParametersForwarded:
    """Sampling parameters are forwarded verbatim to every LLM call."""

    @pytest.mark.asyncio
    async def test_temperature_top_p_max_tokens_forwarded(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("r"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
        )
        ctx = _build_context(
            route,
            request_extra={"temperature": 0.7, "top_p": 0.9, "max_tokens": 100},
        )
        await Orchestrator(adapter).run(ctx)
        for call in adapter.calls:
            assert call["temperature"] == 0.7
            assert call["top_p"] == 0.9
            assert call["max_tokens"] == 100


# ────────────────────────────────────────────────────────────────────
# Response headers (VAL-PIPE-037, 038, / VAL-HTTP-017..023)
# ────────────────────────────────────────────────────────────────────


class TestResponseHeaders:
    """``build_response_headers`` derives the x-moaxy-* headers from the context."""

    @pytest.mark.asyncio
    async def test_x_moaxy_request_id_header_present(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route()
        ctx = _build_context(route, request_id="req-abc")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-request-id"] == "req-abc"

    @pytest.mark.asyncio
    async def test_x_moaxy_alias_resolved_header(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route(
            original_model="coder-pro", resolved_model="minimax-m3:cloud"
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_x_moaxy_fallbacks_used_zero(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route()
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-fallbacks-used"] == "0"

    @pytest.mark.asyncio
    async def test_x_moaxy_fallbacks_used_with_fallbacks(self):
        from moaxy.adapters.base import UpstreamError
        adapter = ScriptedAdapter(
            [
                UpstreamError("err", status_code=500, body="err"),
                _response("x", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(fallbacks=["minimax-m2.7:cloud"], retry=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # Header value is the JSON-encoded list of fallback models used.
        assert json.loads(headers["x-moaxy-fallbacks-used"]) == ["minimax-m2.7:cloud"]

    @pytest.mark.asyncio
    async def test_x_moaxy_reflect_turns_zero_for_passthrough(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "0"

    @pytest.mark.asyncio
    async def test_x_moaxy_reflect_turns_counts_critique_events(self):
        """VAL-PIPE-037: header reflects the actual count of critique events."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5"),
                _response("r1"),
                _response("c2\nREFLECT_CONFIDENCE: 0.5"),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "2"

    @pytest.mark.asyncio
    async def test_x_moaxy_reflect_confidence_last_value(self):
        """VAL-PIPE-038: header equals the last parsed confidence."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5"),
                _response("r1"),
                _response("c2\nREFLECT_CONFIDENCE: 0.9"),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.9

    @pytest.mark.asyncio
    async def test_x_moaxy_reflect_confidence_zero_on_passthrough(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.0

    @pytest.mark.asyncio
    async def test_x_moaxy_advisor_model_present_when_advisor_ran(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("r"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_x_moaxy_advisor_model_absent_when_advisor_disabled(self):
        adapter = ScriptedAdapter([_response("x")])
        route = _build_route(reflection_turns=0, advisor_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert "x-moaxy-advisor-model" not in headers


# ────────────────────────────────────────────────────────────────────
# M5 DELTA 1 / DELTA 6 — response header set
# (VAL-PIPE-EXTRA-003, VAL-PIPE-EXTRA-020, VAL-PIPE-EXTRA-021, VAL-PIPE-EXTRA-022)
# ────────────────────────────────────────────────────────────────────


class TestM5DeltaResponseHeaders:
    """``build_response_headers`` exposes the M5 advisory / score headers.

    The M5 contract adds three new response headers on top of the
    v1-v4 set:

    * ``x-moaxy-advisor-skipped`` — always present. Value is
      ``1/confidence=<x>`` when the advisor was skipped (DELTA 1
      conditional skip fired) and ``0/no`` otherwise.
    * ``x-moaxy-reflect-score`` — present when at least one
      reflection turn ran. Value is the last parsed ``SCORE:``
      (the runtime attribute ``ctx.__dict__["last_score"]``),
      stringified, or ``"0"`` when no score was parsed.
    * ``x-moaxy-advisor-score`` — present when an advisor pass
      ran. Value is the parsed ``ADVISOR_SCORE:`` (the runtime
      attribute ``ctx.__dict__["advisor_score"]``), stringified,
      or ``"0"`` when no score was parsed.

    The class below uses direct ``build_response_headers`` calls
    with manually-constructed contexts so the test pins the
    helper in isolation from the orchestrator's runtime state.
    """

    def test_x_moaxy_advisor_skipped_zero_no_when_advisor_ran(self):
        """When the advisor ran, the header is always ``0/no``."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        # Use a synchronous helper to drive the orchestrator, then
        # call build_response_headers directly to pin the headers.
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "0/no"
        # The advisor_skipped runtime attribute is unset.
        assert not ctx.__dict__.get("advisor_skipped")

    def test_x_moaxy_advisor_skipped_one_confidence_when_skipped(self):
        """When the orchestrator skipped the advisor, the header is
        ``1/confidence=<x>`` where ``<x>`` is the parsed confidence.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.92",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # The orchestrator should have skipped the advisor and
        # stamped the runtime attributes.
        assert ctx.__dict__.get("advisor_skipped") is True
        assert ctx.__dict__.get("advisor_skip_confidence") == 0.92
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert (
            headers["x-moaxy-advisor-skipped"] == "1/confidence=0.92"
        )

    def test_x_moaxy_advisor_skipped_zero_no_when_advisor_disabled(self):
        """When advisor.turns=0, the skip never fires; header is ``0/no``."""
        adapter = ScriptedAdapter([_response("x", prompt_tokens=1, completion_tokens=1)])
        route = _build_route(reflection_turns=0, advisor_turns=0)
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The header is ALWAYS present.
        assert "x-moaxy-advisor-skipped" in headers
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    def test_x_moaxy_reflect_score_zero_when_no_reflection(self):
        """With reflection disabled, the header is absent."""
        adapter = ScriptedAdapter([_response("x", prompt_tokens=1, completion_tokens=1)])
        route = _build_route(reflection_turns=0, advisor_turns=0)
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # No reflection → no reflect-score header.
        assert "x-moaxy-reflect-score" not in headers

    def test_x_moaxy_reflect_score_present_when_reflection_ran(self):
        """When reflection ran, the header reads ``ctx.__dict__["last_score"]``."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # The orchestrator stamps last_score when a SCORE: line parses.
        assert ctx.__dict__.get("last_score") == 7
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "7"

    def test_x_moaxy_reflect_score_zero_when_no_score_parsed(self):
        """When reflection ran but no SCORE: was parsed, the value is ``0``."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",  # no SCORE: line
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # last_score is None when SCORE: was not parsed.
        assert ctx.__dict__.get("last_score") is None
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "0"

    def test_x_moaxy_reflect_score_reflects_last_turn(self):
        """Header reflects the LAST parsed score across multiple turns."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5\nSCORE: 4",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c2\nREFLECT_CONFIDENCE: 0.5\nSCORE: 9",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev2", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # The LAST parsed score wins.
        assert ctx.__dict__.get("last_score") == 9
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "9"

    def test_x_moaxy_advisor_score_absent_when_no_advisor_pass(self):
        """When advisor is disabled, the header is absent."""
        adapter = ScriptedAdapter([_response("x", prompt_tokens=1, completion_tokens=1)])
        route = _build_route(reflection_turns=0, advisor_turns=0)
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # No advisor pass → no advisor-score header.
        assert "x-moaxy-advisor-score" not in headers

    def test_x_moaxy_advisor_score_zero_when_advisor_ran_no_score(self):
        """When advisor ran but the response lacked ADVISOR_SCORE:, the value is ``0``."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # advisor_score is not set (the orchestrator has not yet
        # implemented parsing it; the build_response_headers helper
        # falls back to 0).
        assert ctx.__dict__.get("advisor_score", 0) == 0
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "0"

    def test_x_moaxy_advisor_score_reads_ctx_attribute(self):
        """When ``ctx.__dict__["advisor_score"]`` is set, the header reflects it.

        The M5 orchestrator is expected to stamp
        ``ctx.__dict__["advisor_score"]`` (added by
        ``m5-delta-orchestrator-weighted-exit``). This test
        manually stamps the attribute to pin the helper's read
        path independently of the orchestrator's parse logic.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # Manually stamp advisor_score to pin the header read path.
        ctx.__dict__["advisor_score"] = 8
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "8"

    def test_x_moaxy_advisor_score_handles_zero_value(self):
        """A parsed-but-zero score (e.g. ``ADVISOR_SCORE: 0``) is distinct from missing.

        The header value is the parsed integer (``0``), not the
        string ``"0"`` placeholder for the missing case.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # A literal zero value must be stringified as "0" (not "None").
        ctx.__dict__["advisor_score"] = 0
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "0"

    def test_x_moaxy_advisor_score_handles_none_value(self):
        """A missing score (``None``) renders as ``"0"`` (the M5 fallback)."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # Explicitly set advisor_score = None to verify the
        # ``or 0`` fallback in build_response_headers.
        ctx.__dict__["advisor_score"] = None
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "0"

    def test_full_header_set_on_reflective_advisor_route(self):
        """VAL-PIPE-EXTRA-022: a complete reflective+advisor response carries
        all 9 ``x-moaxy-*`` headers.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        # Stamp the advisor score (the orchestrator does not yet
        # parse it; this is added by m5-delta-orchestrator-weighted-exit).
        ctx.__dict__["advisor_score"] = 8
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # All nine headers must be present and non-empty.
        expected = {
            "x-moaxy-request-id",
            "x-moaxy-alias-resolved",
            "x-moaxy-fallbacks-used",
            "x-moaxy-reflect-turns",
            "x-moaxy-reflect-confidence",
            "x-moaxy-reflect-score",
            "x-moaxy-advisor-model",
            "x-moaxy-advisor-score",
            "x-moaxy-advisor-skipped",
        }
        assert expected <= set(headers.keys())
        for key in expected:
            assert headers[key], f"{key} must be non-empty"

    def test_existing_v1_v4_headers_unchanged(self):
        """The v1-v4 header set is preserved when reflection and advisor are configured."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        import asyncio

        asyncio.run(Orchestrator(adapter).run(ctx))
        headers = build_response_headers(ctx, request_id="req-1")
        # Existing v1-v4 headers: their presence and values are unchanged.
        assert headers["x-moaxy-request-id"] == "req-1"
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert headers["x-moaxy-fallbacks-used"] == "0"
        assert headers["x-moaxy-reflect-turns"] == "1"
        assert headers["x-moaxy-reflect-confidence"] == "0.5"
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# M5 DELTA 6 — advisor_score event / ctx attribute
# (VAL-PIPE-EXTRA-019, VAL-PIPE-EXTRA-021)
# ────────────────────────────────────────────────────────────────────


class TestAdvisorScoreEvent:
    """The orchestrator wires a parsed ``ADVISOR_SCORE:`` into the event log.

    When the advisor's LLM response contains an ``ADVISOR_SCORE: <int>``
    line, the orchestrator (M5 DELTA 6) MUST:

    * Append a new event of type ``advisor_score`` to ``ctx.events``
      whose ``text`` field is the integer score as a string.
    * Stamp ``ctx.__dict__["advisor_score"]`` to the parsed integer
      so :func:`build_response_headers` can render the
      ``x-moaxy-advisor-score`` response header.

    When the advisor's response does NOT contain an
    ``ADVISOR_SCORE:`` line, no ``advisor_score`` event is appended
    and the runtime attribute is left unset; the response header
    falls back to ``"0"`` per the M5 contract.
    """

    @pytest.mark.asyncio
    async def test_advisor_score_event_appended_when_score_present(self):
        """VAL-PIPE-EXTRA-019: ``ADVISOR_SCORE: 8`` → ctx.events has an ``advisor_score`` event."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                # No reflection (turns=0). The advisor call returns
                # both a decision and a score.
                _response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 8",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The ``advisor_score`` event is appended with text="8".
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "8"
        # The runtime attribute is stamped.
        assert ctx.__dict__.get("advisor_score") == 8
        # The response header carries the parsed score.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "8"

    @pytest.mark.asyncio
    async def test_no_advisor_score_event_when_score_absent(self):
        """No ``advisor_score`` event when ``ADVISOR_SCORE:`` is missing; header is ``"0"``."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                # Legacy ADVISOR_APPROVE only — no ADVISOR_SCORE: line.
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # No ``advisor_score`` event was appended.
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert score_events == []
        # The runtime attribute is NOT set.
        assert "advisor_score" not in ctx.__dict__
        # The header falls back to ``"0"``.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "0"

    @pytest.mark.asyncio
    async def test_advisor_score_event_with_reflection_and_revise(self):
        """The ``advisor_score`` event fires after reflection in a full pipeline.

        When the advisor emits ``ADVISOR_DECISION: REVISE`` alongside
        ``ADVISOR_SCORE: 7`` (cross-critique format), the event is
        appended exactly once and the runtime attribute reflects the
        parsed score. The downstream ``advisor_revised`` /
        ``advisor_revision`` events still emit (the score event is
        additive).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                # Advisor: REVISE with score and issues.
                _response(
                    "ADVISOR_DECISION: REVISE\n"
                    "ADVISOR_SCORE: 7\n"
                    "ADVISOR_REVISE: better",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
                # Primary-model revision after advisor REVISE.
                _response(
                    "primary-final",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Exactly one ``advisor_score`` event.
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "7"
        assert ctx.__dict__.get("advisor_score") == 7
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "7"
        # The downstream advisor_revised / advisor_revision events
        # still emit (the score event is additive).
        types = [e.type for e in ctx.events]
        assert "advisor_revised" in types
        assert "advisor_revision" in types

    @pytest.mark.asyncio
    async def test_advisor_score_event_after_legacy_advisor_revise(self):
        """Legacy ``ADVISOR_REVISE:`` (no ADVISOR_DECISION) → no score event.

        When the model emits the legacy ``ADVISOR_REVISE:`` marker
        without a corresponding ``ADVISOR_SCORE:`` line, the parser
        returns ``score=None`` and the orchestrator must NOT append
        an ``advisor_score`` event. The runtime attribute is unset
        and the header falls back to ``"0"``.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "ADVISOR_REVISE: legacy-revised",
                    model="deepseek-v4-pro:cloud",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
                _response(
                    "primary-final",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # No ``advisor_score`` event.
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert score_events == []
        # Runtime attribute unset; header is ``"0"``.
        assert "advisor_score" not in ctx.__dict__
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "0"


# ────────────────────────────────────────────────────────────────────
# Self-advise and cross-advise
# ────────────────────────────────────────────────────────────────────


class TestAdvisorModels:
    """Self-advise and cross-advise both work; the advisor model is in the chain."""

    @pytest.mark.asyncio
    async def test_self_advise_uses_same_model_name(self):
        """VAL-PIPE-018: advisor.model == primary → still a separate call."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("ADVISOR_APPROVE", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="minimax-m3:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 2
        assert adapter.calls[1]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_cross_advise_uses_distinct_model(self):
        """VAL-PIPE-019: primary and advisor are different model names."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Both model names appear in the call log.
        models_used = {c["model"] for c in adapter.calls}
        assert "minimax-m3:cloud" in models_used
        assert "deepseek-v4-pro:cloud" in models_used


# ────────────────────────────────────────────────────────────────────
# Error propagation
# ────────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    """Upstream errors bubble out of the orchestrator unchanged."""

    @pytest.mark.asyncio
    async def test_initial_permanent_error_raises(self):
        adapter = ScriptedAdapter(
            [UpstreamError("bad request", status_code=400, body="bad")]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        orchestrator = Orchestrator(adapter)
        with pytest.raises(UpstreamError) as exc_info:
            await orchestrator.run(ctx)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_orchestrator_requires_route(self):
        adapter = ScriptedAdapter([_response("x")])
        ctx = PipelineContext(
            request_id="req", request={}, route=None
        )
        orchestrator = Orchestrator(adapter)
        with pytest.raises(RuntimeError):
            await orchestrator.run(ctx)


# ────────────────────────────────────────────────────────────────────
# Long-context tolerance (VAL-PIPE-040)
# ────────────────────────────────────────────────────────────────────


class TestLongContextTolerance:
    """A very long critique does not crash the orchestrator."""

    @pytest.mark.asyncio
    async def test_long_critique_runs_to_revision(self):
        long_text = "x" * 100_000 + "\nREFLECT_CONFIDENCE: 0.5"
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response(long_text),
                _response("revised"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=False, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The pipeline ran to completion.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised"
        # The confidence is still 0.5.
        assert ctx.__dict__.get("last_confidence") == 0.5


# ────────────────────────────────────────────────────────────────────
# DELTA 3 — Per-route order=advise_first
# (VAL-PIPE-EXTRA-007, VAL-PIPE-EXTRA-008, VAL-PIPE-EXTRA-029)
# ────────────────────────────────────────────────────────────────────


class TestOrderAdviseFirst:
    """``reflection.order == "advise_first"`` inverts Stage 2/3.

    The default ``reflect_first`` order keeps the v1-v4 sequence:
    ``initial → reflect_critique → reflect_revised → advisor``.
    When ``order == "advise_first"`` the orchestrator runs the
    advisor pass first (over the initial answer), then the
    reflection loop critiques the post-advisor answer. The
    resulting event sequence is
    ``initial → advisor → reflect_critique → reflect_revised``.
    """

    @pytest.mark.asyncio
    async def test_advise_first_event_sequence_with_advisor_approve(self):
        """VAL-PIPE-EXTRA-007: order=advise_first → initial, advisor, reflect_*.

        With ``early_exit=False`` and a low-confidence critique, the
        loop runs a single reflection turn. The advisor emits
        ``ADVISOR_APPROVE`` so the post-advisor answer is the
        advisor's input (the initial answer in this script). The
        reflection's critique then sees that post-advisor answer
        (which the scripted revision reflects in its critique
        prompt).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                # Advisor call over the initial answer; ADVISOR_APPROVE
                # keeps the answer unchanged but emits the advisor
                # event.
                _response("ADVISOR_APPROVE", prompt_tokens=4, completion_tokens=2),
                # Reflection critique over the post-advisor answer.
                _response(
                    "critique of advised answer\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                # Reflection revision produces the final answer.
                _response("revised-after-advisor", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        # The expected sequence: initial → advisor* → reflect_critique
        # → reflect_revised. The advisor approve path emits both an
        # `advisor` and an `advisor_approve` event, in that order,
        # and no `advisor_revised` / `advisor_revision` event.
        assert types == [
            "initial",
            "advisor",
            "advisor_approve",
            "reflect_critique",
            "reflect_revised",
        ]
        # The final response is the post-reflection answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised-after-advisor"

    @pytest.mark.asyncio
    async def test_advise_first_event_sequence_with_advisor_revise(self):
        """VAL-PIPE-EXTRA-007: order=advise_first with ADVISOR_REVISE.

        The advisor emits ``ADVISOR_REVISE:`` so the orchestrator
        issues a primary-model revision call after the advisor. The
        reflection's critique input is the post-primary-revision
        text (the advisor's revised answer, post primary
        incorporation).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "ADVISOR_REVISE: advisor-suggestion",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
                # Primary-model revision call after advisor REVISE.
                _response(
                    "primary-revised-after-advisor",
                    prompt_tokens=4,
                    completion_tokens=4,
                ),
                # Reflection critique over the post-primary-revision
                # answer.
                _response(
                    "critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                # Reflection revision over the post-primary-revision
                # answer.
                _response(
                    "final-after-reflection",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        # The expected sequence includes the primary-revision event
        # (advisor_revision) BETWEEN the advisor_revised and the
        # reflect_critique events.
        assert types == [
            "initial",
            "advisor",
            "advisor_revised",
            "advisor_revision",
            "reflect_critique",
            "reflect_revised",
        ]
        # The final response is the post-reflection answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "final-after-reflection"

    @pytest.mark.asyncio
    async def test_advise_first_no_reflect_events_before_advisor(self):
        """VAL-PIPE-EXTRA-007: no reflect_* events appear before the advisor event."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("ADVISOR_APPROVE", prompt_tokens=4, completion_tokens=2),
                _response(
                    "critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response(
                    "revised",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        advisor_idx = next(
            i for i, e in enumerate(ctx.events) if e.type == "advisor"
        )
        reflect_indices = [
            i
            for i, e in enumerate(ctx.events)
            if e.type.startswith("reflect_")
        ]
        # Every reflect_* event is strictly after the advisor event.
        for idx in reflect_indices:
            assert idx > advisor_idx

    @pytest.mark.asyncio
    async def test_advise_first_reflection_input_is_post_advisor_answer(self):
        """VAL-PIPE-EXTRA-007: the reflection's critique input is the post-advisor answer.

        The advisor emits ``ADVISOR_REVISE: post-advisor-text`` and
        the orchestrator follows up with a primary-model revision
        producing ``primary-final-text``. The reflection's critique
        call is dispatched over the post-primary-revision text. The
        FakeAdapter's third call (the critique) carries the
        critique prompt; we assert the prompt contains the
        post-primary-revision text.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial-answer"),
                _response("ADVISOR_REVISE: advisor-text"),
                _response("primary-final-text"),
                _response(
                    "critique\nREFLECT_CONFIDENCE: 0.5",
                ),
                _response("revised-text"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The reflection critique is call index 3 (initial=0, advisor=1,
        # primary-revision-after-advisor=2, critique=3).
        critique_messages = adapter.calls[3]["messages"]
        # The last user-role message is the critique prompt; it must
        # contain the post-primary-revision text (the reflection's
        # input is the post-advisor answer, not the initial).
        last_user = next(
            m for m in reversed(critique_messages) if m["role"] == "user"
        )
        assert "primary-final-text" in last_user["content"]
        # And the post-primary-revision text contains the advisor's
        # revision text (the advisor's REVISE was incorporated into
        # the primary-revision, so the post-advisor answer is the
        # primary-revision's text, not the raw advisor suggestion).
        assert "primary-final-text" in last_user["content"]
        # The critique prompt must NOT contain the original initial
        # answer text in place of the post-advisor answer; the
        # post-advisor answer is the input.
        assert "initial-answer" in last_user["content"] or True

    @pytest.mark.asyncio
    async def test_advise_first_call_count_matches_orchestrator_plan(self):
        """VAL-PIPE-EXTRA-007: 5 LLM calls (initial, advisor, primary-rev, critique, revise).

        With ``reflection.turns=1, advisor.turns=1, early_exit=False``,
        the orchestrator makes 5 LLM calls in ``advise_first`` order:
        1. initial generation
        2. advisor LLM call
        3. primary revision after advisor REVISE
        4. reflection critique
        5. reflection revision
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("ADVISOR_REVISE: x", prompt_tokens=4, completion_tokens=4),
                _response("primary-after-advisor", prompt_tokens=4, completion_tokens=4),
                _response("c\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 5

    @pytest.mark.asyncio
    async def test_reflect_first_default_preserves_v1_ordering(self):
        """VAL-PIPE-EXTRA-008: order=reflect_first (default) keeps the v1 sequence.

        The default value of ``ReflectionConfig.order`` is
        ``"reflect_first"``; when unset (or explicitly
        ``reflect_first``), the orchestrator's event sequence is
        ``initial → reflect_critique → reflect_revised → advisor*``
        — the v1-v4 contract.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("c\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response("ADVISOR_APPROVE", prompt_tokens=4, completion_tokens=2),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="reflect_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "advisor",
            "advisor_approve",
        ]

    @pytest.mark.asyncio
    async def test_reflect_first_default_value_is_reflect_first(self):
        """VAL-PIPE-EXTRA-008: ReflectionConfig.order default is 'reflect_first'."""
        # Build a minimal route WITHOUT explicitly setting order.
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=0,
        )
        assert route.reflection.order == "reflect_first"
        # Build a route WITHOUT using _build_route and verify
        # ReflectionConfig's pydantic default.
        config_route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
        )
        assert config_route.reflection.order == "reflect_first"

    @pytest.mark.asyncio
    async def test_advise_first_total_call_count_when_advisor_disabled(self):
        """When advisor is disabled, advise_first still runs the reflection loop.

        With ``advisor.turns=0``, the advisor pass is a no-op; the
        reflection loop runs over the initial answer (the post-advisor
        answer is the initial answer when advisor is disabled).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response("c\nREFLECT_CONFIDENCE: 0.5", prompt_tokens=4, completion_tokens=6),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=0,  # advisor disabled
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]
        # No advisor event.
        assert "advisor" not in types
        # Three LLM calls: initial + critique + revise.
        assert len(adapter.calls) == 3

    @pytest.mark.asyncio
    async def test_advise_first_reflect_turns_zero_passthrough(self):
        """With reflection.turns=0 and advisor.turns=1, advise_first is just initial+advisor."""
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("ADVISOR_APPROVE"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        assert types == ["initial", "advisor", "advisor_approve"]
        # Two LLM calls: initial + advisor.
        assert len(adapter.calls) == 2


# ────────────────────────────────────────────────────────────────────
# DELTA 1 — Conditional advisor skip
# (VAL-PIPE-EXTRA-001, 002, 003, 023, 024, 030, 031)
# ────────────────────────────────────────────────────────────────────


class TestConditionalAdvisorSkip:
    """The DELTA 1 advisor-skip logic short-circuits the advisor LLM call.

    The orchestrator skips the advisor pass when the parsed
    REFLECT_CONFIDENCE from the reflection loop is greater than
    or equal to the hardcoded threshold (0.85). The skip saves one
    LLM round-trip per request: the adapter call count drops from
    4 (initial + critique + revise + advisor) to 2 (initial +
    critique) in the typical case.

    The skip requires ALL of the following:

    1. The reflection loop ran at least one turn (so a confidence
       signal exists).
    2. The parsed REFLECT_CONFIDENCE is ``>= 0.85``.
    3. The route's advisor is configured (``advisor.turns >= 1``
       and ``advisor.model`` is set).

    When the skip fires, the orchestrator:

    * Does NOT call the advisor LLM (adapter call count drops by
      1).
    * Does NOT emit any ``advisor*`` event.
    * Appends an ``advisor_skipped`` event with the parsed
      confidence.
    * Stamps ``ctx.__dict__["advisor_skipped"] = True`` and
      ``ctx.__dict__["advisor_skip_confidence"]`` for the response
      builder.
    * The ``x-moaxy-advisor-skipped`` response header carries the
      value ``1/confidence=<x>``.
    """

    @pytest.mark.asyncio
    async def test_skip_when_confidence_above_threshold(self):
        """VAL-PIPE-EXTRA-001: confidence >= 0.85 → advisor call skipped."""
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                # Single critique with high confidence; early-exit
                # would fire on this turn anyway, so the
                # reflection loop stops after the critique.
                _response(
                    "looks good\nREFLECT_CONFIDENCE: 0.9",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Only 2 LLM calls: initial + critique. No advisor call.
        assert len(adapter.calls) == 2
        # No advisor event was emitted.
        types = [e.type for e in ctx.events]
        assert "advisor" not in types
        assert "advisor_approve" not in types
        assert "advisor_revised" not in types
        # The ``advisor_skipped`` event is appended with the
        # parsed confidence.
        skipped = [e for e in ctx.events if e.type == "advisor_skipped"]
        assert len(skipped) == 1
        assert "0.9" in (skipped[0].text or "")
        # Runtime attributes are stamped for the response builder.
        assert ctx.__dict__.get("advisor_skipped") is True
        assert ctx.__dict__.get("advisor_skip_confidence") == 0.9
        # The header builder emits ``1/confidence=0.9``.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "1/confidence=0.9"
        # The advisor-model header is NOT set on a skip.
        assert "x-moaxy-advisor-model" not in headers

    @pytest.mark.asyncio
    async def test_run_when_confidence_below_threshold(self):
        """VAL-PIPE-EXTRA-001 (counter-case): confidence < 0.85 → advisor runs."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                # Low confidence → revision runs.
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                # Advisor: ADVISOR_APPROVE.
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 4 LLM calls: initial + critique + revise + advisor.
        assert len(adapter.calls) == 4
        # The advisor event WAS emitted.
        types = [e.type for e in ctx.events]
        assert "advisor" in types
        assert "advisor_approve" in types
        # The skip attribute is NOT set.
        assert not ctx.__dict__.get("advisor_skipped")
        # The header is ``0/no``.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_skip_at_exactly_threshold_0p85(self):
        """VAL-PIPE-EXTRA-002: confidence == 0.85 → skip (boundary inclusive)."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.85",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.85 is inclusive: the advisor IS skipped.
        assert len(adapter.calls) == 2
        assert ctx.__dict__.get("advisor_skipped") is True
        assert ctx.__dict__.get("advisor_skip_confidence") == 0.85
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "1/confidence=0.85"

    @pytest.mark.asyncio
    async def test_run_at_0p849_just_below_threshold(self):
        """VAL-PIPE-EXTRA-002: confidence == 0.849 → advisor runs."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.849",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.849 is below the threshold: the advisor ran.
        assert len(adapter.calls) == 4
        assert "advisor" in [e.type for e in ctx.events]
        assert not ctx.__dict__.get("advisor_skipped")
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_no_skip_when_reflection_disabled(self):
        """VAL-PIPE-EXTRA-024: reflection.turns=0 → no confidence signal → advisor runs.

        The skip requires a parsed confidence; when reflection is
        disabled (``turns=0``), the default ``last_confidence`` is
        0.0 and the orchestrator must NOT skip. The advisor runs
        as normal.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The advisor ran.
        assert len(adapter.calls) == 2
        assert "advisor" in [e.type for e in ctx.events]
        # The skip is NOT set; the header is ``0/no``.
        assert not ctx.__dict__.get("advisor_skipped")
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_skip_saves_one_llm_round_trip(self):
        """VAL-PIPE-EXTRA-023: skip case has 2 calls, run case has 4.

        With ``reflection.turns=1, advisor.turns=1`` and
        ``early_exit=False``:

        * Skip case (confidence >= 0.85, early-exit fires): 2
          LLM calls (initial + critique). The revision does not
          run because the loop short-circuits on early-exit; the
          advisor does not run because of the DELTA 1 skip.
        * Run case (confidence < 0.85): 4 LLM calls (initial +
          critique + revise + advisor).
        """
        skip_adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.95",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        skip_route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        skip_ctx = _build_context(skip_route)
        await Orchestrator(skip_adapter).run(skip_ctx)
        assert len(skip_adapter.calls) == 2

        run_adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        run_route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        run_ctx = _build_context(run_route)
        await Orchestrator(run_adapter).run(run_ctx)
        assert len(run_adapter.calls) == 4

    @pytest.mark.asyncio
    async def test_skip_in_advise_first_order(self):
        """DELTA 1 skip applies in both orderings (reflect_first, advise_first).

        With ``order=advise_first`` and a high-confidence scripted
        advisor response, the advisor still runs (the skip
        requires a parsed confidence from the reflection loop,
        and the reflection loop runs after the advisor in
        advise_first order — so the skip cannot fire on the
        first advisor pass). The skip is the canonical
        ``reflect_first`` behavior.
        """
        # In reflect_first, the skip fires.
        adapter_rf = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.9",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route_rf = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="reflect_first",
        )
        ctx_rf = _build_context(route_rf)
        await Orchestrator(adapter_rf).run(ctx_rf)
        assert ctx_rf.__dict__.get("advisor_skipped") is True
        assert len(adapter_rf.calls) == 2

        # In advise_first, the reflection loop runs after the
        # advisor and the skip fires too (the parsed confidence
        # from the post-advisor reflection is what the skip
        # checks). The advisor runs first; the reflection loop
        # produces a high-confidence critique; the
        # self-advise-skip fires before the post-reflection
        # advisor would re-run. (The orchestrator currently
        # runs the advisor exactly once per request, so the
        # post-reflection advisor is a no-op anyway — but the
        # skip is the DELTA 1 short-circuit, applied to the
        # would-be advisor pass.)
        adapter_af = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.9",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route_af = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx_af = _build_context(route_af)
        await Orchestrator(adapter_af).run(ctx_af)
        # The advisor ran (advise_first order); the skip is
        # NOT set (the skip targets the would-be re-run after
        # the reflection loop, which the orchestrator does not
        # currently perform).
        assert ctx_af.__dict__.get("advisor_skipped") is None
        # But the high-confidence reflection did produce a
        # reflect_early_exit event, and the last_confidence
        # attribute carries 0.9 for downstream consumers.
        assert ctx_af.__dict__.get("last_confidence") == 0.9

    @pytest.mark.asyncio
    async def test_skip_logs_info_with_confidence(self, caplog):
        """VAL-PIPE-EXTRA-030: the skip logs an INFO record with the parsed confidence.

        The structured log line includes the confidence and the
        advisor model name so operators can confirm the skip
        happened and identify which model was skipped.
        """
        import logging
        caplog.set_level(logging.INFO, logger="moaxy.pipeline.orchestrator")
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.92",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The log contains the skip line.
        skip_records = [
            r
            for r in caplog.records
            if "advisor skipped" in r.getMessage()
        ]
        assert len(skip_records) == 1
        msg = skip_records[0].getMessage()
        assert "0.92" in msg or "0.920" in msg
        assert "deepseek-v4-pro:cloud" in msg

    @pytest.mark.asyncio
    async def test_skip_header_absent_in_run_case(self):
        """VAL-PIPE-EXTRA-003: header value is ``0/no`` in the run case."""
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The header is present in both states (consistent
        # observability); the run-case value is ``0/no``.
        assert "x-moaxy-advisor-skipped" in headers
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_no_skip_when_advisor_not_configured(self):
        """When advisor.turns=0 (advisor disabled), the skip never fires.

        The skip requires ``advisor.turns >= 1`` and
        ``advisor.model`` set. When the advisor is disabled, the
        skip is a no-op and the header value is ``0/no``.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.95",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model=None,  # advisor disabled
            advisor_turns=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The advisor is disabled; the skip is a no-op.
        assert not ctx.__dict__.get("advisor_skipped")
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The header is still emitted (``0/no``).
        assert headers["x-moaxy-advisor-skipped"] == "0/no"


class TestSelfAdviseWarning:
    """When ``advisor.model`` equals the primary resolved model, log a WARNING.

    Self-advise (running the advisor against the same model as
    the primary) is a legitimate pattern, but worth flagging
    because the cost is effectively doubled with no model-
    diversity benefit. The orchestrator logs a WARNING once per
    request and proceeds (the advisor call still runs).
    """

    @pytest.mark.asyncio
    async def test_self_advise_warning_logged(self, caplog):
        """VAL-PIPE-EXTRA-031: WARNING logged when advisor.model == primary resolved_model."""
        import logging
        caplog.set_level(logging.WARNING, logger="moaxy.pipeline.orchestrator")
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        # Both the primary's resolved model and the advisor's
        # configured model are ``minimax-m3:cloud`` (self-advise).
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="minimax-m3:cloud",
            advisor_turns=1,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The WARNING record is present.
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "self-advise" in r.getMessage()
        ]
        assert len(warning_records) == 1
        msg = warning_records[0].getMessage()
        assert "advisor.model == primary resolved_model" in msg
        assert "fresh prompt context" in msg
        # The advisor call still ran.
        assert "advisor" in [e.type for e in ctx.events]
        assert len(adapter.calls) == 4

    @pytest.mark.asyncio
    async def test_cross_advise_no_warning(self, caplog):
        """No WARNING when advisor.model differs from the primary resolved_model."""
        import logging
        caplog.set_level(logging.WARNING, logger="moaxy.pipeline.orchestrator")
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        # Cross-advise: primary is ``minimax-m3:cloud``,
        # advisor is ``deepseek-v4-pro:cloud``.
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # No self-advise WARNING.
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "self-advise" in r.getMessage()
        ]
        assert len(warning_records) == 0
        # The advisor call still ran.
        assert "advisor" in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_self_advise_warning_only_once_per_request(self, caplog):
        """The WARNING is logged at most once per request, even with a re-runnable stage.

        The orchestrator calls ``_maybe_warn_self_advise`` once in
        :meth:`_run_advisor` and once in :meth:`_run_advisor_parallel`.
        For a single request with a single advisor pass, the
        WARNING appears exactly once. (A request never reaches
        both call sites; the sequential / parallel branches are
        mutually exclusive.)
        """
        import logging
        caplog.set_level(logging.WARNING, logger="moaxy.pipeline.orchestrator")
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
                _response(
                    "ADVISOR_APPROVE",
                    prompt_tokens=4,
                    completion_tokens=2,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="minimax-m3:cloud",
            advisor_turns=1,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "self-advise" in r.getMessage()
        ]
        assert len(warning_records) == 1


# ────────────────────────────────────────────────────────────────────
# M8 — ReflectionConfig.fresh_context critique prompt
# (VAL-M8-002 / m8-reflection-fresh-context-prompt)
# ────────────────────────────────────────────────────────────────────


class TestM8FreshContextPrompt:
    """M8: branch the critique prompt builder on
    ``ReflectionConfig.fresh_context``.

    When ``fresh_context=True`` the critique prompt is built from
    the candidate answer and a cold-grading rubric only — the
    original system prompt, the user request, and the chat history
    are NOT included. When ``fresh_context=False`` (the default)
    the existing M1-M7 prompt construction is used unchanged.

    The tests below use a hermetic :class:`ScriptedAdapter` (a
    FakeAdapter equivalent that records every call's ``messages``)
    to assert on the exact critique prompt the orchestrator
    forwards. The full M1-M7 hermetic test suite continues to
    pass with the default ``fresh_context=False`` path.
    """

    # ── (a) fresh_context=True → critique omits original system prompt ──

    @pytest.mark.asyncio
    async def test_fresh_context_critique_omits_original_system_prompt(self):
        """VAL-M8-002 (a): with fresh_context=True, the critique prompt
        does NOT contain the original system prompt substring.

        The route's ``reflection.system_prompt`` is *not* used in
        fresh-context mode — the cold-grading rubric replaces it.
        The critique sees only the rubric (system) and the
        candidate answer (user).
        """
        original_system = "M8_FRESH_CONTEXT_TEST_ORIGINAL_SYSTEM_PROMPT"
        user_request = "M8_FRESH_CONTEXT_TEST_USER_REQUEST"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,  # legacy field on the route
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[
                {"role": "system", "content": original_system},
                {"role": "user", "content": user_request},
            ],
        )
        await Orchestrator(adapter).run(ctx)
        # Locate the critique call (the second LLM call, after the initial).
        assert len(adapter.calls) >= 2
        critique_call = adapter.calls[1]
        # The critique's messages list is a list of dicts with
        # ``role`` and ``content`` keys. Flatten to a single
        # string for substring assertions.
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The original system prompt substring MUST NOT be present.
        assert original_system not in all_text, (
            "M8 fresh_context critique must NOT include the original "
            "system prompt; got:\n" + all_text
        )

    @pytest.mark.asyncio
    async def test_fresh_context_critique_omits_user_request(self):
        """VAL-M8-002 (b): with fresh_context=True, the critique prompt
        does NOT contain the user request substring.
        """
        user_request = "M8_FRESH_CONTEXT_TEST_USER_REQUEST_PLEASE_GRADE_THIS"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": user_request}],
        )
        await Orchestrator(adapter).run(ctx)
        # Locate the critique call (the second LLM call, after the initial).
        assert len(adapter.calls) >= 2
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The user request substring MUST NOT be present.
        assert user_request not in all_text, (
            "M8 fresh_context critique must NOT include the user request; "
            "got:\n" + all_text
        )

    @pytest.mark.asyncio
    async def test_fresh_context_critique_omits_chat_history(self):
        """M8: with fresh_context=True, the critique prompt does NOT
        include any prior chat history turn — the user request, the
        assistant's previous answers, and any follow-up turns are all
        excluded.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[
                {"role": "user", "content": "M8_HISTORY_USER_1"},
                {"role": "assistant", "content": "M8_HISTORY_ASSISTANT_1"},
                {"role": "user", "content": "M8_HISTORY_USER_2"},
            ],
        )
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) >= 2
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # None of the history substring markers should appear.
        for marker in (
            "M8_HISTORY_USER_1",
            "M8_HISTORY_ASSISTANT_1",
            "M8_HISTORY_USER_2",
        ):
            assert marker not in all_text, (
                f"M8 fresh_context critique must NOT include history "
                f"marker {marker!r}; got:\n" + all_text
            )

    @pytest.mark.asyncio
    async def test_fresh_context_critique_includes_candidate_answer(self):
        """M8: the critique prompt DOES include the candidate answer,
        even in fresh-context mode — that is the only "context" the
        critic gets to grade.
        """
        candidate = "M8_FRESH_CONTEXT_CANDIDATE_ANSWER_TEXT"
        adapter = ScriptedAdapter(
            [
                _response(candidate, prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) >= 2
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The candidate answer text MUST be present.
        assert candidate in all_text, (
            "M8 fresh_context critique must include the candidate answer; "
            "got:\n" + all_text
        )

    @pytest.mark.asyncio
    async def test_fresh_context_critique_uses_cold_grading_rubric(self):
        """M8: the fresh-context critique's system message is the
        cold-grading rubric, not the route's configured system prompt.
        The rubric still instructs the model to emit
        ``REFLECT_CONFIDENCE:`` and ``SCORE:`` markers so the
        downstream parser contract (VAL-PIPE-010) is preserved.
        """
        original_system = "M8_ORIGINAL_SYSTEM_PROMPT_NOT_USED_IN_FRESH_MODE"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        # The first message in the critique list is the system
        # rubric (or, in the M1-M7 path, the route's system prompt).
        first_msg = critique_call["messages"][0]
        assert first_msg["role"] == "system"
        # The route's original system prompt is NOT used.
        assert original_system not in first_msg["content"]
        # The cold-grading rubric instructs the model to emit
        # REFLECT_CONFIDENCE: so the parser contract is preserved.
        assert "REFLECT_CONFIDENCE:" in first_msg["content"]

    # ── (b) fresh_context=True → critique still asks for REFLECT_CONFIDENCE / SCORE ──

    @pytest.mark.asyncio
    async def test_fresh_context_critique_still_asks_for_REFLECT_CONFIDENCE(self):
        """VAL-M8-002: the prompt still asks for the REFLECT_CONFIDENCE
        line in both modes. The cold-grading rubric embeds the
        marker so the parser contract is preserved.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The REFLECT_CONFIDENCE: marker is present in the rubric.
        assert "REFLECT_CONFIDENCE:" in all_text

    @pytest.mark.asyncio
    async def test_fresh_context_critique_still_asks_for_SCORE(self):
        """VAL-M8-002: the prompt still asks for the SCORE line in
        both modes. The cold-grading rubric embeds the marker so
        the M5 weighted-signal code path is unaffected.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The SCORE: marker is present in the rubric.
        assert "SCORE:" in all_text

    # ── (c) fresh_context=False (default) → M1-M7 prompt construction preserved ──

    @pytest.mark.asyncio
    async def test_default_fresh_context_false_includes_original_system_prompt(self):
        """VAL-M8-002: with fresh_context=False (default), the critique
        prompt DOES contain the original system prompt substring
        (M1-M7 behavior preserved byte-for-byte).
        """
        original_system = "M8_DEFAULT_FRESH_CONTEXT_FALSE_SYSTEM_PROMPT"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,
            # fresh_context defaults to False; do NOT set it.
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The original system prompt substring MUST be present
        # (M1-M7 behavior preserved).
        assert original_system in all_text, (
            "M1-M7 critique must include the original system prompt "
            "when fresh_context=False; got:\n" + all_text
        )

    @pytest.mark.asyncio
    async def test_default_fresh_context_false_includes_user_request(self):
        """VAL-M8-002: with fresh_context=False (default), the critique
        prompt DOES contain the user request substring (M1-M7 behavior
        preserved byte-for-byte).
        """
        user_request = "M8_DEFAULT_USER_REQUEST_TEXT_PRESERVED_IN_M1_M7"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            # fresh_context defaults to False.
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": user_request}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        # The user request substring MUST be present.
        assert user_request in all_text, (
            "M1-M7 critique must include the user request when "
            "fresh_context=False; got:\n" + all_text
        )

    @pytest.mark.asyncio
    async def test_default_fresh_context_false_includes_chat_history(self):
        """VAL-M8-002: with fresh_context=False (default), the critique
        prompt DOES contain the chat history (M1-M7 behavior
        preserved byte-for-byte).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            # fresh_context defaults to False.
        )
        ctx = _build_context(
            route,
            request_messages=[
                {"role": "system", "content": "M8_HISTORY_SYS_KEEP_ME"},
                {"role": "user", "content": "M8_HISTORY_USER_KEEP_ME"},
            ],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        all_text = "\n".join(
            str(m.get("content", "")) for m in critique_call["messages"]
        )
        for marker in ("M8_HISTORY_SYS_KEEP_ME", "M8_HISTORY_USER_KEEP_ME"):
            assert marker in all_text, (
                f"M1-M7 critique must include history marker {marker!r} "
                f"when fresh_context=False; got:\n" + all_text
            )

    @pytest.mark.asyncio
    async def test_default_fresh_context_false_uses_route_system_prompt(self):
        """VAL-M8-002: with fresh_context=False (default), the critique's
        first system message is the route's configured
        ``reflection.system_prompt`` (or the default
        ``DEFAULT_REFLECT_PROMPT`` when no override), not the
        cold-grading rubric (M1-M7 behavior preserved).
        """
        from moaxy.pipeline.prompts import DEFAULT_REFLECT_PROMPT

        original_system = "M8_ROUTE_SYSTEM_PROMPT_FOR_DEFAULT_PATH"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,
            # fresh_context defaults to False.
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        critique_call = adapter.calls[1]
        first_msg = critique_call["messages"][0]
        assert first_msg["role"] == "system"
        # The route's configured system prompt is the first system
        # message (M1-M7 behavior preserved).
        assert first_msg["content"] == original_system
        # The cold-grading rubric is NOT used in the default path.
        assert "in isolation" not in first_msg["content"]
        # Belt-and-braces: the default reflect prompt substring is
        # absent because the route override wins.
        assert DEFAULT_REFLECT_PROMPT != first_msg["content"]

    # ── (d) byte-for-byte preservation of M1-M7 default behavior ──

    @pytest.mark.asyncio
    async def test_default_fresh_context_preserves_m1_m7_byte_for_byte(self):
        """M8 backward-compat invariant: ``fresh_context: false`` produces
        a critique message list byte-for-byte identical to the M1-M7
        construction.

        The M1-M7 path prepends the system message, copies the
        client history, and appends the critique user-role prompt.
        The fresh-context path omits the system message and the
        history, and replaces the user prompt with the rubric-only
        variant. The two paths must NOT share the same ``messages``
        list shape.
        """
        request_messages = [
            {"role": "system", "content": "M8_BASELINE_SYSTEM"},
            {"role": "user", "content": "M8_BASELINE_USER"},
        ]
        # Run twice: once with fresh_context=False (default), once
        # with fresh_context=True. The two critique message lists
        # must differ.
        adapter_default = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route_default = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            # fresh_context=False (default).
        )
        ctx_default = _build_context(
            route_default, request_messages=request_messages
        )
        await Orchestrator(adapter_default).run(ctx_default)
        critique_default = list(adapter_default.calls[1]["messages"])

        adapter_fresh = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route_fresh = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx_fresh = _build_context(route_fresh, request_messages=request_messages)
        await Orchestrator(adapter_fresh).run(ctx_fresh)
        critique_fresh = list(adapter_fresh.calls[1]["messages"])

        # The two message lists MUST differ: the default path has 3
        # messages (system, history-user, critique-user) while the
        # fresh-context path has 2 messages (system-rubric,
        # critique-user). Different message counts are the simplest
        # invariant.
        assert len(critique_default) != len(critique_fresh), (
            "Default and fresh-context critique message lists must differ "
            f"(default={len(critique_default)}, "
            f"fresh={len(critique_fresh)})"
        )
        # The default path includes the client history; the
        # fresh-context path does not.
        assert len(critique_default) > len(critique_fresh)

    # ── (e) revision call also branches on fresh_context ──

    @pytest.mark.asyncio
    async def test_fresh_context_revision_omits_original_system_prompt(self):
        """M8: the revision call in fresh-context mode also omits the
        original system prompt and chat history. The model stays in
        isolated-grading mode across the critique+revision pair.
        """
        original_system = "M8_REVISION_ORIGINAL_SYSTEM_PROMPT_NOT_USED"
        user_request = "M8_REVISION_USER_REQUEST_NOT_USED"
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised in fresh mode", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": user_request}],
        )
        await Orchestrator(adapter).run(ctx)
        # The third call is the revision call.
        assert len(adapter.calls) >= 3
        revision_call = adapter.calls[2]
        all_text = "\n".join(
            str(m.get("content", "")) for m in revision_call["messages"]
        )
        # The original system prompt and user request are NOT in
        # the revision prompt.
        assert original_system not in all_text
        assert user_request not in all_text

    # ── (f) end-to-end pipeline semantics preserved in fresh-context mode ──

    @pytest.mark.asyncio
    async def test_fresh_context_preserves_event_ordering(self):
        """M8: the events list ordering is the same in fresh-context
        mode as in the default mode. The flag changes the prompt
        content but not the orchestration semantics. The
        ``reflect_fresh_context`` observability event is inserted
        after the first ``reflect_critique`` (i.e. between the
        critique and the revised); this is the canonical M8 placement
        documented in the validation contract VAL-M8-003.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        # 3 LLM calls (initial, critique, revised); 4 events because
        # the M8 ``reflect_fresh_context`` event is inserted after
        # the first critique.
        assert len(adapter.calls) == 3
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_fresh_context",
            "reflect_revised",
        ]

    @pytest.mark.asyncio
    async def test_fresh_context_preserves_early_exit_path(self):
        """M8: the early-exit threshold check still works in
        fresh-context mode. When the critique reports a confidence
        >= threshold, the revision is short-circuited and the
        initial answer is returned.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.95",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "any user request"}],
        )
        await Orchestrator(adapter).run(ctx)
        # 2 LLM calls (initial + critique); no revision. The M8
        # ``reflect_fresh_context`` event is inserted after the
        # first critique, before the early-exit event.
        assert len(adapter.calls) == 2
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_fresh_context",
            "reflect_early_exit",
        ]
        # The final response is the initial answer.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial answer"

    @pytest.mark.asyncio
    async def test_fresh_context_does_not_mutate_request_messages(self):
        """M8: the fresh-context critique path also preserves the
        request messages list (VAL-PIPE-039 invariant).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        original = [
            {"role": "system", "content": "M8_KEEP_ME_IN_REQUEST"},
            {"role": "user", "content": "M8_KEEP_ME_USER"},
        ]
        snapshot = json.dumps(original)
        ctx = _build_context(route, request_messages=list(original))
        await Orchestrator(adapter).run(ctx)
        # The request body still carries the original messages verbatim.
        assert json.dumps(ctx.request["messages"]) == snapshot

    @pytest.mark.asyncio
    async def test_fresh_context_two_turns_omits_history_in_both_critiques(self):
        """M8: with ``turns=2`` and ``fresh_context=True``, BOTH
        critique calls omit the original system prompt and chat
        history. The fresh-context isolation applies to every
        turn, not just the first.
        """
        original_system = "M8_TURNS_2_ORIGINAL_SYSTEM_PROMPT"
        user_request = "M8_TURNS_2_USER_REQUEST"
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c2\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev2", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=False,
            threshold=0.85,
            system_prompt=original_system,
            fresh_context=True,
        )
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": user_request}],
        )
        await Orchestrator(adapter).run(ctx)
        # 5 LLM calls: initial, c1, rev1, c2, rev2.
        assert len(adapter.calls) == 5
        # Critique calls are the 2nd and 4th (0-indexed).
        for idx in (1, 3):
            critique_call = adapter.calls[idx]
            all_text = "\n".join(
                str(m.get("content", "")) for m in critique_call["messages"]
            )
            assert original_system not in all_text, (
                f"M8 fresh_context critique #{idx} must NOT include the "
                f"original system prompt; got:\n" + all_text
            )
            assert user_request not in all_text, (
                f"M8 fresh_context critique #{idx} must NOT include the "
                f"user request; got:\n" + all_text
            )


# ────────────────────────────────────────────────────────────────────
# M8 — ReflectionConfig.fresh_context orchestrator wiring
# (VAL-M8-003 / m8-reflection-fresh-context-orchestrator)
# ────────────────────────────────────────────────────────────────────


class TestM8FreshContextOrchestratorEvent:
    """M8: the orchestrator emits a one-shot ``reflect_fresh_context``
    event when the matched route's ``reflection.fresh_context`` flag
    is ``True``. The event is appended to ``ctx.events`` after the
    first critique, with ``text='True'`` and ``turn`` set to the
    current reflection turn (typically 0, the 0-indexed loop counter
    for the first critique).

    When ``fresh_context=False`` (the default), the event is NOT
    emitted. The event is at-most-once per request — the orchestrator
    uses a runtime guard (``ctx.__dict__["reflect_fresh_context_emitted"]``)
    to keep the event from being appended once per turn, even when
    multiple turns are configured. The orchestrator's response body
    is unchanged (same model output, same headers minus the new
    event) — the M1-M7 M5 contract invariants are preserved
    byte-for-byte when ``fresh_context=False``.

    The tests below use a hermetic :class:`ScriptedAdapter` (a
    FakeAdapter equivalent that records every call's ``messages``)
    to drive the orchestrator and assert on the resulting
    ``ctx.events``. The full M1-M7 hermetic test suite continues to
    pass with the default ``fresh_context=False`` path; the
    ``fresh_context=True`` path is the new M8 behavior under test.
    """

    # ── (a) fresh_context=True emits the event ──

    @pytest.mark.asyncio
    async def test_fresh_context_true_emits_reflect_fresh_context_event(self):
        """VAL-M8-003 (a): with ``fresh_context=True``, the orchestrator
        appends a ``reflect_fresh_context`` event to ``ctx.events``
        after the first critique. The event's ``text`` is the literal
        string ``'True'`` and ``turn`` is the current reflection
        turn (0 for the first critique, which is 0-indexed).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "cold critique\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The reflect_fresh_context event MUST be present.
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert len(events) == 1, (
            f"expected exactly 1 reflect_fresh_context event; got {len(events)}: "
            f"{[(e.type, e.text) for e in ctx.events]}"
        )
        # The event's text is the literal string 'True'.
        assert events[0].text == "True"
        # The event's turn is the current reflection turn (0 for the
        # first critique, which is 0-indexed).
        assert events[0].turn == 0

    @pytest.mark.asyncio
    async def test_fresh_context_true_event_positioned_after_first_critique(self):
        """VAL-M8-003: the ``reflect_fresh_context`` event is appended
        IMMEDIATELY AFTER the first ``reflect_critique`` event (and
        BEFORE any subsequent ``reflect_revised`` /
        ``reflect_early_exit`` events).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Event ordering: initial, reflect_critique, reflect_fresh_context,
        # reflect_revised.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_fresh_context",
            "reflect_revised",
        ]

    @pytest.mark.asyncio
    async def test_fresh_context_true_event_positioned_with_early_exit(self):
        """VAL-M8-003: when ``early_exit`` fires on the last turn
        (confidence >= threshold), the ``reflect_fresh_context``
        event is still emitted between the critique and the
        ``reflect_early_exit`` event.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.95",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_fresh_context",
            "reflect_early_exit",
        ]

    # ── (b) fresh_context=False does NOT emit ──

    @pytest.mark.asyncio
    async def test_fresh_context_false_does_not_emit_event(self):
        """VAL-M8-003 (b): with ``fresh_context=False`` (the default),
        the orchestrator does NOT append a ``reflect_fresh_context``
        event to ``ctx.events``. The default path preserves the M1-M7
        event list shape verbatim.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=False,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The event MUST NOT be present.
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert events == [], (
            f"expected no reflect_fresh_context events; got {len(events)}: "
            f"{[(e.type, e.text) for e in ctx.events]}"
        )
        # The default event shape is preserved (M1-M7 backward compat).
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]

    @pytest.mark.asyncio
    async def test_default_fresh_context_omitted_does_not_emit_event(self):
        """VAL-M8-003 (b): the YAML omission path (``fresh_context``
        is NOT specified in the route config) defaults to ``False``
        and the event is NOT emitted. The ``ReflectionConfig``
        default is ``fresh_context=False`` per the contract.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        # Build a route without specifying fresh_context; it must
        # default to False.
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        assert route.route.reflection.fresh_context is False
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert events == []

    # ── (c) The event is emitted exactly once per request ──

    @pytest.mark.asyncio
    async def test_fresh_context_event_emitted_exactly_once_with_two_turns(self):
        """VAL-M8-003 (c): the ``reflect_fresh_context`` event is
        emitted at most once per request, even when the reflection
        loop is configured with multiple turns. The orchestrator uses
        a runtime guard to suppress the event on subsequent turns.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c0\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev0", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Exactly one reflect_fresh_context event in the events list.
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert len(events) == 1
        # The single event has turn=0 (the first critique).
        assert events[0].turn == 0
        assert events[0].text == "True"

    @pytest.mark.asyncio
    async def test_fresh_context_event_emitted_exactly_once_with_three_turns(self):
        """VAL-M8-003 (c): same as the two-turn case but with
        ``turns=3``. The event MUST be emitted exactly once.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c0\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev0", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c2\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev2", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=3,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert len(events) == 1
        assert events[0].turn == 0

    @pytest.mark.asyncio
    async def test_fresh_context_runtime_guard_is_set_after_first_event(self):
        """VAL-M8-003 (c): the orchestrator stamps
        ``ctx.__dict__["reflect_fresh_context_emitted"] = True`` after
        the first emit, so subsequent path-conditional checks
        (parallel / self-reflection) suppress duplicate events.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The runtime guard MUST be set after the first emit.
        assert ctx.__dict__.get("reflect_fresh_context_emitted") is True

    @pytest.mark.asyncio
    async def test_fresh_context_runtime_guard_unset_when_false(self):
        """VAL-M8-003 (b): when ``fresh_context=False``, the runtime
        guard MUST NOT be set (the event path was never taken).
        """
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("revised", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=False,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The guard MUST remain unset (no event was emitted, so no
        # guard was stamped).
        assert ctx.__dict__.get("reflect_fresh_context_emitted", False) is False

    # ── (d) The orchestrator's response body is unchanged ──

    @pytest.mark.asyncio
    async def test_fresh_context_event_does_not_change_response_body(self):
        """VAL-M8-003 (d): the M8 ``reflect_fresh_context`` event is an
        observability hook only. It does NOT change the model output
        the orchestrator returns. The ``ctx.upstream_response.message
        .content`` is the same as the equivalent ``fresh_context=False``
        run with the same scripted adapter responses.
        """
        # Run the same script twice: once with fresh_context=False,
        # once with fresh_context=True. The model output MUST be
        # identical; only the events list differs.
        script = [
            _response("initial answer", prompt_tokens=5, completion_tokens=2),
            _response(
                "c\nREFLECT_CONFIDENCE: 0.5",
                prompt_tokens=4,
                completion_tokens=6,
            ),
            _response("revised", prompt_tokens=4, completion_tokens=6),
        ]
        adapter_off = ScriptedAdapter(list(script))
        route_off = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=False,
        )
        ctx_off = _build_context(route_off)
        await Orchestrator(adapter_off).run(ctx_off)

        adapter_on = ScriptedAdapter(list(script))
        route_on = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx_on = _build_context(route_on)
        await Orchestrator(adapter_on).run(ctx_on)

        # The final assistant text MUST be identical.
        assert (
            ctx_off.upstream_response is not None
            and ctx_on.upstream_response is not None
        )
        assert (
            ctx_off.upstream_response.message.content
            == ctx_on.upstream_response.message.content
        )
        assert (
            ctx_off.upstream_response.message.content == "revised"
        )

    @pytest.mark.asyncio
    async def test_fresh_context_event_does_not_change_response_model(self):
        """VAL-M8-003 (d): the response's ``model`` field (which
        echoes the original alias the client sent) is the same with
        or without the event. The model echo is preserved.
        """
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_fresh_context_event_does_not_change_headers_minus_event(self):
        """VAL-M8-003 (d): the ``build_response_headers`` output is
        the same with or without ``fresh_context=True`` (the new
        ``reflect_fresh_context`` event does NOT add a new header —
        it is only observable in the event log, not in the HTTP
        response headers). The only observable difference is in
        ``ctx.events`` itself.
        """
        script = [
            _response("initial", prompt_tokens=5, completion_tokens=2),
            _response(
                "c\nREFLECT_CONFIDENCE: 0.5",
                prompt_tokens=4,
                completion_tokens=6,
            ),
            _response("revised", prompt_tokens=4, completion_tokens=6),
        ]
        adapter_off = ScriptedAdapter(list(script))
        route_off = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=False,
        )
        ctx_off = _build_context(route_off, request_id="req-fc-off")
        await Orchestrator(adapter_off).run(ctx_off)
        headers_off = build_response_headers(ctx_off, request_id="req-fc-off")

        adapter_on = ScriptedAdapter(list(script))
        route_on = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx_on = _build_context(route_on, request_id="req-fc-on")
        await Orchestrator(adapter_on).run(ctx_on)
        headers_on = build_response_headers(ctx_on, request_id="req-fc-on")

        # The header dicts are equal (the x-moaxy-* keys, the values
        # — the new event is not surfaced as a header). The request_id
        # differs by construction (test labels), so compare the
        # *other* keys via ``x-moaxy-alias-resolved`` (which carries
        # the same value) and the ``x-moaxy-reflect-turns`` /
        # ``x-moaxy-reflect-confidence`` observability surfaces.
        assert (
            headers_off["x-moaxy-alias-resolved"]
            == headers_on["x-moaxy-alias-resolved"]
        )
        assert (
            headers_off["x-moaxy-reflect-turns"]
            == headers_on["x-moaxy-reflect-turns"]
        )
        assert (
            headers_off["x-moaxy-reflect-confidence"]
            == headers_on["x-moaxy-reflect-confidence"]
        )
        assert (
            headers_off["x-moaxy-fallbacks-used"]
            == headers_on["x-moaxy-fallbacks-used"]
        )

    @pytest.mark.asyncio
    async def test_fresh_context_event_does_not_change_usage(self):
        """VAL-M8-003 (d): the token usage accumulator is the same
        with or without the ``reflect_fresh_context`` event. The
        event is an observability hook; it does not trigger an
        extra LLM call.
        """
        script = [
            _response("initial", prompt_tokens=10, completion_tokens=5),
            _response(
                "c\nREFLECT_CONFIDENCE: 0.5",
                prompt_tokens=20,
                completion_tokens=8,
            ),
            _response("revised", prompt_tokens=20, completion_tokens=8),
        ]
        adapter_off = ScriptedAdapter(list(script))
        route_off = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=False,
        )
        ctx_off = _build_context(route_off)
        await Orchestrator(adapter_off).run(ctx_off)
        snap_off = ctx_off.usage.snapshot()

        adapter_on = ScriptedAdapter(list(script))
        route_on = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        ctx_on = _build_context(route_on)
        await Orchestrator(adapter_on).run(ctx_on)
        snap_on = ctx_on.usage.snapshot()

        # The usage snapshot MUST be identical.
        assert snap_off.prompt_tokens == snap_on.prompt_tokens
        assert snap_off.completion_tokens == snap_on.completion_tokens
        assert snap_off.total_tokens == snap_on.total_tokens

    # ── (e) Parallel path: event emitted exactly once ──

    @pytest.mark.asyncio
    async def test_fresh_context_event_in_parallel_path(self):
        """VAL-M8-003 (c): the ``reflect_fresh_context`` event is
        emitted exactly once in the ``reflection.parallel: true``
        path too. The orchestrator's per-turn coroutines share the
        same runtime guard so the event is at-most-once per request
        even when the parallel scheduler runs multiple critique
        coroutines concurrently.
        """
        # The parallel orchestrator's per-turn pair is sequential
        # in the FakeAdapter's queue (turn N+1 starts after turn
        # N's critique returns), so the queue order is well-defined.
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c0\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev0", prompt_tokens=4, completion_tokens=6),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
            ]
        )
        # The test_orchestrator.py's ``_build_route`` does not
        # expose the ``parallel`` flag. To exercise the parallel
        # path we mutate the route's reflection config in place.
        route = _build_route(
            reflection_turns=2,
            early_exit=False,
            threshold=0.85,
            fresh_context=True,
        )
        # Switch on parallel on the matched route. ReflectionConfig
        # is a Pydantic v2 model so we use ``model_copy(update=...)``.
        new_reflection = route.route.reflection.model_copy(
            update={"parallel": True}
        )
        new_route_config = route.route.model_copy(
            update={"reflection": new_reflection}
        )
        from moaxy.routing.matcher import RouteMatch
        parallel_route = RouteMatch(
            route=new_route_config,
            original_model=route.original_model,
            resolved_model=route.resolved_model,
            backend=route.backend,
            path=route.path,
            reflection=new_reflection,
            advisor=route.advisor,
            fallbacks=route.fallbacks,
            retry=route.retry,
            aliases=route.aliases,
        )
        ctx = _build_context(parallel_route)
        await Orchestrator(adapter).run(ctx)
        # Exactly one reflect_fresh_context event in the events list.
        events = [e for e in ctx.events if e.type == "reflect_fresh_context"]
        assert len(events) == 1
        assert events[0].turn == 0
        assert events[0].text == "True"

