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
