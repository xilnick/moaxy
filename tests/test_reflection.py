"""Tests for :mod:`moaxy.pipeline.orchestrator` reflection behaviour.

The M2 reflection loop is the smallest, most testable unit in the
moaxy pipeline. It runs the configured number of critique/revision
turns, parses a ``REFLECT_CONFIDENCE: <float>`` line off each
critique, and may short-circuit on the early-exit threshold. This
file pins every property the validation contract asserts in the
"Reflection" area: turns=0 passthrough, turns=1 / 2 with and
without early exit, missing confidence, custom system prompts,
oversized critique tolerance, usage summation, header accuracy,
event ordering, event count vs. LLM call count, request immutability,
and ``finish_reason`` preservation.

The tests use the shared :class:`FakeAdapter` defined in
``tests/fixtures/fake_adapter.py``. The fake records every call's
``model``/``messages``/``**kwargs`` so the tests can assert on what
the orchestrator forwarded. The test suite runs in well under 2s
with no real LLM.

Mirrors the file's design but is the dedicated, focused reflection
suite — :file:`tests/test_orchestrator.py` is the broader coverage
of the whole pipeline, including advisor. This file stays narrower
on reflection and goes deeper on the cases the contract pins.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

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
from moaxy.models.config import RouteMatch as ConfigRouteMatch
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.pipeline.prompts import DEFAULT_REFLECT_PROMPT
from moaxy.routing.matcher import RouteMatch
from tests.fixtures.fake_adapter import FakeAdapter

# ────────────────────────────────────────────────────────────────────
# Response / route / context factories
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


def _critique_response(
    body: str,
    confidence: float | None,
    *,
    model: str = "minimax-m3:cloud",
    prompt_tokens: int = 4,
    completion_tokens: int = 6,
) -> ChatResponse:
    """Build a critique response. ``confidence=None`` omits the line entirely."""
    if confidence is None:
        text = body
    else:
        text = f"{body}\nREFLECT_CONFIDENCE: {confidence}"
    return _response(
        text, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def _build_route(
    *,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    parallel: bool = False,
    system_prompt: str | None = None,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    trust_verbal: float = 0.6,
    trust_score: float = 0.4,
) -> RouteMatch:
    config_route = RouteConfig(
        name="reflection-test-route",
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases={"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
            parallel=parallel,
            system_prompt=system_prompt,
            trust_verbal=trust_verbal,
            trust_score=trust_score,
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
    request_id: str = "req-1",
    request_messages: list[dict[str, Any]] | None = None,
    request_extra: dict[str, Any] | None = None,
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
# 1. turns=0 passthrough
# ────────────────────────────────────────────────────────────────────


class TestTurns0Passthrough:
    """``turns=0`` means no reflection runs; the pipeline is a single LLM call."""

    @pytest.mark.asyncio
    async def test_turns_0_passthrough(self):
        """VAL-PIPE-001: turns=0 → exactly 1 LLM call and 1 'initial' event."""
        adapter = FakeAdapter(
            [_response("initial answer", prompt_tokens=10, completion_tokens=5)]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # LLM call count = 1.
        assert len(adapter.calls) == 1
        # Event list contains exactly one event, of type 'initial'.
        assert [e.type for e in ctx.events] == ["initial"]
        # The 'initial' event carries the assistant content.
        assert ctx.events[0].text == "initial answer"
        # The final response content is the initial answer (no revise).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial answer"
        # No reflection-related headers carry non-zero values.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "0"
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.0

    @pytest.mark.asyncio
    async def test_turns_0_initial_call_forwards_messages(self):
        """The initial call's messages match the request body verbatim."""
        adapter = FakeAdapter([_response("ok")])
        route = _build_route(reflection_turns=0)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        ctx = _build_context(route, request_messages=messages)
        await Orchestrator(adapter).run(ctx)
        assert adapter.calls[0]["messages"] == messages

    @pytest.mark.asyncio
    async def test_turns_0_no_fallbacks_used_header(self):
        adapter = FakeAdapter([_response("ok")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-fallbacks-used"] == "0"


# ────────────────────────────────────────────────────────────────────
# 2. turns=1 with revision (early_exit=False, or confidence below threshold)
# ────────────────────────────────────────────────────────────────────


class TestTurns1WithRevision:
    """``turns=1`` with a non-clearing confidence runs the full revise loop."""

    @pytest.mark.asyncio
    async def test_turns_1_with_revision(self):
        """VAL-PIPE-002: turns=1 with confidence below threshold → 3 calls."""
        adapter = FakeAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _critique_response("a critique", confidence=0.5),
                _response("revised answer", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=False, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 3
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised answer"

    @pytest.mark.asyncio
    async def test_turns_1_revised_event_carries_revised_text(self):
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("needs work", confidence=0.4),
                _response("REVISED_HERE"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        revised_events = [e for e in ctx.events if e.type == "reflect_revised"]
        assert len(revised_events) == 1
        assert revised_events[0].text == "REVISED_HERE"
        assert revised_events[0].turn == 0


# ────────────────────────────────────────────────────────────────────
# 3. turns=1 with early_exit at 0.95 (clears threshold)
# ────────────────────────────────────────────────────────────────────


class TestTurns1EarlyExitAt0_95:
    """``turns=1`` with confidence 0.95 ≥ threshold 0.85 short-circuits."""

    @pytest.mark.asyncio
    async def test_turns_1_early_exit_at_0_95(self):
        """VAL-PIPE-003: 0.95 ≥ 0.85 → 2 LLM calls, reflect_early_exit emitted."""
        adapter = FakeAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _critique_response("looks good", confidence=0.95),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 2 LLM calls: initial + critique only. No revision.
        assert len(adapter.calls) == 2
        # Event ordering: initial → reflect_critique → reflect_early_exit.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_early_exit",
        ]
        # The early-exit event carries the same model and turn as the critique.
        early = next(e for e in ctx.events if e.type == "reflect_early_exit")
        assert early.turn == 0
        assert early.model == "minimax-m3:cloud"
        # The final response is the initial answer (no revision ran).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial answer"
        # Header reflects the last parsed confidence.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.95
        assert headers["x-moaxy-reflect-turns"] == "1"


# ────────────────────────────────────────────────────────────────────
# 4. turns=1 with early_exit at 0.50 (does not clear threshold)
# ────────────────────────────────────────────────────────────────────


class TestTurns1EarlyExitAt0_50:
    """``turns=1`` with confidence 0.50 < threshold 0.85 continues to revision."""

    @pytest.mark.asyncio
    async def test_turns_1_early_exit_at_0_50(self):
        """VAL-PIPE-004: 0.50 < 0.85 → 3 LLM calls, no early-exit event."""
        adapter = FakeAdapter(
            [
                _response("initial answer", prompt_tokens=5, completion_tokens=2),
                _critique_response("not great", confidence=0.50),
                _response("revised answer", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 3
        # No early-exit event was emitted.
        types = [e.type for e in ctx.events]
        assert "reflect_early_exit" not in types
        # The full revise pair ran.
        assert types == ["initial", "reflect_critique", "reflect_revised"]
        # Header still records the last parsed confidence (0.50).
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.50

    @pytest.mark.asyncio
    async def test_turns_1_confidence_just_below_threshold_continues(self):
        """0.84 < 0.85 by a hair → revision runs, no early exit."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("hmm", confidence=0.84),
                _response("revised"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]


# ────────────────────────────────────────────────────────────────────
# 5. Missing confidence line
# ────────────────────────────────────────────────────────────────────


class TestMissingConfidenceLine:
    """A critique without ``REFLECT_CONFIDENCE`` parses to 0.0; revision runs."""

    @pytest.mark.asyncio
    async def test_missing_confidence_line_continues(self):
        """VAL-PIPE-005: missing line → confidence=0.0, revision runs."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _critique_response("a critique with no confidence line", confidence=None),
                _response("revised after missing line"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 3
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]
        # No early-exit event was emitted.
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        # The last parsed confidence is 0.0 (recorded for the response header).
        assert ctx.__dict__.get("last_confidence") == 0.0

    @pytest.mark.asyncio
    async def test_missing_confidence_with_turns_2_runs_both_turns(self):
        """When both critiques lack a confidence line, the loop runs to completion."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("no line here", confidence=None),
                _response("rev1"),
                _critique_response("still no line", confidence=None),
                _response("rev2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 5
        assert "reflect_early_exit" not in [e.type for e in ctx.events]


# ────────────────────────────────────────────────────────────────────
# 6. turns=2 full loop
# ────────────────────────────────────────────────────────────────────


class TestTurns2FullLoop:
    """``turns=2`` with both confidences below threshold → 5 calls."""

    @pytest.mark.asyncio
    async def test_turns_2_full_loop(self):
        """VAL-PIPE-006: 2 critique + 2 revision + 1 initial = 5 LLM calls."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _critique_response("c1", confidence=0.5),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
                _critique_response("c2", confidence=0.5),
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
        # The final response content is the second revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev2"
        # Both critique events have sequential turn numbers.
        critique_turns = [e.turn for e in ctx.events if e.type == "reflect_critique"]
        assert critique_turns == [0, 1]

    @pytest.mark.asyncio
    async def test_turns_2_final_response_is_last_revised_answer(self):
        """The 'revised' events' text matches the final response content."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.4),
                _response("rev1-text"),
                _critique_response("c2", confidence=0.4),
                _response("rev2-text-FINAL"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev2-text-FINAL"


# ────────────────────────────────────────────────────────────────────
# 7. turns=2 with turn-1 early-exit
# ────────────────────────────────────────────────────────────────────


class TestTurns2Turn1EarlyExit:
    """``turns=2`` with turn 1 above threshold → stops after the first revision."""

    @pytest.mark.asyncio
    async def test_turns_2_turn_1_early_exit(self):
        """VAL-PIPE-007: turn 1 hits threshold → 3 calls, stop after turn 1."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _critique_response("c1", confidence=0.95),
                _response("rev1", prompt_tokens=4, completion_tokens=6),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 3 LLM calls: initial + critique + revision. The second turn never runs.
        assert len(adapter.calls) == 3
        # Event order: initial → critique → revised → early_exit.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_early_exit",
        ]
        # The final answer is the first revision (rev1).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev1"
        # The early-exit event records the revision's model.
        early = next(e for e in ctx.events if e.type == "reflect_early_exit")
        assert early.turn == 0
        # Only one critique ran (turn 2 was skipped).
        critiques = [e for e in ctx.events if e.type == "reflect_critique"]
        assert len(critiques) == 1
        # Header records 1 reflect turn (only 1 critique emitted).
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "1"


# ────────────────────────────────────────────────────────────────────
# 8. Custom system_prompt
# ────────────────────────────────────────────────────────────────────


class TestCustomSystemPrompt:
    """A route's ``reflection.system_prompt`` is forwarded to critique & revision."""

    @pytest.mark.asyncio
    async def test_custom_system_prompt_forwarded(self):
        custom = "You are a strict code reviewer. Be terse."
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            system_prompt=custom,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # Three calls: initial (no system), critique, revision.
        assert len(adapter.calls) == 3
        # The initial call has no system message in its messages list.
        assert adapter.system_messages[0] is None
        # The critique and revision calls both carry the custom system prompt.
        assert adapter.system_messages[1] == {"role": "system", "content": custom}
        assert adapter.system_messages[2] == {"role": "system", "content": custom}

    @pytest.mark.asyncio
    async def test_default_system_prompt_used_when_unset(self):
        """Without ``reflection.system_prompt`` the default reflect prompt is used."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("r"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # The critique and revision calls both carry the default prompt.
        assert adapter.system_messages[1] == {
            "role": "system",
            "content": DEFAULT_REFLECT_PROMPT,
        }
        assert adapter.system_messages[2] == {
            "role": "system",
            "content": DEFAULT_REFLECT_PROMPT,
        }

    @pytest.mark.asyncio
    async def test_custom_prompt_appears_as_first_message_in_critique(self):
        """The system message is the first message in the critique call."""
        custom = "STRICT_REVIEWER_PROMPT"
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("r"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            system_prompt=custom,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        critique_messages = adapter.calls[1]["messages"]
        assert critique_messages[0]["role"] == "system"
        assert critique_messages[0]["content"] == custom


# ────────────────────────────────────────────────────────────────────
# 9. parallel:true (sequential, not breaking)
# ────────────────────────────────────────────────────────────────────


class TestParallelTrue:
    """``reflection.parallel: true`` engages the M4 parallel path.

    The parallel path uses :func:`asyncio.gather` to schedule the
    per-turn critique+revision pairs concurrently. The chain is
    preserved: turn N+1's critique uses turn N's revision as input.
    The contract pins content equivalence to the sequential path
    (VAL-PIPE-009); no strict timing assertion is made.
    """

    @pytest.mark.asyncio
    async def test_parallel_true_runs_equivalent_to_sequential(self):
        """``reflection.parallel: true, turns: 2`` matches the sequential
        final content (VAL-PIPE-009).
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
                _critique_response("c2", confidence=0.5),
                _response("rev2"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 5 calls; the parallel path uses asyncio.gather but the chain
        # is preserved so the LLM call order matches the sequential
        # reference.
        assert len(adapter.calls) == 5
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
        ]
        # Content equivalence: the final answer is rev2, matching
        # the sequential path.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev2"

    @pytest.mark.asyncio
    async def test_parallel_true_does_not_break_early_exit(self):
        """Early-exit still fires under ``parallel: true``."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.95),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 3
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_early_exit",
        ]
        # Content equivalence: the final answer is rev1 (the
        # post-early-exit revision), matching the sequential path.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev1"


# ────────────────────────────────────────────────────────────────────
# 10. Oversized critique tolerance
# ────────────────────────────────────────────────────────────────────


class TestOversizedCritiqueTolerance:
    """A multi-megabyte critique does not crash the orchestrator."""

    @pytest.mark.asyncio
    async def test_oversized_critique_runs_to_revision(self):
        long_body = "x" * 200_000
        long_text = long_body + "\nREFLECT_CONFIDENCE: 0.5"
        adapter = FakeAdapter(
            [
                _response("initial"),
                _response(long_text),
                _response("revised after huge critique"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=False, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # The pipeline ran to completion.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "revised after huge critique"
        # The full oversized text is preserved on the critique event.
        critique_events = [e for e in ctx.events if e.type == "reflect_critique"]
        assert len(critique_events) == 1
        assert critique_events[0].text == long_text
        # The parsed confidence is still 0.5.
        assert ctx.__dict__.get("last_confidence") == 0.5

    @pytest.mark.asyncio
    async def test_oversized_critique_with_early_exit_path(self):
        """An oversized critique can also drive an early-exit path."""
        long_body = "y" * 100_000
        long_text = long_body + "\nREFLECT_CONFIDENCE: 0.95"
        adapter = FakeAdapter(
            [
                _response("initial"),
                _response(long_text),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 2
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_early_exit",
        ]
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial"


# ────────────────────────────────────────────────────────────────────
# 11. Usage summation across all calls
# ────────────────────────────────────────────────────────────────────


class TestUsageSummation:
    """Usage is summed across every LLM call (initial, critiques, revisions)."""

    @pytest.mark.asyncio
    async def test_usage_summation_across_all_calls(self):
        """VAL-PIPE-022: full-turns reflection sums 5 calls' worth of usage."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _critique_response("c1", confidence=0.5, prompt_tokens=20, completion_tokens=8),
                _response("rev1", prompt_tokens=20, completion_tokens=8),
                _critique_response("c2", confidence=0.5, prompt_tokens=20, completion_tokens=8),
                _response("rev2", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        snap = ctx.usage.snapshot()
        # 10 + 20*4 prompt; 5 + 8*4 completion.
        assert snap.prompt_tokens == 10 + 20 * 4
        assert snap.completion_tokens == 5 + 8 * 4
        assert snap.total_tokens == snap.prompt_tokens + snap.completion_tokens

    @pytest.mark.asyncio
    async def test_usage_preserved_in_final_response(self):
        """The final ``ctx.upstream_response.usage`` equals the accumulated snapshot."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=7, completion_tokens=3),
                _critique_response("c", confidence=0.5, prompt_tokens=11, completion_tokens=5),
                _response("r", prompt_tokens=13, completion_tokens=7),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert ctx.upstream_response is not None
        snap = ctx.usage.snapshot()
        assert ctx.upstream_response.usage.prompt_tokens == snap.prompt_tokens
        assert ctx.upstream_response.usage.completion_tokens == snap.completion_tokens
        assert ctx.upstream_response.usage.total_tokens == snap.total_tokens

    @pytest.mark.asyncio
    async def test_usage_per_call_logged_in_adapter(self):
        """The fake's per-call usage log matches the script."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _critique_response("c", confidence=0.5, prompt_tokens=20, completion_tokens=8),
                _response("rev", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # The fake's per-call usage log carries the same values the script set.
        assert len(adapter.usages) == 3
        assert adapter.usages[0].prompt_tokens == 10
        assert adapter.usages[0].completion_tokens == 5
        assert adapter.usages[1].prompt_tokens == 20
        assert adapter.usages[2].prompt_tokens == 20


# ────────────────────────────────────────────────────────────────────
# 12. Usage summation on early-exit path
# ────────────────────────────────────────────────────────────────────


class TestUsageSummationOnEarlyExit:
    """Early-exit does not include the would-have-been revision's usage."""

    @pytest.mark.asyncio
    async def test_usage_summation_on_early_exit_path(self):
        """VAL-PIPE-023: early-exit → only 2 calls' usage, no revision usage."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _critique_response("c1", confidence=0.95, prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        snap = ctx.usage.snapshot()
        # Only 2 calls happened: 10+20 = 30 prompt; 5+8 = 13 completion.
        assert snap.prompt_tokens == 30
        assert snap.completion_tokens == 13
        assert snap.total_tokens == 43

    @pytest.mark.asyncio
    async def test_usage_summation_turns_2_turn1_early_exit(self):
        """Turn 1 early-exit (3 calls) sums initial + critique + revision usage."""
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _critique_response("c1", confidence=0.95, prompt_tokens=20, completion_tokens=8),
                _response("rev1", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        snap = ctx.usage.snapshot()
        # 3 calls: 10+20+20 prompt; 5+8+8 completion. Turn 2 (which would
        # have added 2 more calls) never ran.
        assert snap.prompt_tokens == 50
        assert snap.completion_tokens == 21


# ────────────────────────────────────────────────────────────────────
# 13. Header accuracy
# ────────────────────────────────────────────────────────────────────


class TestHeaderAccuracy:
    """``build_response_headers`` accurately reflects pipeline state."""

    @pytest.mark.asyncio
    async def test_header_request_id_matches_context(self):
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route, request_id="req-xyz")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-request-id"] == "req-xyz"

    @pytest.mark.asyncio
    async def test_header_alias_resolved(self):
        adapter = FakeAdapter([_response("x")])
        route = _build_route(
            original_model="coder-pro", resolved_model="minimax-m3:cloud"
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_header_reflect_turns_counts_critique_events(self):
        """VAL-PIPE-037: header equals the number of critique events emitted."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
                _critique_response("c2", confidence=0.5),
                _response("rev2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "2"

    @pytest.mark.asyncio
    async def test_header_reflect_turns_zero_for_passthrough(self):
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "0"

    @pytest.mark.asyncio
    async def test_header_reflect_confidence_is_last_value(self):
        """VAL-PIPE-038: header equals the LAST parsed confidence."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
                _critique_response("c2", confidence=0.9),
                _response("rev2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.9

    @pytest.mark.asyncio
    async def test_header_fallbacks_used_is_json_list(self):
        """When fallbacks ran, the header is a JSON-encoded list."""
        from moaxy.adapters.base import UpstreamError

        adapter = FakeAdapter(
            [
                UpstreamError("err", status_code=500, body="err"),
                _response("fallback", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert json.loads(headers["x-moaxy-fallbacks-used"]) == ["minimax-m2.7:cloud"]

    @pytest.mark.asyncio
    async def test_header_fallbacks_used_zero_on_happy_path(self):
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-fallbacks-used"] == "0"

    @pytest.mark.asyncio
    async def test_header_advisor_model_absent_when_advisor_disabled(self):
        """With no advisor configured, the advisor header is not present."""
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0, advisor_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert "x-moaxy-advisor-model" not in headers


# ────────────────────────────────────────────────────────────────────
# 14. Event ordering
# ────────────────────────────────────────────────────────────────────


class TestEventOrdering:
    """Events follow the documented order: initial, then per-turn critique+revised."""

    @pytest.mark.asyncio
    async def test_event_ordering_turns_0(self):
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert [e.type for e in ctx.events] == ["initial"]

    @pytest.mark.asyncio
    async def test_event_ordering_initial_first_always(self):
        """VAL-PIPE-033: events list always starts with 'initial'."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.95),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.events[0].type == "initial"

    @pytest.mark.asyncio
    async def test_event_ordering_turns_2_full_loop(self):
        """The ordering for turns=2 full loop is initial, [critique, revised] x2."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("r1"),
                _critique_response("c2", confidence=0.5),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Each 'reflect_revised' follows its 'reflect_critique' at the same turn.
        events = ctx.events
        for i in range(1, len(events)):
            if events[i].type == "reflect_revised":
                assert events[i - 1].type == "reflect_critique"
                assert events[i - 1].turn == events[i].turn

    @pytest.mark.asyncio
    async def test_event_turn_numbers_in_sequential_order(self):
        """Critique and revised events carry 0, 1, … in source order."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("r0"),
                _critique_response("c1", confidence=0.5),
                _response("r1"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        turn_list = [
            e.turn
            for e in ctx.events
            if e.type in {"reflect_critique", "reflect_revised"}
        ]
        assert turn_list == [0, 0, 1, 1]

    @pytest.mark.asyncio
    async def test_event_early_exit_emitted_after_revised_when_not_last_turn(self):
        """When early-exit fires on a non-last turn, the early-exit event
        follows the revised event (the revision is still produced)."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.95),
                _response("rev1"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_early_exit",
        ]


# ────────────────────────────────────────────────────────────────────
# 15. Event count matches LLM call count
# ────────────────────────────────────────────────────────────────────


class TestEventCountVsCallCount:
    """The relationship between events and LLM calls is deterministic."""

    @pytest.mark.asyncio
    async def test_event_count_equals_call_count_for_full_loop(self):
        """VAL-PIPE-027: turns=2 full loop → 5 events, 5 calls (1:1)."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("r1"),
                _critique_response("c2", confidence=0.5),
                _response("r2"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 5 calls, 5 events (1:1 mapping, no early-exit).
        assert len(adapter.calls) == 5
        assert len(ctx.events) == 5

    @pytest.mark.asyncio
    async def test_event_count_equals_call_count_for_turns_1_revise(self):
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("rev"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 calls, 3 events (1:1 mapping).
        assert len(adapter.calls) == 3
        assert len(ctx.events) == 3

    @pytest.mark.asyncio
    async def test_event_count_exceeds_call_count_for_early_exit_last_turn(self):
        """On the LAST turn, early-exit emits 1 event without a call."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.95),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 2 calls, 3 events (the early-exit event has no matching call).
        assert len(adapter.calls) == 2
        assert len(ctx.events) == 3
        # The extra event is exactly one 'reflect_early_exit'.
        early = [e for e in ctx.events if e.type == "reflect_early_exit"]
        assert len(early) == 1

    @pytest.mark.asyncio
    async def test_event_count_exceeds_call_count_for_early_exit_non_last_turn(self):
        """On a NON-LAST turn, early-exit emits 1 extra event AFTER the revision."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.95),
                _response("rev1"),
            ]
        )
        route = _build_route(reflection_turns=2, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 calls, 4 events (1 critique + 1 revised + 1 initial + 1 early_exit).
        assert len(adapter.calls) == 3
        assert len(ctx.events) == 4
        # The extra event is exactly one 'reflect_early_exit'.
        early = [e for e in ctx.events if e.type == "reflect_early_exit"]
        assert len(early) == 1

    @pytest.mark.asyncio
    async def test_event_count_for_turns_0(self):
        """turns=0 → 1 call, 1 event (just 'initial')."""
        adapter = FakeAdapter([_response("x")])
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert len(ctx.events) == 1
        assert ctx.events[0].type == "initial"


# ────────────────────────────────────────────────────────────────────
# 16. No request mutation
# ────────────────────────────────────────────────────────────────────


class TestNoRequestMutation:
    """The pipeline does not mutate the request body the client sent."""

    @pytest.mark.asyncio
    async def test_request_messages_not_mutated(self):
        """VAL-PIPE-039: the request messages list is not mutated."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("rev"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        original = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        snapshot = json.dumps(original)
        ctx = _build_context(route, request_messages=list(original))
        await Orchestrator(adapter).run(ctx)
        assert json.dumps(ctx.request["messages"]) == snapshot
        # The original list reference is also intact.
        assert original == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    @pytest.mark.asyncio
    async def test_request_body_sampling_fields_preserved(self):
        """Non-message fields (temperature, top_p, max_tokens) survive the run."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("rev"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(
            route,
            request_extra={"temperature": 0.7, "top_p": 0.9, "max_tokens": 100},
        )
        await Orchestrator(adapter).run(ctx)
        # Every call received the sampling fields unchanged.
        for call in adapter.calls:
            assert call["temperature"] == 0.7
            assert call["top_p"] == 0.9
            assert call["max_tokens"] == 100
        # The request body itself still carries them.
        assert ctx.request["temperature"] == 0.7
        assert ctx.request["top_p"] == 0.9
        assert ctx.request["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_critique_call_messages_independent_of_request(self):
        """The critique call's messages list is a deep copy, not the request body."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("rev"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(
            route,
            request_messages=[{"role": "user", "content": "hello"}],
        )
        await Orchestrator(adapter).run(ctx)
        # The critique call's messages list is not the same object as the request.
        assert adapter.calls[1]["messages"] is not ctx.request["messages"]


# ────────────────────────────────────────────────────────────────────
# 17. finish_reason preservation
# ────────────────────────────────────────────────────────────────────


class TestFinishReasonPreservation:
    """The initial response's ``finish_reason`` is preserved on the final response."""

    @pytest.mark.asyncio
    async def test_finish_reason_preserved_passthrough(self):
        """The 'stop' finish_reason from the initial call is preserved."""
        adapter = FakeAdapter(
            [_response("ok", finish_reason="stop")]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finish_reason_preserved_after_revision(self):
        """The 'length' finish_reason from the initial call is preserved
        across the reflection loop."""
        adapter = FakeAdapter(
            [
                _response("initial", finish_reason="length"),
                _critique_response("c", confidence=0.5),
                _response("revised", finish_reason="stop"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        # The orchestrator preserves the INITIAL call's finish_reason
        # (the one the server saw first). The revision's finish_reason
        # is not promoted to the final response.
        assert ctx.upstream_response.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_finish_reason_default_stop_when_initial_unset(self):
        """When no finish_reason is set, the default 'stop' is used."""
        adapter = FakeAdapter(
            [ChatResponse(
                id="1",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content="ok"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason=None,
            )]
        )
        route = _build_route(reflection_turns=0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.finish_reason == "stop"


# ────────────────────────────────────────────────────────────────────
# 18. Weighted early-exit + DELTA 7 safety (M5)
# ────────────────────────────────────────────────────────────────────


class TestWeightedEarlyExit:
    """M5: the SEQUENTIAL reflection loop uses ``parse_weighted_signal``.

    These tests pin the M5 contract: the threshold check is now driven
    by the combined confidence+score signal (with the route's trust
    weights), and a missing ``REFLECT_CONFIDENCE:`` line is treated as
    a malformed response that does NOT short-circuit.
    """

    @pytest.mark.asyncio
    async def test_weighted_signal_used_for_threshold_check(self):
        """VAL-PIPE-EXTRA-010: trust_verbal=1.0, trust_score=0.0 → confidence only.

        With ``trust_score=0.0`` the SCORE: line is parsed but does not
        affect the threshold decision. The check reduces to the v1
        ``confidence >= threshold`` invariant.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.9),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            trust_verbal=1.0,
            trust_score=0.0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.9 >= 0.85 → early exit (no revision).
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_early_exit",
        ]
        assert len(adapter.calls) == 2
        # The combined signal equals the raw confidence (no score).
        assert ctx.__dict__["last_combined_signal"] == 0.9

    @pytest.mark.asyncio
    async def test_reflect_score_event_emitted_when_score_parsed(self):
        """VAL-PIPE-EXTRA-018: a parsed SCORE: line emits a reflect_score event."""
        # We need a critique response with both a REFLECT_CONFIDENCE
        # and a SCORE: line. Build it inline.
        crit = ChatResponse(
            id="c1",
            model="minimax-m3:cloud",
            message=Message(
                role="assistant",
                content="looks good\nREFLECT_CONFIDENCE: 0.9\nSCORE: 7",
            ),
            usage=Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
            finish_reason="stop",
        )
        adapter = FakeAdapter(
            [_response("initial"), crit, _response("revised")]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # A 'reflect_score' event is appended with text="7" and turn=0.
        score_events = [e for e in ctx.events if e.type == "reflect_score"]
        assert len(score_events) == 1
        assert score_events[0].text == "7"
        assert score_events[0].turn == 0
        # The ctx.__dict__['last_score'] is also set.
        assert ctx.__dict__["last_score"] == 7

    @pytest.mark.asyncio
    async def test_reflect_score_event_not_emitted_when_score_missing(self):
        """When the model does not emit a SCORE: line, no reflect_score event."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("revised"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # No reflect_score event in the list.
        assert "reflect_score" not in [e.type for e in ctx.events]
        # last_score is None.
        assert ctx.__dict__["last_score"] is None

    @pytest.mark.asyncio
    async def test_delta7_safety_malformed_critique_continues(self):
        """VAL-PIPE-EXTRA-016: malformed critique (no REFLECT_CONFIDENCE line).

        A critique with NO ``REFLECT_CONFIDENCE:`` line is treated as a
        malformed response. The orchestrator MUST NOT short-circuit;
        the revision runs as if ``early_exit: false`` for that turn.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("a critique with no confidence line", confidence=None),
                _response("revised after missing line"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.0)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # Even with threshold=0.0, the malformed safety rule forces
        # clears_threshold=False; the revision runs.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]

    @pytest.mark.asyncio
    async def test_explicit_zero_confidence_not_malformed(self):
        """VAL-PIPE-EXTRA-017: explicit zero confidence is NOT malformed.

        The line ``REFLECT_CONFIDENCE: 0.0`` is a successfully parsed
        value, distinct from the missing-line case. The v1 threshold
        check applies: 0.0 < 0.85 → revision runs (not the safety path).
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.0),
                _response("revised"),
            ]
        )
        route = _build_route(reflection_turns=1, early_exit=True, threshold=0.85)
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.0 < 0.85 → revision runs.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_combined_signal_exposed_on_context(self):
        """The combined signal is stored on ctx.__dict__['last_combined_signal']."""
        # With trust_verbal=0.6, trust_score=0.4 and
        # REFLECT_CONFIDENCE: 0.6, SCORE: 9: combined = 0.6*0.6 + 0.4*0.9 = 0.72
        crit = ChatResponse(
            id="c1",
            model="minimax-m3:cloud",
            message=Message(
                role="assistant",
                content="c\nREFLECT_CONFIDENCE: 0.6\nSCORE: 9",
            ),
            usage=Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
            finish_reason="stop",
        )
        adapter = FakeAdapter(
            [_response("initial"), crit, _response("revised")]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            trust_verbal=0.6,
            trust_score=0.4,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.72 < 0.85 → revision runs.
        assert len(adapter.calls) == 3
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.72)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_existing_reflection_tests_still_pass(self):
        """Regression smoke: the weighted-signal refactor does not change
        the v1 event ordering for the standard 2-turn full loop.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
                _critique_response("c2", confidence=0.5),
                _response("rev2"),
            ]
        )
        route = _build_route(
            reflection_turns=2, early_exit=True, threshold=0.85
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The standard v1 ordering is preserved.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
        ]
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev2"
