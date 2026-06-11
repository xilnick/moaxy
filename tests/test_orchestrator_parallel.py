"""Tests for the M4 parallel orchestrator path.

The M4 feature adds bounded ``asyncio.gather`` parallelism to two
stages of the pipeline:

* ``reflection.parallel: true`` — each turn's critique+revision pair
  runs concurrently with the next pair (chained, bounded
  parallelism). Turn N+1 starts as soon as turn N's critique
  returns so the revision of turn N overlaps with the critique of
  turn N+1.
* ``advisor.parallel: true`` (with ``reflection.parallel: true``) —
  the final advisor revision runs concurrently with a self-
  reflection on the original answer, taking whichever finishes
  last.

The validation contract pins the behaviour by content equivalence
to sequential (VAL-PIPE-009, VAL-PIPE-021). The tests in this
file exercise the parallel path with a scripted :class:`FakeAdapter`
and assert that the final response content equals the sequential
counterpart, that the events list is well-formed, and that the
header values are correct. No timing assertion is made.

Test layout
-----------

The tests mirror the existing sequential tests in
``tests/test_reflection.py`` and ``tests/test_advisor.py`` so the
contract pins are easy to compare side-by-side. The
:class:`FakeAdapter` is reused; its response queue is consumed in
the order the parallel orchestrator happens to call
``adapter.chat`` (which is non-deterministic across coroutines).
The tests below supply a generous response queue so the parallel
path never starves.

The tests are hermetic: no real Ollama, no on-disk plugins, no
in-process HTTP. They run in well under a second on the in-process
event loop.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from moaxy.adapters.base import (
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
    chatcmpl_id: str = "chatcmpl-parallel",
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
    if confidence is None:
        text = body
    else:
        text = f"{body}\nREFLECT_CONFIDENCE: {confidence}"
    return _response(
        text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _critique_with_score_response(
    body: str,
    confidence: float | None,
    score: int | None,
    *,
    model: str = "minimax-m3:cloud",
) -> ChatResponse:
    """Build a critique response carrying both a REFLECT_CONFIDENCE
    and an optional SCORE: line. ``confidence=None`` and
    ``score=None`` omit their respective lines entirely.
    """
    parts = [body]
    if confidence is not None:
        parts.append(f"REFLECT_CONFIDENCE: {confidence}")
    if score is not None:
        parts.append(f"SCORE: {score}")
    return _response("\n".join(parts), model=model)


def _build_route(
    *,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    reflection_parallel: bool = False,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    advisor_parallel: bool = False,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    trust_verbal: float = 0.6,
    trust_score: float = 0.4,
) -> RouteMatch:
    config_route = RouteConfig(
        name="parallel-test-route",
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases={"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
            parallel=reflection_parallel,
            trust_verbal=trust_verbal,
            trust_score=trust_score,
        ),
        advisor=AdvisorConfig(
            model=advisor_model,
            turns=advisor_turns,
            parallel=advisor_parallel,
        ),
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
    request_id: str = "req-parallel-1",
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
# 1. reflection.parallel: true — content equivalence to sequential
# ────────────────────────────────────────────────────────────────────


class TestReflectionParallelContentEquivalence:
    """``reflection.parallel: true`` produces the same final content
    as the sequential path when scripted.

    The FakeAdapter consumes its response queue in arrival order. The
    parallel orchestrator dispatches the per-turn calls in a chain so
    the queue order is well-defined: the next turn's critique is
    scheduled after the previous turn's critique returns. We provide
    scripted responses for the canonical M4 scripted case.
    """

    @pytest.mark.asyncio
    async def test_turns_2_parallel_equivalent_to_sequential(self):
        """VAL-PIPE-009: ``reflection.parallel: true, turns: 2`` matches
        the sequential final content.
        """
        # Script: initial, critique_0, rev_0, critique_1, rev_1.
        # The parallel orchestrator issues the calls in the same
        # observable order (critique_0 → rev_0 → critique_1 → rev_1)
        # because the next critique depends on the previous revision.
        adapter = FakeAdapter(
            [
                _response("initial answer"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 5 LLM calls (initial + 2 critiques + 2 revisions).
        assert len(adapter.calls) == 5
        # The final content matches the sequential path: rev1.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev1"
        # The header reflects the configured 2 turns.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "2"

    @pytest.mark.asyncio
    async def test_turns_2_parallel_no_early_exit(self):
        """VAL-PIPE-009: parallel path emits both critique and revised events
        for both turns, in source order.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 5 events after the initial: reflect_critique(0), reflect_revised(0),
        # reflect_critique(1), reflect_revised(1). Plus the initial event.
        event_types = [e.type for e in ctx.events]
        assert event_types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
        ]
        # The turns are in 0,0,1,1 order.
        turn_list = [
            e.turn
            for e in ctx.events
            if e.type in {"reflect_critique", "reflect_revised"}
        ]
        assert turn_list == [0, 0, 1, 1]

    @pytest.mark.asyncio
    async def test_turns_2_parallel_early_exit_after_turn1(self):
        """Early-exit on a non-last turn still fires under ``parallel: true``."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.95),
                _response("rev0"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 3 LLM calls: initial, critique, revision. Turn 1 is short-circuited.
        assert len(adapter.calls) == 3
        # Events: initial, reflect_critique(0), reflect_revised(0), reflect_early_exit(0).
        event_types = [e.type for e in ctx.events]
        assert event_types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_early_exit",
        ]
        # The final content is the post-early-exit revision (rev0).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev0"

    @pytest.mark.asyncio
    async def test_turns_2_parallel_early_exit_last_turn(self):
        """Early-exit on the LAST turn skips the revision; the initial
        answer is kept (matches sequential)."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.95),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 4 LLM calls: initial + 2 critiques + 1 revision (turn 1's revision
        # is skipped because the critique cleared the threshold on the
        # last turn).
        assert len(adapter.calls) == 4
        # Events: initial, critique(0), revised(0), critique(1), early_exit(1).
        event_types = [e.type for e in ctx.events]
        assert event_types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_early_exit",
        ]
        # The final content is rev0 (the post-turn-0 revision), not the
        # initial answer (the early-exit happened on the last turn, so
        # the answer the loop saw was the rev0 from the previous turn).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev0"

    @pytest.mark.asyncio
    async def test_turns_1_parallel_equivalent_to_sequential(self):
        """``turns=1, parallel=true`` produces the same final content
        as ``turns=1, parallel=false``.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 LLM calls: initial, critique, revision.
        assert len(adapter.calls) == 3
        # Final content is the revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev0"
        # Events: initial, reflect_critique, reflect_revised.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
        ]

    @pytest.mark.asyncio
    async def test_turns_0_parallel_is_passthrough(self):
        """``turns=0, parallel=true`` is a passthrough (no reflection runs)."""
        adapter = FakeAdapter([_response("initial")])
        route = _build_route(
            reflection_turns=0,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial"
        assert [e.type for e in ctx.events] == ["initial"]


# ────────────────────────────────────────────────────────────────────
# 2. advisor.parallel: true — content equivalence to sequential
# ────────────────────────────────────────────────────────────────────


class TestAdvisorParallelContentEquivalence:
    """``advisor.parallel: true`` (with ``reflection.parallel: true``)
    produces the same final content as the sequential path.

    The M4 contract is that the orchestrator runs the final advisor
    revision concurrently with a self-reflection on the original
    answer, taking whichever finishes last.
    """

    @pytest.mark.asyncio
    async def test_advisor_parallel_with_approve_matches_sequential(self):
        """``advisor.parallel=true, advisor approves`` matches sequential
        approve: the post-reflection answer is kept.
        """
        # The advisor.parallel path runs the advisor AND a
        # self-reflection concurrently. The scripted queue must
        # have enough responses for both paths (4 for the
        # reflection+advisor path, 2 for the self-reflection on
        # the initial answer). The contract pins content
        # equivalence, not the order in which the queue is
        # consumed, so we supply extra responses.
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
                _critique_response("self_c0", confidence=0.5),
                _response("self_rev0"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 6 LLM calls: initial, critique, revised, advisor, self-critique, self-revised.
        assert len(adapter.calls) == 6
        # The final answer is the post-reflection revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev0"
        # The advisor header is set.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        # The advisor events are still emitted in order.
        event_types = [e.type for e in ctx.events]
        assert "advisor" in event_types
        assert "advisor_approve" in event_types

    @pytest.mark.asyncio
    async def test_advisor_parallel_no_reflection_runs(self):
        """``advisor.parallel: true`` with ``reflection.turns: 0`` (so
        reflection.parallel has no effect) still runs the advisor.
        """
        # With ``reflection.turns: 0`` the self-reflection path
        # is not engaged (it would have nothing to do). The
        # advisor.parallel path degenerates to a sequential
        # advisor pass.
        adapter = FakeAdapter(
            [
                _response("initial"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 2 LLM calls: initial + advisor. No self-reflection runs.
        assert len(adapter.calls) == 2
        # The final answer is the initial answer (advisor approved).
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial"
        # Events: initial, advisor, advisor_approve.
        assert [e.type for e in ctx.events] == [
            "initial",
            "advisor",
            "advisor_approve",
        ]

    @pytest.mark.asyncio
    async def test_advisor_parallel_revise_replaces_answer(self):
        """``advisor.parallel=true, advisor revises`` triggers the
        primary-revision path; the final answer is the primary
        model's revision.
        """
        # The advisor.parallel path runs the advisor path
        # (3 calls: advisor, primary-revision) AND a
        # self-reflection on the initial answer (2 calls).
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _response(
                    "ADVISOR_REVISE: new feedback", model="deepseek-v4-pro:cloud"
                ),
                _response("primary-revised-final"),
                _critique_response("self_c0", confidence=0.5),
                _response("self_rev0"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 7 LLM calls: initial, critique, revised, advisor,
        # primary-revision, self-critique, self-revision.
        assert len(adapter.calls) == 7
        # The final answer is the primary model's revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "primary-revised-final"
        # The advisor events are emitted.
        event_types = [e.type for e in ctx.events]
        assert "advisor_revised" in event_types
        assert "advisor_revision" in event_types

    @pytest.mark.asyncio
    async def test_advisor_turns_0_is_noop_even_in_parallel(self):
        """``advisor.turns: 0`` skips the advisor regardless of
        ``advisor.parallel``.
        """
        adapter = FakeAdapter([_response("initial")])
        route = _build_route(
            reflection_turns=0,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=0,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # Only the initial call.
        assert len(adapter.calls) == 1
        # No advisor events.
        assert all(not e.type.startswith("advisor") for e in ctx.events)


# ────────────────────────────────────────────────────────────────────
# 3. reflection.parallel: false → sequential reference behaviour
# ────────────────────────────────────────────────────────────────────


class TestReflectionSequentialByDefault:
    """The M2/M3 reference: ``reflection.parallel: false`` (the
    default) still runs the reflection loop sequentially. This
    test guards against accidental regressions in the M4 parallel
    branch.
    """

    @pytest.mark.asyncio
    async def test_parallel_false_unchanged_sequential(self):
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=False,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 5 LLM calls, 5 events.
        assert len(adapter.calls) == 5
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
        ]
        # Final content matches rev1.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev1"


# ────────────────────────────────────────────────────────────────────
# 4. Parallel path uses asyncio.gather; no missing events
# ────────────────────────────────────────────────────────────────────


class TestParallelUsesAsyncioGather:
    """The M4 parallel orchestrator path uses ``asyncio.gather`` to
    dispatch concurrent work. The path is observable through
    (a) the same set of events as sequential, (b) the same call
    count, (c) the same header values.

    We don't make a strict timing assertion; we just verify the
    structural invariants the validation contract pins.
    """

    @pytest.mark.asyncio
    async def test_parallel_no_missing_events(self):
        """All reflection events present in the parallel path."""
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 2 critique events, 2 revised events, 1 initial.
        critiques = [e for e in ctx.events if e.type == "reflect_critique"]
        revised = [e for e in ctx.events if e.type == "reflect_revised"]
        assert len(critiques) == 2
        assert len(revised) == 2
        # No early-exit event.
        assert not any(e.type == "reflect_early_exit" for e in ctx.events)

    @pytest.mark.asyncio
    async def test_parallel_no_content_corruption(self):
        """The final response content is exactly the last revised text
        in the parallel path; no interleaving of partial revisions.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("PARALLEL_REV_0"),
                _critique_response("c1", confidence=0.5),
                _response("PARALLEL_REV_1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # No content corruption: the final text is exactly the last
        # revision's text.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "PARALLEL_REV_1"
        # The per-revision events also carry the correct text.
        revised_texts = [
            e.text for e in ctx.events if e.type == "reflect_revised"
        ]
        assert revised_texts == ["PARALLEL_REV_0", "PARALLEL_REV_1"]

    @pytest.mark.asyncio
    async def test_parallel_usage_summation(self):
        """The parallel path's usage accumulation matches the
        sequential path (sum across all LLM calls).
        """
        adapter = FakeAdapter(
            [
                _response("initial", prompt_tokens=10, completion_tokens=5),
                _critique_response(
                    "c0", confidence=0.5, prompt_tokens=20, completion_tokens=8
                ),
                _response("rev0", prompt_tokens=20, completion_tokens=8),
                _critique_response(
                    "c1", confidence=0.5, prompt_tokens=20, completion_tokens=8
                ),
                _response("rev1", prompt_tokens=20, completion_tokens=8),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        snap = ctx.usage.snapshot()
        # 10 + 20*4 prompt; 5 + 8*4 completion.
        assert snap.prompt_tokens == 10 + 20 * 4
        assert snap.completion_tokens == 5 + 8 * 4
        assert snap.total_tokens == snap.prompt_tokens + snap.completion_tokens

    @pytest.mark.asyncio
    async def test_parallel_headers_equivalent_to_sequential(self):
        """The x-moaxy-* headers in the parallel path match the
        sequential path.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "2"
        assert float(headers["x-moaxy-reflect-confidence"]) == 0.5
        assert headers["x-moaxy-fallbacks-used"] == "0"
        # The alias-resolved header is present.
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        # The request-id header is present.
        assert headers["x-moaxy-request-id"] == ctx.request_id

    @pytest.mark.asyncio
    async def test_parallel_request_messages_not_mutated(self):
        """The parallel path does not mutate the request body."""
        original = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        snapshot = json.dumps(original)
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _critique_response("c1", confidence=0.5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route, request_messages=list(original))
        await Orchestrator(adapter).run(ctx)
        assert json.dumps(ctx.request["messages"]) == snapshot

    @pytest.mark.asyncio
    async def test_parallel_with_fallbacks(self):
        """The parallel path composes with the per-model fallback chain."""
        adapter = FakeAdapter(
            [
                UpstreamError("primary failed", status_code=500, body="err"),
                _response("initial-fallback", model="minimax-m2.7:cloud"),
                _critique_response("c0", confidence=0.5, model="minimax-m2.7:cloud"),
                _response("rev0", model="minimax-m2.7:cloud"),
                _critique_response("c1", confidence=0.5, model="minimax-m2.7:cloud"),
                _response("rev1", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # The walker fell back to the route's fallback on the initial call.
        # Subsequent calls go to the resolved model; we don't re-walk the
        # chain on every call (the chain is per-step).
        assert ctx.upstream_response is not None
        # The final content is the last revision.
        assert ctx.upstream_response.message.content == "rev1"


# ────────────────────────────────────────────────────────────────────
# 5. advisor.parallel without reflection.parallel falls back to sequential
# ────────────────────────────────────────────────────────────────────


class TestAdvisorParallelWithoutReflectionParallel:
    """When ``advisor.parallel: true`` is set but ``reflection.parallel``
    is ``false``, the parallel path is gated off. The advisor still
    runs after the (sequential) reflection loop, content-equivalent
    to the sequential path.
    """

    @pytest.mark.asyncio
    async def test_advisor_parallel_with_sequential_reflection(self):
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c0", confidence=0.5),
                _response("rev0"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            reflection_parallel=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 4 LLM calls.
        assert len(adapter.calls) == 4
        # The final answer is the post-reflection revision.
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "rev0"
        # Advisor events emitted.
        event_types = [e.type for e in ctx.events]
        assert "advisor" in event_types
        assert "advisor_approve" in event_types


# ────────────────────────────────────────────────────────────────────
# 6. M5 weighted early-exit in the PARALLEL reflection path
# ────────────────────────────────────────────────────────────────────


class TestParallelWeightedEarlyExit:
    """M5: the PARALLEL reflection path uses ``parse_weighted_signal``.

    Mirrors the M5 sequential weighted-early-exit contract pins in
    :mod:`tests.test_reflection` (``TestWeightedEarlyExit``) but
    for the parallel path. The M5 change set extends the parallel
    path with the same weighted-signal computation, the same
    ``reflect_score`` event emission, the same DELTA 7 safety rule,
    and the same context-attribute updates the sequential path
    performs.
    """

    @pytest.mark.asyncio
    async def test_parallel_uses_weighted_signal(self):
        """VAL-PIPE-EXTRA-010 (parallel): trust_verbal=1.0, trust_score=0.0
        reduces to the v1 ``confidence >= threshold`` check in the
        parallel path.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c", confidence=0.9, score=5),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            trust_verbal=1.0,
            trust_score=0.0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 0.9 >= 0.85 → early-exit; 2 LLM calls.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_score",
            "reflect_early_exit",
        ]
        assert len(adapter.calls) == 2
        # The combined signal equals the raw confidence.
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.9)
        assert ctx.__dict__["last_score"] == 5

    @pytest.mark.asyncio
    async def test_parallel_score_only_threshold(self):
        """VAL-PIPE-EXTRA-011 (parallel): trust_verbal=0.0, trust_score=1.0
        makes the early-exit check driven by ``score / 10`` only.
        """
        # REFLECT_CONFIDENCE: 0.5, SCORE: 9. combined = 0.0 * 0.5 + 1.0 * 0.9 = 0.9
        # 0.9 >= 0.85 → early-exit.
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c", confidence=0.5, score=9),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            trust_verbal=0.0,
            trust_score=1.0,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        event_types = [e.type for e in ctx.events]
        assert "reflect_early_exit" in event_types
        assert len(adapter.calls) == 2
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.9)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_parallel_combined_weights(self):
        """VAL-PIPE-EXTRA-012 (parallel): trust_verbal=0.5, trust_score=0.5
        combines confidence and score; revision runs when the
        combined signal is below the threshold.
        """
        # REFLECT_CONFIDENCE: 0.6, SCORE: 9. combined = 0.5*0.6 + 0.5*0.9 = 0.75
        # 0.75 < 0.85 → revision runs.
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c", confidence=0.6, score=9),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            trust_verbal=0.5,
            trust_score=0.5,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.75)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_parallel_reflect_score_event_emitted(self):
        """VAL-PIPE-EXTRA-018 (parallel): a parsed SCORE: line emits
        a ``reflect_score`` event in the parallel path with the
        integer score as a string.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c", confidence=0.5, score=8),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        score_events = [e for e in ctx.events if e.type == "reflect_score"]
        assert len(score_events) == 1
        assert score_events[0].text == "8"
        assert score_events[0].turn == 0
        assert ctx.__dict__["last_score"] == 8

    @pytest.mark.asyncio
    async def test_parallel_reflect_score_event_not_emitted_when_missing(self):
        """When the model omits the SCORE: line, no ``reflect_score``
        event is emitted and ``last_score`` is None.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.5),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        assert "reflect_score" not in [e.type for e in ctx.events]
        assert ctx.__dict__["last_score"] is None

    @pytest.mark.asyncio
    async def test_parallel_delta7_safety_malformed_critique_continues(self):
        """VAL-PIPE-EXTRA-016 (parallel): DELTA 7 safety rule applies
        in the parallel path. A critique with NO
        ``REFLECT_CONFIDENCE:`` line is treated as a malformed
        response; the revision runs as if ``early_exit: false`` for
        that turn.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("a critique with no confidence line", confidence=None),
                _response("revised after missing line"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.0,  # extreme threshold; the safety rule still applies
            reflection_parallel=True,
        )
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
    async def test_parallel_explicit_zero_confidence_not_malformed(self):
        """VAL-PIPE-EXTRA-017 (parallel): ``REFLECT_CONFIDENCE: 0.0``
        is NOT malformed in the parallel path. The v1 threshold
        check applies: 0.0 < 0.85 → revision runs (not the safety
        path).
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_response("c", confidence=0.0),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.0 < 0.85 → revision runs.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_parallel_combined_signal_exposed_on_context(self):
        """The combined weighted-signal result is stored on
        ``ctx.__dict__['last_combined_signal']`` in the parallel
        path.
        """
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
            reflection_parallel=True,
            trust_verbal=0.6,
            trust_score=0.4,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 0.6 * 0.6 + 0.4 * 0.9 = 0.72 → revision runs.
        assert len(adapter.calls) == 3
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.72)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_parallel_score_fallback_when_score_missing(self):
        """When the SCORE: line is missing, the combined signal falls
        back to the raw ``confidence`` (v1 invariant), regardless of
        the trust weights.
        """
        # REFLECT_CONFIDENCE: 0.9, no SCORE. With trust_verbal=0.6,
        # trust_score=0.4 the combined should equal 0.9 (NOT
        # 0.6 * 0.9 = 0.54). The "score missing" path falls back to
        # the raw confidence verbatim.
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c", confidence=0.9, score=None),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            trust_verbal=0.6,
            trust_score=0.4,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 0.9 >= 0.85 → early-exit.
        assert "reflect_early_exit" in [e.type for e in ctx.events]
        # The combined signal equals the raw confidence (not 0.54).
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.9)
        # The score is None.
        assert ctx.__dict__["last_score"] is None

    @pytest.mark.asyncio
    async def test_parallel_event_ordering_with_score(self):
        """When a critique carries both REFLECT_CONFIDENCE: and SCORE:,
        the ``reflect_score`` event is appended between
        ``reflect_critique`` and ``reflect_revised``.
        """
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c0", confidence=0.5, score=5),
                _response("rev0"),
                _critique_with_score_response("c1", confidence=0.5, score=5),
                _response("rev1"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # Event ordering: initial, (critique, score, revised) per turn.
        event_types = [e.type for e in ctx.events]
        assert event_types == [
            "initial",
            "reflect_critique",
            "reflect_score",
            "reflect_revised",
            "reflect_critique",
            "reflect_score",
            "reflect_revised",
        ]
        score_events = [e for e in ctx.events if e.type == "reflect_score"]
        assert len(score_events) == 2
        assert [e.turn for e in score_events] == [0, 1]
        assert [e.text for e in score_events] == ["5", "5"]
        # The last_score reflects the LAST turn's parsed score.
        assert ctx.__dict__["last_score"] == 5
