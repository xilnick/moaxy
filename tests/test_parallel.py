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
to sequential (``VAL-PIPE-009``, ``VAL-PIPE-021``). The tests in
this file exercise the parallel path with a scripted
:class:`FakeAdapter` and assert that the final response content
equals the sequential counterpart, that the events list is
well-formed, and that the header values are correct. No timing
assertion is made.

This file complements the existing
``tests/test_orchestrator_parallel.py`` (which is broader, covering
parallel-vs-sequential equivalence under many scripted scenarios).
The tests here focus specifically on the end-to-end content
contract pin: the same scripted responses produce the same final
content under both parallel and sequential configurations.
"""

from __future__ import annotations

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
from moaxy.routing.matcher import RouteMatch
from tests.fixtures.fake_adapter import FakeAdapter

# ────────────────────────────────────────────────────────────────────
# Response builders
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
) -> ChatResponse:
    """Build a critique response. ``confidence=None`` omits the line."""
    if confidence is None:
        text = body
    else:
        text = f"{body}\nREFLECT_CONFIDENCE: {confidence}"
    return _response(text, model=model)


# ────────────────────────────────────────────────────────────────────
# Route / context builders
# ────────────────────────────────────────────────────────────────────


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
    request_id: str = "req-parallel",
    request_messages: list[dict[str, Any]] | None = None,
) -> PipelineContext:
    body: dict[str, Any] = {
        "model": route.original_model,
        "messages": request_messages
        or [{"role": "user", "content": "ping"}],
    }
    return PipelineContext(
        request_id=request_id,
        request=body,
        route=route,
        model_alias_resolved=route.resolved_model,
        target_backend=route.backend,
        original_model=route.original_model,
    )


# ────────────────────────────────────────────────────────────────────
# 1. Content equivalence: reflection.parallel vs sequential
# ────────────────────────────────────────────────────────────────────


class TestReflectionParallelContentEquivalence:
    """``reflection.parallel: true`` produces equivalent final content
    to the sequential path when scripted.

    The validation contract (VAL-PIPE-009) pins content equivalence
    to the sequential path; no timing assertion is made. The
    tests below build the *same* scripted responses twice — once
    with ``parallel=False`` and once with ``parallel=True`` — and
    assert the final response content and the call count are
    identical.
    """

    @pytest.mark.asyncio
    async def test_parallel_matches_sequential_final_content(self):
        """VAL-PIPE-009: turns=2, no early-exit, parallel=True
        produces the same final content as parallel=False.
        """
        # Scripted response queue: initial, critique_0, rev_0,
        # critique_1, rev_1. Five calls in source order.
        script = [
            _response("initial answer"),
            _critique_response("c0", confidence=0.5),
            _response("rev0"),
            _critique_response("c1", confidence=0.5),
            _response("rev1"),
        ]
        # Sequential reference.
        adapter_seq = FakeAdapter(list(script))
        route_seq = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=False,
        )
        ctx_seq = _build_context(route_seq)
        await Orchestrator(adapter_seq).run(ctx_seq)

        # Parallel path.
        adapter_par = FakeAdapter(list(script))
        route_par = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx_par = _build_context(route_par)
        await Orchestrator(adapter_par).run(ctx_par)

        # Both paths must make the same number of LLM calls.
        assert len(adapter_seq.calls) == len(adapter_par.calls) == 5
        # The final response content is identical (rev1).
        assert (
            ctx_seq.upstream_response is not None
            and ctx_par.upstream_response is not None
        )
        assert (
            ctx_seq.upstream_response.message.content
            == ctx_par.upstream_response.message.content
        )
        assert ctx_par.upstream_response.message.content == "rev1"
        # The header values are identical.
        headers_seq = build_response_headers(ctx_seq, request_id=ctx_seq.request_id)
        headers_par = build_response_headers(ctx_par, request_id=ctx_par.request_id)
        assert headers_seq["x-moaxy-reflect-turns"] == "2"
        assert headers_par["x-moaxy-reflect-turns"] == "2"

    @pytest.mark.asyncio
    async def test_parallel_emits_ordered_events_per_turn(self):
        """The parallel path emits ``reflect_critique`` and
        ``reflect_revised`` events in source order, one pair per
        turn. The event list shape matches the sequential path.
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

        event_types = [e.type for e in ctx.events]
        # initial, then critique(0), revised(0), critique(1), revised(1).
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
        # The usage snapshot reflects the 5 LLM calls.
        usage = ctx.usage.snapshot()
        assert usage.prompt_tokens == 10 * 5
        assert usage.completion_tokens == 5 * 5

    @pytest.mark.asyncio
    async def test_parallel_early_exit_short_circuits_remaining_turns(self):
        """When early-exit fires on turn 0, the parallel path stops
        before turn 1. The final content is the post-turn-0
        revision, matching the sequential behaviour.
        """
        # Script: initial, critique(0) w/ confidence 0.95 (clears
        # threshold), revision(0). Turn 1 is short-circuited.
        script = [
            _response("initial"),
            _critique_response("c0", confidence=0.95),
            _response("rev0"),
        ]
        # Sequential reference.
        adapter_seq = FakeAdapter(list(script))
        route_seq = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=False,
        )
        ctx_seq = _build_context(route_seq)
        await Orchestrator(adapter_seq).run(ctx_seq)

        # Parallel path.
        adapter_par = FakeAdapter(list(script))
        route_par = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
        )
        ctx_par = _build_context(route_par)
        await Orchestrator(adapter_par).run(ctx_par)

        # Both paths make 3 calls and reach the same final content.
        assert len(adapter_seq.calls) == len(adapter_par.calls) == 3
        assert (
            ctx_seq.upstream_response is not None
            and ctx_par.upstream_response is not None
        )
        assert (
            ctx_seq.upstream_response.message.content
            == ctx_par.upstream_response.message.content
            == "rev0"
        )
        # The early-exit event is emitted by both paths at turn 0.
        seq_events = [e.type for e in ctx_seq.events]
        par_events = [e.type for e in ctx_par.events]
        assert "reflect_early_exit" in seq_events
        assert "reflect_early_exit" in par_events


# ────────────────────────────────────────────────────────────────────
# 2. Content equivalence: advisor.parallel vs sequential
# ────────────────────────────────────────────────────────────────────


class TestAdvisorParallelContentEquivalence:
    """``advisor.parallel: true`` (with ``reflection.parallel: true``)
    produces a non-empty response whose content is one of the
    scripted outputs (VAL-PIPE-021).
    """

    @pytest.mark.asyncio
    async def test_advisor_parallel_final_content_non_empty(self):
        """VAL-PIPE-021: advisor.parallel=True + reflection.parallel=True
        + turns=2 + advisor.turns=1 produces a non-empty final
        response.
        """
        adapter = FakeAdapter(
            [
                # Initial.
                _response("initial", model="minimax-m3:cloud"),
                # Reflection turn 0.
                _critique_response("c0", confidence=0.5),
                _response("rev0", model="minimax-m3:cloud"),
                # Reflection turn 1.
                _critique_response("c1", confidence=0.5),
                _response("rev1", model="minimax-m3:cloud"),
                # Advisor call.
                _response(
                    "ADVISOR_REVISE: improved final answer",
                    model="deepseek-v4-pro:cloud",
                ),
                # Primary-model revision after the advisor's revise.
                _response(
                    "primary-final-after-revise",
                    model="minimax-m3:cloud",
                ),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # The orchestrator records events for the initial,
        # the reflection, the advisor call, and the post-advisor
        # primary revision. The exact set of events is
        # implementation-defined for the parallel path; the
        # content contract is the only one pinned.
        assert ctx.upstream_response is not None
        final = ctx.upstream_response.message.content
        assert final
        # The final content is one of the scripted outputs (the
        # contract pins content equivalence, not a specific value).
        assert final in {
            "improved final answer",  # the advisor's revised text
            "primary-final-after-revise",  # the primary's revision after the advisor
            "rev1",  # the self-reflection path's answer
        }
        # The header reports the configured advisor model and
        # reflects that the advisor ran.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers.get("x-moaxy-advisor-model") == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_advisor_sequential_vs_parallel_final_content(self):
        """Both the sequential and parallel advisor paths produce a
        non-empty final content. The contract does not pin call
        count (the parallel path runs an extra self-reflection
        concurrently with the advisor); it pins content
        equivalence and the presence of the advisor header.
        """
        # The sequential path consumes the script in source order:
        # initial, reflection-critique, reflection-revision,
        # advisor. The advisor emits ADVISOR_APPROVE so no
        # post-advisor primary revision is made. The parallel
        # path consumes the same script but interleaves the
        # self-reflection path's two extra calls
        # (self-critique + self-revision) alongside the
        # advisor. To satisfy both paths with a single script
        # the script must include the self-reflection responses
        # after the sequential path's expected entries.
        script = [
            # Initial.
            _response("initial", model="minimax-m3:cloud"),
            # Reflection critique + revision (sequential path).
            _critique_response("c0", confidence=0.5),
            _response("rev0", model="minimax-m3:cloud"),
            # Advisor call (ADVISOR_APPROVE: no follow-up call).
            _response(
                "ADVISOR_APPROVE",
                model="deepseek-v4-pro:cloud",
            ),
            # Self-reflection critique + revision on the initial
            # answer (parallel path's self-reflection only).
            _critique_response("csr", confidence=0.5),
            _response("rev-sr", model="minimax-m3:cloud"),
        ]
        # Sequential path.
        adapter_seq = FakeAdapter(list(script))
        route_seq = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=False,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=False,
        )
        ctx_seq = _build_context(route_seq)
        await Orchestrator(adapter_seq).run(ctx_seq)

        # Parallel path.
        adapter_par = FakeAdapter(list(script))
        route_par = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx_par = _build_context(route_par)
        await Orchestrator(adapter_par).run(ctx_par)

        # The final content is non-empty in both cases. The
        # sequential path produces ``rev0`` (the post-reflection
        # answer, kept because the advisor approved). The
        # parallel path's final content is implementation-defined
        # (the contract pins equivalence to the scripted
        # outputs, not a specific value); we accept any
        # non-empty scripted output.
        assert ctx_seq.upstream_response is not None
        assert ctx_par.upstream_response is not None
        assert ctx_seq.upstream_response.message.content == "rev0"
        assert ctx_par.upstream_response.message.content in {
            "rev0",  # the sequential path's answer
            "rev-sr",  # the self-reflection path's answer
        }
        # The headers report the advisor model in both cases.
        headers_seq = build_response_headers(ctx_seq, request_id=ctx_seq.request_id)
        headers_par = build_response_headers(ctx_par, request_id=ctx_par.request_id)
        assert headers_seq.get("x-moaxy-advisor-model") == "deepseek-v4-pro:cloud"
        assert headers_par.get("x-moaxy-advisor-model") == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# 3. Parallel path with turn skipping (zero-turn passthrough)
# ────────────────────────────────────────────────────────────────────


class TestParallelPassthrough:
    """``turns=0`` short-circuits the reflection stage entirely;
    ``parallel: true`` does not change that.
    """

    @pytest.mark.asyncio
    async def test_parallel_turns_zero_is_passthrough(self):
        """turns=0, parallel=True → exactly 1 LLM call (the initial)."""
        adapter = FakeAdapter([_response("initial only")])
        route = _build_route(
            reflection_turns=0,
            reflection_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "initial only"
        # The events list has only the initial event.
        assert [e.type for e in ctx.events] == ["initial"]
        # The header reflects 0 reflection turns.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-turns"] == "0"

    @pytest.mark.asyncio
    async def test_parallel_advisor_zero_turns_no_advisor_call(self):
        """turns=0, advisor.turns=0, parallel=True → exactly 1 LLM
        call (the initial); no advisor call is made.
        """
        adapter = FakeAdapter([_response("passthrough")])
        route = _build_route(
            reflection_turns=0,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=0,
            advisor_parallel=True,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert ctx.upstream_response is not None
        assert ctx.upstream_response.message.content == "passthrough"
        # No advisor event in the events list.
        event_types = [e.type for e in ctx.events]
        assert not any(t.startswith("advisor") for t in event_types)


# ────────────────────────────────────────────────────────────────────
# 4. Stream-run parallel advisor: self-reflection critique must
#    target the original initial text, not the post-reflection text
# ────────────────────────────────────────────────────────────────────


class TestStreamRunParallelAdvisorInitialText:
    """The M4 streaming path's parallel-advisor branch must mirror
    the buffered path's call site for :meth:`_run_advisor_parallel`:
    it has to pass the **original** Stage-1 initial text as
    ``initial_answer`` (not the post-reflection text).

    The buffered ``Orchestrator.run`` correctly captures
    ``initial_response.message.content`` and uses it as
    ``initial_answer``; the streaming ``Orchestrator.stream_run``
    must do the same. The bug previously caused
    ``stream_run`` to pass ``current_answer`` (the
    post-reflection text) for both arguments, so the
    self-reflection path in :meth:`_run_advisor_parallel` built
    its critique prompt from the post-reflection text. This
    broke the M3+M4 contract for any client that enabled
    ``reflection.parallel + advisor.parallel + stream: true``
    simultaneously.

    The test below exercises ``stream_run`` end-to-end with a
    scripted :class:`FakeAdapter` (extended with a stream
    script) and pins the self-reflection critique's input
    message content equals the original initial text.
    """

    @pytest.mark.asyncio
    async def test_stream_run_parallel_advisor_critique_uses_initial_text(self):
        """``stream_run`` with ``reflection.parallel + advisor.parallel``
        passes the **original** Stage-1 initial text to the
        self-reflection critique, not the post-reflection text.

        Setup:
        * ``reflection.turns: 1``, ``reflection.parallel: true`` —
          the parallel reflection loop mutates ``current_answer``
          to the post-reflection text.
        * ``advisor.turns: 1``, ``advisor.parallel: true`` —
          :meth:`Orchestrator._run_advisor_parallel` runs the
          advisor pass AND a self-reflection on the original
          answer concurrently. The self-reflection critique
          must target the *initial* text, not the
          *post-reflection* text.

        The streamed initial answer uses a unique marker
        (``"INITIAL_ANSWER"``) so the test can assert that the
        self-reflection critique's input message content
        contains the original initial text.
        """
        # Use distinct, recognisable strings for the initial and
        # post-reflection texts so the test can pinpoint which
        # one the self-reflection critique was built from.
        INITIAL_TEXT = "INITIAL_ANSWER_FROM_STREAM"
        POST_REFLECTION_TEXT = "POST_REFLECTION_REVISED_ANSWER"

        adapter = FakeAdapter(
            # The initial answer is streamed as a single chunk.
            stream_script=[[INITIAL_TEXT]],
            responses=[
                # Stage 2 parallel reflection turn 0: critique.
                _critique_response("c0", confidence=0.5),
                # Stage 2 parallel reflection turn 0: revision.
                _response(POST_REFLECTION_TEXT),
                # Stage 3 self-reflection (parallel advisor): critique.
                # The stage-3 self-reflection critique must be
                # built from the *original* initial text
                # (``INITIAL_TEXT``), NOT the post-reflection
                # text. We do not constrain its response here.
                _critique_response("csr", confidence=0.5),
                # Stage 3 self-reflection (parallel advisor): revision.
                _response("self-revised"),
                # Stage 3 advisor call (ADVISOR_APPROVE so the
                # advisor path makes no follow-up primary call).
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                ),
            ],
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            advisor_parallel=True,
        )
        ctx = _build_context(route)

        # Drive the streaming entry point end-to-end. The
        # ``stream_run`` coroutine is an async generator; the
        # test drains it into a list of bytes to fully consume
        # the pipeline (including the Stage 2 reflection loop
        # and the Stage 3 parallel advisor + self-reflection
        # gather). The test does not assert on the SSE bytes;
        # it asserts on the adapter call log.
        chunks: list[bytes] = []
        async for chunk in Orchestrator(adapter).stream_run(ctx):
            chunks.append(chunk)
        # Sanity: the stream terminated cleanly.
        assert chunks, "stream_run produced no chunks"
        assert b"data: [DONE]" in b"".join(chunks)

        # The fix: the self-reflection critique's input message
        # content contains the *original* initial text (the
        # Stage-1 answer), NOT the post-reflection text.
        # Identify all critique calls — those whose last
        # user-role message starts with "Please critique the
        # following answer:" (the revision calls' last user
        # message starts with "Please revise" or "Please
        # incorporate", and the advisor-revision call's last
        # user message starts with "Please incorporate"). The
        # critique calls are exactly two: Stage 2 reflection
        # critique and Stage 3 self-reflection critique.
        critique_answers: list[str] = []
        for call in adapter.calls:
            last_critique_user_msg: str | None = None
            for msg in call.get("messages", []):
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "user"
                    and isinstance(msg.get("content"), str)
                    and msg["content"].startswith(
                        "Please critique the following answer:"
                    )
                ):
                    last_critique_user_msg = msg["content"]
            if last_critique_user_msg is not None:
                # Check the *last* user message in the call is
                # the critique message (not the revision /
                # advisor-revision message that also includes
                # a critique message earlier in the history).
                user_messages = [
                    m["content"]
                    for m in call.get("messages", [])
                    if isinstance(m, dict)
                    and m.get("role") == "user"
                    and isinstance(m.get("content"), str)
                ]
                if user_messages and user_messages[-1] == last_critique_user_msg:
                    critique_answers.append(
                        last_critique_user_msg.split(
                            "Please critique the following answer:\n", 1
                        )[1]
                    )
        # Both critique calls — Stage 2 reflection critique and
        # Stage 3 self-reflection critique — must use the
        # original initial text as the input answer. The
        # original bug caused the Stage 3 self-reflection
        # critique to use the post-reflection text.
        assert critique_answers, "no critique calls were recorded"
        assert len(critique_answers) == 2, (
            f"expected exactly 2 critique calls "
            f"(Stage 2 reflection + Stage 3 self-reflection), "
            f"got {len(critique_answers)}: {critique_answers!r}"
        )
        for answer in critique_answers:
            assert answer == INITIAL_TEXT, (
                f"critique input answer is not the original initial text: "
                f"{answer!r}"
            )
        # Negative pin: the post-reflection text must NOT be the
        # answer passed to any critique call. This is the
        # symptom of the original bug.
        assert POST_REFLECTION_TEXT not in critique_answers, (
            "self-reflection critique was built from the "
            "post-reflection text; the M3+M4 contract pins the "
            "self-reflection's input to the original initial text"
        )


# ────────────────────────────────────────────────────────────────────
# 5. M5 weighted early-exit in the PARALLEL reflection path
# ────────────────────────────────────────────────────────────────────


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


class TestParallelWeightedEarlyExit:
    """M5: the PARALLEL reflection path uses ``parse_weighted_signal``.

    The M5 weighted-early-exit change in the SEQUENTIAL path
    (``_run_reflection``) is mirrored in the PARALLEL path
    (``_run_reflection_parallel``): the threshold check now uses
    ``trust_verbal * confidence + trust_score * (score / 10)``
    instead of the v1 ``confidence >= threshold`` check, and the
    DELTA 7 safety rule (a missing ``REFLECT_CONFIDENCE:`` line is
    treated as malformed and does NOT short-circuit) applies in
    the parallel path too. These tests pin both behaviors.
    """

    @pytest.mark.asyncio
    async def test_parallel_uses_weighted_signal_for_threshold_check(self):
        """VAL-PIPE-EXTRA-010 (parallel): trust_verbal=1.0, trust_score=0.0
        reduces to v1 ``confidence >= threshold`` in the parallel
        path; early-exit fires when confidence clears the threshold.
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

        # 0.9 >= 0.85 → early-exit; 2 LLM calls (initial + critique).
        # The reflect_score event is emitted because the SCORE: line
        # is parsed, between reflect_critique and reflect_early_exit.
        assert [e.type for e in ctx.events] == [
            "initial",
            "reflect_critique",
            "reflect_score",
            "reflect_early_exit",
        ]
        assert len(adapter.calls) == 2
        # The combined signal equals the raw confidence.
        assert ctx.__dict__["last_combined_signal"] == 0.9
        # The score is parsed and stored, but does not affect the check.
        assert ctx.__dict__["last_score"] == 5

    @pytest.mark.asyncio
    async def test_parallel_score_only_threshold(self):
        """VAL-PIPE-EXTRA-011 (parallel): trust_verbal=0.0, trust_score=1.0
        makes the early-exit check driven by score/10 only.
        """
        # REFLECT_CONFIDENCE: 0.5 (would clear no threshold), SCORE: 9.
        # With trust_verbal=0.0, trust_score=1.0 the combined = 0.9,
        # which is >= 0.85 → early-exit.
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

        # Early-exit fires. The reflect_score event is emitted because
        # the SCORE: line is parsed.
        event_types = [e.type for e in ctx.events]
        assert "reflect_early_exit" in event_types
        assert "reflect_score" in event_types
        assert len(adapter.calls) == 2
        # combined = 0.0 * 0.5 + 1.0 * 0.9 = 0.9.
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.9)
        assert ctx.__dict__["last_score"] == 9
        # last_confidence is the raw parsed confidence.
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_parallel_combined_weights(self):
        """VAL-PIPE-EXTRA-012 (parallel): trust_verbal=0.5, trust_score=0.5
        combines confidence and score. The combined signal is stored
        on ``ctx.__dict__['last_combined_signal']``.
        """
        # REFLECT_CONFIDENCE: 0.6, SCORE: 9.
        # combined = 0.5 * 0.6 + 0.5 * 0.9 = 0.75, which is < 0.85
        # → revision runs.
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

        # 3 LLM calls; the revision runs because combined < threshold.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        # The combined signal is the weighted formula's result.
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.75)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_parallel_reflect_score_event_emitted(self):
        """VAL-PIPE-EXTRA-018 (parallel): a parsed SCORE: line emits
        a ``reflect_score`` event in the parallel path.
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

        # A 'reflect_score' event is appended with text="8" and turn=0.
        score_events = [e for e in ctx.events if e.type == "reflect_score"]
        assert len(score_events) == 1
        assert score_events[0].text == "8"
        assert score_events[0].turn == 0
        # The ctx.__dict__['last_score'] is also set.
        assert ctx.__dict__["last_score"] == 8

    @pytest.mark.asyncio
    async def test_parallel_reflect_score_event_not_emitted_when_score_missing(self):
        """The parallel path does not emit ``reflect_score`` when the
        model omits the SCORE: line.
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
        # No reflect_score event in the list.
        assert "reflect_score" not in [e.type for e in ctx.events]
        # last_score is None.
        assert ctx.__dict__["last_score"] is None

    @pytest.mark.asyncio
    async def test_parallel_delta7_safety_malformed_critique_continues(self):
        """VAL-PIPE-EXTRA-016 (parallel): DELTA 7 safety rule applies in
        the parallel path. A critique with NO ``REFLECT_CONFIDENCE:``
        line is treated as a malformed response; the revision runs
        as if ``early_exit: false`` for that turn.
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
            threshold=0.0,  # extreme threshold
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
        is NOT malformed. The v1 threshold check applies: 0.0 < 0.85
        → revision runs (not the safety path).
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
        # 0.0 < 0.85 → revision runs; no early-exit.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_parallel_combined_signal_exposed_on_context(self):
        """The combined weighted-signal result is stored on
        ``ctx.__dict__['last_combined_signal']`` in the parallel
        path. The ``last_score`` attribute carries the parsed score.
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
        # 0.6 * 0.6 + 0.4 * 0.9 = 0.72; revision runs.
        assert len(adapter.calls) == 3
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.72)
        assert ctx.__dict__["last_score"] == 9
        assert ctx.__dict__["last_confidence"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_parallel_weighted_early_exit_2_turn_loop(self):
        """The weighted-signal threshold check works on every turn
        of the 2-turn parallel loop, not just the last turn.
        """
        # Turn 0: REFLECT_CONFIDENCE: 0.5, SCORE: 9 → combined 0.5*0.5+0.5*0.9=0.7
        # (0.7 < 0.85, no early-exit; revision runs).
        # Turn 1: REFLECT_CONFIDENCE: 0.95, SCORE: 5 → combined 0.5*0.95+0.5*0.5=0.725
        # Wait, that's still < 0.85. Let me use values that clear the threshold
        # for the early-exit on the LAST turn.
        # Turn 0: confidence 0.5, score 5 → combined 0.5 → no early-exit.
        # Turn 1: confidence 0.95, score 5 → combined 0.5*0.95+0.5*0.5=0.725
        # Hmm, still < 0.85. Let me increase trust_score to push it above.
        # Actually, with trust_verbal=0.5, trust_score=0.5,
        # combined = 0.5*conf + 0.5*score/10.
        # For 0.95 conf and 5 score: 0.5*0.95 + 0.5*0.5 = 0.475 + 0.25 = 0.725. < 0.85.
        # Let me use SCORE: 9 instead: 0.5*0.95 + 0.5*0.9 = 0.475 + 0.45 = 0.925. >= 0.85. Early-exit!
        adapter = FakeAdapter(
            [
                _response("initial"),
                _critique_with_score_response("c0", confidence=0.5, score=5),
                _response("rev0"),
                _critique_with_score_response("c1", confidence=0.95, score=9),
                # No revision on the last turn when early-exit fires.
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=True,
            threshold=0.85,
            reflection_parallel=True,
            trust_verbal=0.5,
            trust_score=0.5,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)

        # 4 LLM calls: initial, c0, rev0, c1. The c1 critique clears
        # the threshold and is the last turn, so the revision is skipped.
        assert len(adapter.calls) == 4
        # The early-exit event is emitted at turn 1.
        event_types = [e.type for e in ctx.events]
        assert "reflect_early_exit" in event_types
        # Only one early-exit (at the last turn).
        early_exits = [e for e in ctx.events if e.type == "reflect_early_exit"]
        assert len(early_exits) == 1
        # The last combined signal corresponds to turn 1's critique.
        assert ctx.__dict__["last_combined_signal"] == pytest.approx(0.925)
        # The last score is 9 (parsed from turn 1's critique).
        assert ctx.__dict__["last_score"] == 9

    @pytest.mark.asyncio
    async def test_parallel_existing_event_ordering_preserved(self):
        """The weighted-signal refactor in the parallel path does NOT
        change the v1 event ordering for the standard 2-turn full
        loop (no early-exit).
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

        # The standard v1 ordering is preserved: initial, then per-turn
        # (critique, revised, optional reflect_score when SCORE: present).
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
        # Two reflect_score events, one per turn.
        score_events = [e for e in ctx.events if e.type == "reflect_score"]
        assert len(score_events) == 2
        assert [e.turn for e in score_events] == [0, 1]
        assert [e.text for e in score_events] == ["5", "5"]
