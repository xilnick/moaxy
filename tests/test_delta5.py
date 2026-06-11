"""M5 Research-Deltas test suite.

This module is the consolidated M5 milestone test file. It
covers all 42 new M5 validation-contract assertions
(``VAL-PIPE-EXTRA-001..042``) with one or more dedicated test
cases. The file is intentionally self-contained: it imports
the existing :class:`FakeAdapter` fixture from
``tests.fixtures.fake_adapter``, the shared helpers from
``tests.conftest``, and the model / route / context factories
declared at the top of this module.

The contract is the definition of "done" for the M5 milestone;
every assertion is a black-box, behavior-based check that the
validators (programmatic scrutiny validator and user-testing
validator) can execute against the running proxy or in-process
via :class:`FakeAdapter`. The tests here are hermetic: no
in-process HTTP, no real Ollama, no on-disk plugins.

Backwards compatibility
-----------------------

The M5 deltas are STRICTLY ADDITIVE. The default
``trust_verbal: 0.6, trust_score: 0.4, order: "reflect_first"``
settings preserve the v1-v4 behavior for any test or
configuration that emits ``REFLECT_CONFIDENCE:`` without
``SCORE:``. The new ``SCORE:`` line is parsed but the v1
``confidence >= threshold`` invariant is preserved via the
"score missing" fallback in :func:`parse_weighted_signal`.

The M5 backwards-compat invariant VAL-PIPE-EXTRA-036 pins
that ``tests/test_plugins.py`` passes unchanged. The dedicated
test in this file runs the file as a subprocess and asserts
the exit code, satisfying the contract's "explicitly run
tests/test_plugins.py" requirement.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    Usage,
)
from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    MoaxyConfig,
    ReflectionConfig,
    RouteConfig,
)
from moaxy.models.config import RouteMatch as ConfigRouteMatch
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.pipeline.prompts import DEFAULT_ADVISOR_PROMPT, DEFAULT_REFLECT_PROMPT
from moaxy.pipeline.reflector import (
    parse_confidence,
    parse_score,
    parse_weighted_signal,
)
from moaxy.routing.matcher import RouteMatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_TEST_FILE = PROJECT_ROOT / "tests" / "test_plugins.py"


# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter — hermetic, in-process
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    Mirrors the helpers in ``test_orchestrator.py`` and
    ``test_reflection.py``. The script is a list of either
    :class:`ChatResponse` (success) or :class:`BaseException`
    (raised). Calls are recorded in :attr:`calls` so tests can
    assert on the ``model`` / ``messages`` / ``**kwargs`` the
    orchestrator forwarded.
    """

    name = "scripted_delta5"

    def __init__(
        self,
        script: list[Any] | None = None,
        stream_script: list[Any] | None = None,
    ) -> None:
        self._script: list[Any] = list(script or [])
        self._stream_script: list[Any] = list(stream_script or [])
        self._index: int = 0
        self._stream_index: int = 0
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

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

    async def stream(  # pragma: no cover - exercised in streaming section
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        self.stream_calls.append(
            {"model": model, "messages": messages, **kwargs}
        )
        if self._stream_index >= len(self._stream_script):
            yield "stream-ok"
            return
        entry = self._stream_script[self._stream_index]
        self._stream_index += 1
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, list):
            for delta in entry:
                yield str(delta)
            return
        yield "stream-ok"

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


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


def _build_route(
    *,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    parallel: bool = False,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    advisor_parallel: bool = False,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    aliases: dict[str, str] | None = None,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    order: str = "reflect_first",
    trust_verbal: float = 0.6,
    trust_score: float = 0.4,
) -> RouteMatch:
    config_route = RouteConfig(
        name="delta5-test-route",
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
            parallel=parallel,
            order=order,
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


def _run(adapter: ScriptedAdapter, ctx: PipelineContext) -> PipelineContext:
    """Drive the async orchestrator synchronously for tests."""
    return asyncio.run(Orchestrator(adapter).run(ctx))


# ────────────────────────────────────────────────────────────────────
# DELTA 1 — Conditional advisor skip (VAL-PIPE-EXTRA-001, 002, 003, 023, 024, 030)
# ────────────────────────────────────────────────────────────────────


class TestDelta1ConditionalAdvisorSkip:
    """VAL-PIPE-EXTRA-001: advisor skipped when parsed confidence >= 0.85.

    The orchestrator MUST skip the advisor LLM call when the
    post-reflection confidence is at or above the hardcoded
    threshold (0.85). The response MUST include the
    ``x-moaxy-advisor-skipped: 1/confidence=<x>`` header and
    MUST NOT include ``x-moaxy-advisor-model``. The events
    list MUST NOT contain any ``advisor*`` event.
    """

    @pytest.mark.asyncio
    async def test_extra_001_advisor_skipped_at_confidence_0_9(self):
        # scripted calls:
        # 0. initial
        # 1. critique with confidence 0.9 (>= 0.85) → short-circuit
        #    on last turn (turns=1). No revision, no advisor.
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.9",
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

        # 2 LLM calls: initial + critique. No revision, no advisor.
        assert len(adapter.calls) == 2
        # No advisor events.
        types = [e.type for e in ctx.events]
        assert "advisor" not in types
        assert "advisor_approve" not in types
        assert "advisor_revised" not in types
        # Headers: skip header is set; advisor-model is NOT set.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "1/confidence=0.9"
        assert "x-moaxy-advisor-model" not in headers
        # The runtime attributes are stamped on the context.
        assert ctx.__dict__.get("advisor_skipped") is True
        assert ctx.__dict__.get("advisor_skip_confidence") == 0.9

    def test_extra_002_threshold_boundary_inclusive(self):
        # 0.85 is inclusive (skips); 0.849 is exclusive (runs).
        # Two scripted adapters, both reflect turns=1, threshold=0.85.
        # First: confidence=0.85 → skip. Second: confidence=0.849 → run.

        async def _case(confidence: float) -> tuple[int, list[str]]:
            adapter = ScriptedAdapter(
                [
                    _response("initial", prompt_tokens=5, completion_tokens=2),
                    _response(
                        f"c\nREFLECT_CONFIDENCE: {confidence}",
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
                early_exit=True,
                threshold=0.85,
                advisor_model="deepseek-v4-pro:cloud",
                advisor_turns=1,
            )
            ctx = _build_context(route)
            await Orchestrator(adapter).run(ctx)
            return (
                len(adapter.calls),
                [e.type for e in ctx.events],
            )

        # 0.85: skip → 2 LLM calls (initial + critique only).
        calls_85, events_85 = asyncio.run(_case(0.85))
        assert calls_85 == 2
        assert "advisor" not in events_85
        assert "advisor_approve" not in events_85

        # 0.849: run → 4 LLM calls (initial + critique + revision + advisor).
        calls_849, events_849 = asyncio.run(_case(0.849))
        assert calls_849 == 4
        assert "advisor" in events_849
        assert "advisor_approve" in events_849

    @pytest.mark.asyncio
    async def test_extra_003_skip_header_present_in_both_states(self):
        # When advisor runs (low-confidence case) → 0/no.
        # When advisor skipped → 1/confidence=<x>.
        # The header is ALWAYS present (consistent observability).
        adapter_ran = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        route_ran = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route_ran)
        await Orchestrator(adapter_ran).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert "x-moaxy-advisor-skipped" in headers
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

        # Skipped case (separate route with early_exit=True).
        adapter_skip = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.95"),
            ]
        )
        route_skip = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx2 = _build_context(route_skip)
        await Orchestrator(adapter_skip).run(ctx2)
        headers2 = build_response_headers(ctx2, request_id=ctx2.request_id)
        assert "x-moaxy-advisor-skipped" in headers2
        assert headers2["x-moaxy-advisor-skipped"] == "1/confidence=0.95"

    @pytest.mark.asyncio
    async def test_extra_023_skip_saves_one_round_trip(self):
        # With confidence >= 0.85: 2 calls. With confidence < 0.85: 4 calls.
        # (initial + critique [+ revised + advisor] = 2 vs 4)
        skip_adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.9"),
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
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        run_ctx = _build_context(skip_route)
        await Orchestrator(run_adapter).run(run_ctx)
        assert len(run_adapter.calls) == 4

    @pytest.mark.asyncio
    async def test_extra_024_skip_does_not_apply_when_reflection_disabled(self):
        # When reflection.turns=0, there is no confidence signal.
        # The advisor MUST run; the header MUST be 0/no.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
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

        assert len(adapter.calls) == 2
        assert "advisor" in [e.type for e in ctx.events]
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-skipped"] == "0/no"

    def test_extra_030_skip_logs_info_message(self, caplog):
        # When the advisor is skipped, an INFO log line is emitted
        # containing the confidence and the model name.
        import logging

        async def _go() -> None:
            adapter = ScriptedAdapter(
                [
                    _response("initial"),
                    _response("c\nREFLECT_CONFIDENCE: 0.92"),
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
            with caplog.at_level(
                logging.INFO, logger="moaxy.pipeline.orchestrator"
            ):
                await Orchestrator(adapter).run(ctx)

        asyncio.run(_go())
        messages = [rec.message for rec in caplog.records]
        joined = " ".join(messages)
        assert "advisor skipped" in joined
        assert "confidence=0.92" in joined
        assert "deepseek-v4-pro:cloud" in joined


# ────────────────────────────────────────────────────────────────────
# DELTA 2 — Cross-critique prompt upgrade (VAL-PIPE-EXTRA-004, 005, 006)
# ────────────────────────────────────────────────────────────────────


class TestDelta2CrossCritique:
    """VAL-PIPE-EXTRA-004: cross-critique prompt parsed into event context."""

    @pytest.mark.asyncio
    async def test_extra_004_decision_and_score_flow_into_context(self):
        # When the advisor emits ADVISOR_DECISION: REVISE and
        # ADVISOR_SCORE: 7, the parsed event context exposes the
        # decision ("revise") and the integer score (7). The
        # orchestrator stamps ctx.__dict__["advisor_score"].
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                # Advisor emits cross-critique format.
                _response(
                    "ADVISOR_DECISION: REVISE\n"
                    "ADVISOR_SCORE: 7\n"
                    "ADVISOR_REVISE: better answer",
                    model="deepseek-v4-pro:cloud",
                ),
                # Primary-model revision after advisor REVISE.
                _response("primary-final"),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # advisor_score attribute is set to 7.
        assert ctx.__dict__.get("advisor_score") == 7
        # An advisor_score event was emitted.
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "7"
        # The headers reflect the score.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "7"

    @pytest.mark.asyncio
    async def test_extra_005_legacy_markers_still_work(self):
        # A model that emits ONLY the legacy ADVISOR_APPROVE marker
        # (no ADVISOR_DECISION:) is still parsed correctly.
        # decision == "approve", score is None, advisor_approve event
        # still emits.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
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
        types = [e.type for e in ctx.events]
        assert "advisor" in types
        assert "advisor_approve" in types
        # No advisor_score event was emitted.
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert score_events == []
        # advisor_score runtime attribute is unset.
        assert "advisor_score" not in ctx.__dict__

    @pytest.mark.asyncio
    async def test_extra_006_advisor_issues_parsed_into_list(self):
        # The orchestrator's plugin context exposes the parsed
        # ADVISOR_ISSUES bullet list. The test invokes advisor_turn
        # directly to inspect ctx.
        from moaxy.pipeline.advisor import advisor_turn

        adapter = ScriptedAdapter(
            [
                _response(
                    "ADVISOR_DECISION: REVISE\n"
                    "ADVISOR_ISSUES:\n"
                    "- issue 1\n"
                    "- issue 2\n"
                    "ADVISOR_REVISE: better",
                    model="deepseek-v4-pro:cloud",
                )
            ]
        )
        ctx_dict: dict[str, Any] = {"adapter": adapter}
        await advisor_turn(
            ctx_dict,
            advisor_model="deepseek-v4-pro:cloud",
            history=[],
            current_answer="x",
        )
        assert ctx_dict["advisor_issues"] == ["issue 1", "issue 2"]


# ────────────────────────────────────────────────────────────────────
# DELTA 3 — Per-route order (VAL-PIPE-EXTRA-007, 008, 009, 029)
# ────────────────────────────────────────────────────────────────────


class TestDelta3Order:
    """VAL-PIPE-EXTRA-007: order=advise_first routes calls in new order."""

    @pytest.mark.asyncio
    async def test_extra_007_advise_first_event_order(self):
        # Advise-first order: initial → advisor → reflect_critique
        # → reflect_revised. The reflection's critique input is
        # the post-advisor answer.
        adapter = ScriptedAdapter(
            [
                _response("initial", prompt_tokens=5, completion_tokens=2),
                # Advisor: REVISE with ADVISOR_REVISE so a revision
                # is produced.
                _response(
                    "ADVISOR_REVISE: advisor-suggested",
                    model="deepseek-v4-pro:cloud",
                ),
                # Primary-model revision after advisor REVISE.
                _response("primary-after-advisor"),
                # Reflection critique of the post-advisor answer.
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    prompt_tokens=4,
                    completion_tokens=6,
                ),
                # Reflection revision.
                _response("reflected-final"),
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
        # initial → advisor (event) → reflect_critique → reflect_revised.
        assert types[0] == "initial"
        assert "advisor" in types
        # The first 'advisor' event index must come before any
        # 'reflect_critique' or 'reflect_revised' index.
        first_advisor_idx = types.index("advisor")
        first_reflect_idx = next(
            i
            for i, t in enumerate(types)
            if t in ("reflect_critique", "reflect_revised")
        )
        assert first_advisor_idx < first_reflect_idx
        # Specifically, the sequence has at least one advisor event
        # BEFORE any reflect_critique event.
        assert "reflect_critique" in types
        # The reflection critique's call input should reflect the
        # post-advisor answer (the orchestrator's
        # current_answer after the advisor pass).
        reflect_critique_idx = types.index("reflect_critique")
        assert types[reflect_critique_idx - 1] in (
            "advisor_revised",
            "advisor",
            "advisor_approve",
            "advisor_revision",
        )

    @pytest.mark.asyncio
    async def test_extra_008_reflect_first_is_default(self):
        # When reflection.order is unset, the default is
        # "reflect_first" and the event ordering is the v1-v4
        # sequence: initial → reflect_* → advisor*.
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
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        # Default order is reflect_first.
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        # Confirm the default field value.
        assert route.reflection.order == "reflect_first"

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

    def test_extra_009_order_field_validates_literal(self):
        # ReflectionConfig.order is Literal["reflect_first", "advise_first"].
        # Invalid values raise ValidationError mentioning "order".
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ReflectionConfig(order="invalid_value")  # type: ignore[arg-type]
        msg = str(exc_info.value)
        assert "order" in msg

        # Both valid values parse successfully.
        r1 = ReflectionConfig(order="reflect_first")
        assert r1.order == "reflect_first"
        r2 = ReflectionConfig(order="advise_first")
        assert r2.order == "advise_first"

    @pytest.mark.asyncio
    async def test_extra_029_order_field_e2e_for_both_values(self):
        # End-to-end: both order values produce the expected event
        # sequence in the orchestrator. (Covered more strictly by
        # test_extra_007 and test_extra_008; this case consolidates
        # the two paths into one suite entry.)
        for order_value, expected_prefix in [
            ("reflect_first", ["initial", "reflect_critique"]),
            ("advise_first", ["initial", "advisor"]),
        ]:
            adapter = ScriptedAdapter(
                [
                    _response("initial"),
                    _response(
                        "ADVISOR_REVISE: better",
                        model="deepseek-v4-pro:cloud",
                    ),
                    _response("primary-after-advisor"),
                    _response("c\nREFLECT_CONFIDENCE: 0.5"),
                    _response("revised"),
                ]
            )
            route = _build_route(
                reflection_turns=1,
                early_exit=False,
                threshold=0.85,
                advisor_model="deepseek-v4-pro:cloud",
                advisor_turns=1,
                order=order_value,
            )
            ctx = _build_context(route)
            await Orchestrator(adapter).run(ctx)
            types = [e.type for e in ctx.events]
            assert types[: len(expected_prefix)] == expected_prefix, (
                f"order={order_value} produced types {types!r}; "
                f"expected prefix {expected_prefix!r}"
            )


# ────────────────────────────────────────────────────────────────────
# DELTA 5 — Weighted early exit (VAL-PIPE-EXTRA-010..015, 042)
# ────────────────────────────────────────────────────────────────────


class TestDelta5WeightedEarlyExit:
    """VAL-PIPE-EXTRA-010: trust_verbal=1.0, trust_score=0.0 uses
    REFLECT_CONFIDENCE only (v1 behavior)."""

    @pytest.mark.asyncio
    async def test_extra_010_verbal_only_uses_confidence(self):
        # trust_verbal=1.0, trust_score=0.0. SCORE: line is parsed
        # but does not affect the threshold. With confidence 0.9
        # and threshold 0.85, early-exit fires.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.9\nSCORE: 5",
                ),
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
        # 2 LLM calls (no revision) because the combined signal
        # equals confidence (0.9) and 0.9 >= 0.85.
        assert len(adapter.calls) == 2
        assert "reflect_early_exit" in [e.type for e in ctx.events]
        # The runtime attribute last_combined_signal equals
        # confidence (the score was parsed but unused).
        assert ctx.__dict__.get("last_combined_signal") == pytest.approx(0.9)
        # SCORE was parsed into last_score; it just didn't drive
        # the decision.
        assert ctx.__dict__.get("last_score") == 5

    @pytest.mark.asyncio
    async def test_extra_011_score_only_uses_score(self):
        # trust_verbal=0.0, trust_score=1.0. Early-exit check is
        # (score / 10) >= threshold. SCORE: 9 → 0.9 >= 0.85 → skip.
        # SCORE: 8 → 0.8 < 0.85 → continue.

        async def _case(score_value: int) -> tuple[int, bool]:
            adapter = ScriptedAdapter(
                [
                    _response("initial"),
                    _response(
                        f"c\nREFLECT_CONFIDENCE: 0.5\nSCORE: {score_value}"
                    ),
                    _response("revised"),
                ]
            )
            route = _build_route(
                reflection_turns=1,
                early_exit=True,
                threshold=0.85,
                trust_verbal=0.0,
                trust_score=1.0,
            )
            ctx = _build_context(route)
            await Orchestrator(adapter).run(ctx)
            return (
                len(adapter.calls),
                "reflect_early_exit" in [e.type for e in ctx.events],
            )

        # SCORE=9 → combined = 0.9 >= 0.85 → early-exit.
        calls_9, early_9 = await _case(9)
        assert calls_9 == 2
        assert early_9 is True

        # SCORE=8 → combined = 0.8 < 0.85 → continue to revision.
        calls_8, early_8 = await _case(8)
        assert calls_8 == 3
        assert early_8 is False

    @pytest.mark.asyncio
    async def test_extra_012_combined_weights(self):
        # trust_verbal=0.5, trust_score=0.5. With confidence 0.6,
        # score 9, threshold 0.7: combined = 0.5*0.6 + 0.5*0.9 = 0.75.
        # 0.75 >= 0.7 → early-exit fires.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.6\nSCORE: 9"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.7,
            trust_verbal=0.5,
            trust_score=0.5,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert "reflect_early_exit" in [e.type for e in ctx.events]
        assert ctx.__dict__.get("last_combined_signal") == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_extra_013_both_signals_missing_no_short_circuit(self):
        # When neither REFLECT_CONFIDENCE nor SCORE: is present, the
        # combined value is 0.0; with threshold 0.5, no early-exit.
        # The orchestrator continues to revision (the DELTA 7 safety
        # rule, since the missing-line case is malformed).
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("a critique with no signal lines"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.5,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 LLM calls: initial + critique + revision.
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        assert "reflect_revised" in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_extra_014_score_missing_falls_back_to_confidence(self):
        # REFLECT_CONFIDENCE: 0.9 only. With trust_verbal=0.6,
        # trust_score=0.4 the combined is 0.9 (v1 fallback).
        # 0.9 >= 0.85 → early-exit fires.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.9"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert "reflect_early_exit" in [e.type for e in ctx.events]
        # last_score is None (SCORE: was missing).
        assert ctx.__dict__.get("last_score") is None
        # last_combined_signal equals the confidence (fallback path).
        assert ctx.__dict__.get("last_combined_signal") == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_extra_015_confidence_missing_uses_score(self):
        # SCORE: 9 only, no REFLECT_CONFIDENCE. With trust_verbal=0.6,
        # trust_score=0.4: combined = 0.6*0.0 + 0.4*0.9 = 0.36.
        # With threshold 0.4, 0.36 < 0.4 → no early-exit.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("a critique with no confidence\nSCORE: 9"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.4,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        # The revision call did run.
        assert "reflect_revised" in [e.type for e in ctx.events]
        assert ctx.__dict__.get("last_score") == 9
        # Confidence is 0.0 (parse_confidence default for missing line).
        assert ctx.__dict__.get("last_combined_signal") == pytest.approx(0.36)


# ────────────────────────────────────────────────────────────────────
# DELTA 7 — Self-critique safety (VAL-PIPE-EXTRA-016, 017)
# ────────────────────────────────────────────────────────────────────


class TestDelta7SelfCritiqueSafety:
    """VAL-PIPE-EXTRA-016: malformed confidence does not short-circuit."""

    @pytest.mark.asyncio
    async def test_extra_016_malformed_does_not_short_circuit(self):
        # When early_exit=true and the critique has NO
        # REFLECT_CONFIDENCE: line at all (parse returns 0.0
        # because the regex did not match), the orchestrator
        # MUST NOT short-circuit. The DELTA 7 safety rule:
        # "model failed to follow the protocol" is treated as
        # the revision path, not the early-exit path.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("a critique with no signal lines"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.0,  # would short-circuit on any signal >= 0
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 LLM calls (initial + critique + revision).
        assert len(adapter.calls) == 3
        # No early-exit event (the malformed case is safety-routed).
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        # Revision ran.
        assert "reflect_revised" in [e.type for e in ctx.events]

    @pytest.mark.asyncio
    async def test_extra_017_explicit_zero_confidence_not_malformed(self):
        # When the model explicitly emits REFLECT_CONFIDENCE: 0.0
        # (a successfully parsed zero, not a missing line), the
        # orchestrator's behavior is governed by the v1 threshold
        # check: 0.0 < 0.85, so the revision runs as normal.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.0"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # 3 calls: initial + critique + revision (parity with
        # VAL-PIPE-EXTRA-016 in outcome, but via the threshold
        # path, not the safety path).
        assert len(adapter.calls) == 3
        assert "reflect_early_exit" not in [e.type for e in ctx.events]
        assert "reflect_revised" in [e.type for e in ctx.events]


# ────────────────────────────────────────────────────────────────────
# DELTA 6 — Score events and headers (VAL-PIPE-EXTRA-018..022)
# ────────────────────────────────────────────────────────────────────


class TestDelta6ScoreEventsAndHeaders:
    """VAL-PIPE-EXTRA-018: reflect_score event carries the parsed score."""

    @pytest.mark.asyncio
    async def test_extra_018_reflect_score_event_appended(self):
        # When the critique contains SCORE: 7, a reflect_score event
        # is appended with text="7" and turn=0. When SCORE: is
        # absent, no reflect_score event is appended.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7"),
                _response("revised"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        score_events = [
            e for e in ctx.events if e.type == "reflect_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "7"
        assert score_events[0].turn == 0

    @pytest.mark.asyncio
    async def test_extra_018b_no_reflect_score_event_when_absent(self):
        # When SCORE: is missing, no reflect_score event is appended.
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
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        score_events = [
            e for e in ctx.events if e.type == "reflect_score"
        ]
        assert score_events == []

    @pytest.mark.asyncio
    async def test_extra_019_advisor_score_event_appended(self):
        # When the advisor emits ADVISOR_SCORE: 8, an
        # advisor_score event is appended with text="8".
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 8",
                    model="deepseek-v4-pro:cloud",
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
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "8"

    @pytest.mark.asyncio
    async def test_extra_020_reflect_score_header_reflects_last_turn(self):
        # Header value is the LAST parsed SCORE: across all turns.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c1\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7"),
                _response("rev1"),
                _response("c2\nREFLECT_CONFIDENCE: 0.5\nSCORE: 9"),
                _response("rev2"),
            ]
        )
        route = _build_route(
            reflection_turns=2,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "9"

    @pytest.mark.asyncio
    async def test_extra_020b_reflect_score_header_zero_when_no_score(self):
        # When no SCORE: was parsed, the header value is "0".
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
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "0"

    @pytest.mark.asyncio
    async def test_extra_021_advisor_score_header_reflects_value(self):
        # When the advisor emits ADVISOR_SCORE: 8, the header is "8".
        # When ADVISOR_SCORE: is absent, the header is "0".
        adapter_present = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 8",
                    model="deepseek-v4-pro:cloud",
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter_present).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-advisor-score"] == "8"

        adapter_absent = ScriptedAdapter(
            [
                _response("initial"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        ctx2 = _build_context(route)
        await Orchestrator(adapter_absent).run(ctx2)
        headers2 = build_response_headers(ctx2, request_id=ctx2.request_id)
        assert headers2["x-moaxy-advisor-score"] == "0"

    @pytest.mark.asyncio
    async def test_extra_021b_advisor_score_header_when_skipped(self):
        # When the advisor is skipped (DELTA 1), the
        # x-moaxy-advisor-score header is absent (no advisor pass
        # ran; the helper only emits the header when an advisor
        # event was emitted). The x-moaxy-advisor-skipped header
        # is the canonical signal of the skip.
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.95"),
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
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The skip header is set with the parsed confidence.
        assert headers["x-moaxy-advisor-skipped"] == "1/confidence=0.95"
        # The advisor-model header is NOT set on a skip.
        assert "x-moaxy-advisor-model" not in headers

    @pytest.mark.asyncio
    async def test_extra_022_full_header_set_on_reflective_advisor_route(self):
        # A reflective+advisor route's response carries the full
        # M5 header set: request-id, alias-resolved,
        # fallbacks-used, reflect-turns, reflect-confidence,
        # reflect-score, advisor-model, advisor-score,
        # advisor-skipped.
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("c\nREFLECT_CONFIDENCE: 0.6\nSCORE: 5"),
                _response("revised", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 7\n",
                    model="deepseek-v4-pro:cloud",
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
        ctx = _build_context(route, request_id="req-x")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The required M5 header set is present.
        required = {
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
        assert required.issubset(headers.keys())
        # Specific values.
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert headers["x-moaxy-reflect-turns"] == "1"
        assert headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        assert headers["x-moaxy-reflect-score"] == "5"
        assert headers["x-moaxy-advisor-score"] == "7"
        # 0.6 formats as "0.6" via the :g format.
        assert headers["x-moaxy-reflect-confidence"] == "0.6"
        assert headers["x-moaxy-advisor-skipped"] == "0/no"


# ────────────────────────────────────────────────────────────────────
# Self-advise warning (VAL-PIPE-EXTRA-031)
# ────────────────────────────────────────────────────────────────────


class TestDeltaSelfAdviseWarning:
    """VAL-PIPE-EXTRA-031: self-advise warning when advisor.model == primary."""

    def test_extra_031_self_advise_logs_warning(self, caplog):
        # When advisor.model equals the primary's resolved model,
        # a WARNING is logged. The advisor call still proceeds.
        import logging

        async def _go() -> None:
            adapter = ScriptedAdapter(
                [
                    _response("initial", model="minimax-m3:cloud"),
                    _response(
                        "ADVISOR_APPROVE", model="minimax-m3:cloud"
                    ),
                ]
            )
            route = _build_route(
                reflection_turns=0,
                advisor_model="minimax-m3:cloud",
                advisor_turns=1,
                original_model="minimax-m3:cloud",
                resolved_model="minimax-m3:cloud",
            )
            ctx = _build_context(route)
            with caplog.at_level(
                logging.WARNING, logger="moaxy.pipeline.orchestrator"
            ):
                await Orchestrator(adapter).run(ctx)

        asyncio.run(_go())
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ]
        assert any(
            "self-advise" in rec.message.lower() for rec in warnings
        )

    @pytest.mark.asyncio
    async def test_extra_031b_self_advise_still_runs_advisor(self):
        # The self-advise warning is non-fatal; the advisor call
        # still proceeds.
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_APPROVE", model="minimax-m3:cloud"
                ),
            ]
        )
        route = _build_route(
            reflection_turns=0,
            advisor_model="minimax-m3:cloud",
            advisor_turns=1,
            original_model="minimax-m3:cloud",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert "advisor" in [e.type for e in ctx.events]
        assert "advisor_approve" in [e.type for e in ctx.events]


# ────────────────────────────────────────────────────────────────────
# Streaming parity (VAL-PIPE-EXTRA-032, 033, 034)
# ────────────────────────────────────────────────────────────────────


class TestDeltaStreamingParity:
    """VAL-PIPE-EXTRA-032: weighted early exit on streaming path."""

    @pytest.mark.asyncio
    async def test_extra_032_stream_reflect_score_in_trailer(self):
        # The streaming path emits the trailing SSE trailer with
        # the M5 x-moaxy-* headers.
        adapter = ScriptedAdapter(
            stream_script=[["initial ", "answer"]],
            script=[
                _response("c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 6"),
                _response("revised"),
            ],
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        ctx = _build_context(route, request_id="req-stream-32")
        # Drive the streaming path.
        chunks: list[bytes] = []
        async for chunk in Orchestrator(adapter).stream_run(ctx):
            chunks.append(chunk)
        # The trailer's `data:` payload is the second-to-last event
        # (the last is [DONE]). Find the trailer in the events.
        events = _parse_sse_events(_chunks_to_text(chunks))
        trailer_payload = None
        for ev_name, data in events:
            if ev_name is None and "x_moaxy" in data:
                try:
                    trailer_payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
        assert trailer_payload is not None, (
            f"No x_moaxy trailer found in {events!r}"
        )
        x_moaxy = trailer_payload["x_moaxy"]
        # The M5 score headers are present.
        assert x_moaxy.get("x-moaxy-reflect-score") == "6"

    @pytest.mark.asyncio
    async def test_extra_033_stream_advise_first_order(self):
        # The streaming path honors reflection.order=advise_first.
        # The advisor_revised revision event is emitted BEFORE the
        # reflect_revised event in the SSE stream.
        adapter = ScriptedAdapter(
            stream_script=[["initial"]],
            script=[
                _response(
                    "ADVISOR_REVISE: better",
                    model="deepseek-v4-pro:cloud",
                ),
                _response("primary-after-advisor"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("reflected-final"),
            ],
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
            order="advise_first",
        )
        ctx = _build_context(route, request_id="req-stream-33")
        chunks: list[bytes] = []
        async for chunk in Orchestrator(adapter).stream_run(ctx):
            chunks.append(chunk)
        events = _parse_sse_events(_chunks_to_text(chunks))
        # Two 'revision' events: one for advisor_revised, one for
        # reflect_revised. The advisor_revised must come first.
        revision_indices = [
            i for i, (ev_name, _data) in enumerate(events)
            if ev_name == "revision"
        ]
        assert len(revision_indices) >= 2
        # The events list ordering is preserved by index; the
        # advisor_revised's text is "better" and the
        # reflect_revised's text is "reflected-final".
        revision_texts = [
            json.loads(events[i][1])["text"]
            for i in revision_indices
        ]
        # The first revision event is the advisor revision, the
        # second is the reflection revision.
        assert revision_texts[0] == "better"
        assert revision_texts[1] == "reflected-final"

    @pytest.mark.asyncio
    async def test_extra_034_stream_skip_advisor_when_confidence_high(self):
        # The streaming path applies DELTA 1 conditional skip.
        # No event: revision for the advisor; trailer carries
        # x-moaxy-advisor-skipped: 1/confidence=0.9.
        adapter = ScriptedAdapter(
            stream_script=[["Hello, ", "world!"]],
            script=[
                _response("c\nREFLECT_CONFIDENCE: 0.9"),
            ],
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        ctx = _build_context(route, request_id="req-stream-34")
        chunks: list[bytes] = []
        async for chunk in Orchestrator(adapter).stream_run(ctx):
            chunks.append(chunk)
        events = _parse_sse_events(_chunks_to_text(chunks))
        # No revision event in the stream (early-exit + skip).
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 0
        # The trailer carries the skip header.
        trailer_payload = None
        for ev_name, data in events:
            if ev_name is None and "x_moaxy" in data:
                try:
                    trailer_payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
        assert trailer_payload is not None
        x_moaxy = trailer_payload["x_moaxy"]
        assert x_moaxy["x-moaxy-advisor-skipped"] == "1/confidence=0.9"


def _chunks_to_text(chunks: list[bytes]) -> str:
    """Concatenate the SSE chunks into a single body string."""
    return b"".join(chunks).decode("utf-8", errors="replace")


def _parse_sse_events(body: str) -> list[tuple[str | None, str]]:
    """Parse an SSE response body into ``(event_name, data_payload)`` pairs.

    Mirrors the helper in ``tests/test_streaming.py``.
    """
    events: list[tuple[str | None, str]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in body.split("\n"):
        if line == "":
            if current_data or current_event is not None:
                events.append((current_event, "\n".join(current_data)))
            current_event = None
            current_data = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                current_event = value
            elif field == "data":
                current_data.append(value)
    if current_data or current_event is not None:
        events.append((current_event, "\n".join(current_data)))
    return events


# ────────────────────────────────────────────────────────────────────
# Prompts (VAL-PIPE-EXTRA-025, 026)
# ────────────────────────────────────────────────────────────────────


class TestDeltaPrompts:
    """VAL-PIPE-EXTRA-025: default cross-critique prompt contains all
    four new markers + the two legacy markers."""

    def test_extra_025_default_advisor_prompt_has_all_markers(self):
        # The default advisor prompt contains all six contract
        # substrings: ADVISOR_DECISION, ADVISOR_SCORE,
        # ADVISOR_ISSUES, ADVISOR_SUGGESTIONS, plus the legacy
        # ADVISOR_APPROVE and ADVISOR_REVISE.
        for marker in (
            "ADVISOR_DECISION:",
            "ADVISOR_SCORE:",
            "ADVISOR_ISSUES:",
            "ADVISOR_SUGGESTIONS:",
            "ADVISOR_APPROVE",
            "ADVISOR_REVISE:",
        ):
            assert marker in DEFAULT_ADVISOR_PROMPT

    def test_extra_026_default_advisor_prompt_backward_compat(self):
        # The default advisor prompt still contains the legacy
        # ADVISOR_APPROVE and ADVISOR_REVISE: markers. The
        # substring check is a regression guard.
        assert "ADVISOR_APPROVE" in DEFAULT_ADVISOR_PROMPT
        assert "ADVISOR_REVISE:" in DEFAULT_ADVISOR_PROMPT

    def test_extra_025b_default_reflect_prompt_has_score_marker(self):
        # The M5 reflector prompt requests the optional SCORE: line.
        assert "SCORE:" in DEFAULT_REFLECT_PROMPT
        # Backward-compat: REFLECT_CONFIDENCE: is still requested.
        assert "REFLECT_CONFIDENCE:" in DEFAULT_REFLECT_PROMPT


# ────────────────────────────────────────────────────────────────────
# ReflectionConfig field validation (VAL-PIPE-EXTRA-027, 028)
# ────────────────────────────────────────────────────────────────────


class TestDeltaReflectionConfig:
    """VAL-PIPE-EXTRA-027: trust_verbal / trust_score field validation."""

    def test_extra_027_trust_fields_default_values(self):
        # Defaults are trust_verbal=0.6, trust_score=0.4.
        r = ReflectionConfig()
        assert r.trust_verbal == 0.6
        assert r.trust_score == 0.4

    def test_extra_027_trust_fields_accept_non_negative(self):
        # Non-negative values are accepted.
        r = ReflectionConfig(trust_verbal=0.0, trust_score=0.0)
        assert r.trust_verbal == 0.0
        assert r.trust_score == 0.0
        r2 = ReflectionConfig(trust_verbal=1.5, trust_score=2.0)
        assert r2.trust_verbal == 1.5
        assert r2.trust_score == 2.0

    def test_extra_027_trust_fields_reject_negative(self):
        # Negative values raise ValidationError.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReflectionConfig(trust_verbal=-0.1)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            ReflectionConfig(trust_score=-0.5)  # type: ignore[arg-type]

    def test_extra_027_trust_fields_reject_non_numeric(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReflectionConfig(trust_verbal="low")  # type: ignore[arg-type]

    def test_extra_028_threshold_field_unchanged(self):
        # The threshold field is unchanged. Default 0.85.
        r = ReflectionConfig()
        assert r.threshold == 0.85
        # Configurable.
        r2 = ReflectionConfig(threshold=0.9)
        assert r2.threshold == 0.9
        # Out-of-range still rejected.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReflectionConfig(threshold=1.5)
        with pytest.raises(ValidationError):
            ReflectionConfig(threshold=-0.1)


# ────────────────────────────────────────────────────────────────────
# Backwards-compat invariants (VAL-PIPE-EXTRA-035, 036, 037, 038)
# ────────────────────────────────────────────────────────────────────


class TestDeltaBackwardCompat:
    """VAL-PIPE-EXTRA-036: tests/test_plugins.py passes unchanged.

    The dedicated test in this class runs the file as a subprocess
    and asserts the exit code, satisfying the contract's
    "explicitly run tests/test_plugins.py" requirement.
    """

    def test_extra_036_test_plugins_py_passes(self):
        # Run the existing tests/test_plugins.py file as a subprocess
        # and assert exit code 0. The contract is explicit:
        # "explicitly runs tests/test_plugins.py and asserts exit code 0".
        assert PLUGINS_TEST_FILE.exists(), (
            f"expected {PLUGINS_TEST_FILE} to exist"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(PLUGINS_TEST_FILE.relative_to(PROJECT_ROOT)),
                "-x",
                "-q",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"pytest tests/test_plugins.py exited with code "
            f"{result.returncode}; stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

    def test_extra_037_reflect_confidence_regex_unchanged(self):
        # The REFLECT_CONFIDENCE regex pins the v1 behavior.
        # parse_confidence must continue to extract floats from
        # the last line of a critique.
        assert parse_confidence("REFLECT_CONFIDENCE: 0.92") == 0.92
        assert parse_confidence("c\nREFLECT_CONFIDENCE: 0.5") == 0.5
        # Multi-line with the last match winning.
        assert (
            parse_confidence("REFLECT_CONFIDENCE: 0.3\nc\nREFLECT_CONFIDENCE: 0.7")
            == 0.7
        )
        # Missing line → 0.0 (v1 invariant).
        assert parse_confidence("no marker") == 0.0
        assert parse_confidence("") == 0.0

    def test_extra_037b_parse_score_is_additive(self):
        # The new SCORE: regex is a NEW parser. It is additive
        # and does NOT alter the REFLECT_CONFIDENCE parser's
        # behavior. parse_score extracts integers; parse_confidence
        # extracts floats; both can coexist in the same text.
        text = "REFLECT_CONFIDENCE: 0.9\nSCORE: 7"
        assert parse_confidence(text) == 0.9
        assert parse_score(text) == 7
        # parse_score returns None when the line is missing.
        assert parse_score("no marker") is None

    def test_extra_038_advisor_marker_regex_unchanged(self):
        # The legacy ADVISOR_APPROVE substring and
        # ADVISOR_REVISE: substring parsers are unchanged.
        from moaxy.pipeline.advisor import (
            parse_advisor_response,
        )

        # Legacy approve.
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_APPROVE"
        )
        assert decision == "approve"
        assert text is None
        # Legacy revise.
        decision, text, _score, _issues = parse_advisor_response(
            "ADVISOR_REVISE: better"
        )
        assert decision == "revise"
        assert text == "better"
        # The new ADVISOR_DECISION:/ADVISOR_SCORE:/ADVISOR_ISSUES:
        # parsers are additive (tried alongside, not replacing,
        # the legacy path).
        text_cross = (
            "ADVISOR_DECISION: APPROVE\n"
            "ADVISOR_SCORE: 8\n"
            "ADVISOR_ISSUES:\n"
            "- a\n"
            "- b\n"
        )
        decision, _text, score, issues = parse_advisor_response(text_cross)
        assert decision == "approve"
        assert score == 8
        assert issues == ["a", "b"]

    @pytest.mark.asyncio
    async def test_extra_035_existing_v1_v4_reflection_tests_parity(self):
        # v1-v4 reflection tests (with REFLECT_CONFIDENCE: only,
        # no SCORE: line) still pass with the M5 defaults. The
        # score-missing path falls back to confidence (v1 invariant).
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.95"),
            ]
        )
        route = _build_route(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # Early-exit fires.
        assert "reflect_early_exit" in [e.type for e in ctx.events]
        # 2 LLM calls.
        assert len(adapter.calls) == 2


# ────────────────────────────────────────────────────────────────────
# Parser regex pin (VAL-PIPE-EXTRA-039, 040, 041, 042)
# ────────────────────────────────────────────────────────────────────


class TestDeltaParserPins:
    """VAL-PIPE-EXTRA-039: parse_score regex is anchored and integer-only."""

    def test_extra_039_parse_score_anchored(self):
        # Anchored per line, MULTILINE, integer-only.
        assert parse_score("SCORE: 7") == 7
        # Integer-only: floats are rejected.
        assert parse_score("SCORE: 7.5") is None
        assert parse_score("SCORE: 0.92") is None
        # Non-numeric rejected.
        assert parse_score("SCORE: seven") is None
        # Out-of-range: parser records the integer (does not clamp).
        assert parse_score("SCORE: 11") == 11
        # Missing line.
        assert parse_score("no marker") is None
        assert parse_score("") is None
        assert parse_score(None) is None  # type: ignore[arg-type]
        # Substring match fails (anchored).
        assert parse_score("XSCORE: 7") is None
        assert parse_score("SCORE: 7Y") is None
        # Case-sensitive.
        assert parse_score("score: 7") is None

    def test_extra_040_parse_advisor_score_anchored(self):
        from moaxy.pipeline.advisor import parse_advisor_score

        # Anchored per line, MULTILINE, integer-only.
        assert parse_advisor_score("ADVISOR_SCORE: 8") == 8
        # Integer-only.
        assert parse_advisor_score("ADVISOR_SCORE: 8.5") is None
        assert parse_advisor_score("ADVISOR_SCORE: eight") is None
        # Out-of-range recorded as-is.
        assert parse_advisor_score("ADVISOR_SCORE: 11") == 11
        # Missing line.
        assert parse_advisor_score("no marker") is None
        assert parse_advisor_score("") is None
        # Substring match fails.
        assert parse_advisor_score("XADVISOR_SCORE: 8") is None
        # Case-sensitive.
        assert parse_advisor_score("advisor_score: 8") is None

    def test_extra_041_advisor_issues_bullet_parsing(self):
        # parse_advisor_issues tolerates - * and • markers.
        from moaxy.pipeline.advisor import parse_advisor_issues

        # - markers.
        text = "ADVISOR_ISSUES:\n- issue 1\n- issue 2"
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]
        # * markers.
        text = "ADVISOR_ISSUES:\n* issue 1\n* issue 2"
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]
        # • (U+2022) markers.
        text = "ADVISOR_ISSUES:\n• issue 1\n• issue 2"
        assert parse_advisor_issues(text) == ["issue 1", "issue 2"]
        # Mixed markers in a single block.
        text = "ADVISOR_ISSUES:\n- one\n* two\n• three"
        assert parse_advisor_issues(text) == ["one", "two", "three"]
        # Empty bullets skipped.
        text = "ADVISOR_ISSUES:\n- \n- real"
        assert parse_advisor_issues(text) == ["real"]
        # Missing header returns [].
        assert parse_advisor_issues("no header here") == []
        # Empty input.
        assert parse_advisor_issues("") == []
        assert parse_advisor_issues(None) == []  # type: ignore[arg-type]

    def test_extra_042_parse_weighted_signal_helper(self):
        # The helper is importable from moaxy.pipeline.reflector and
        # returns (combined, confidence, score).
        # 0.6 * 0.9 + 0.4 * 0.5 = 0.74.
        combined, confidence, score = parse_weighted_signal(
            "REFLECT_CONFIDENCE: 0.9\nSCORE: 5",
            trust_verbal=0.6,
            trust_score=0.4,
        )
        assert combined == pytest.approx(0.74)
        assert confidence == pytest.approx(0.9)
        assert score == 5

        # Score-missing falls back to confidence.
        combined, confidence, score = parse_weighted_signal(
            "REFLECT_CONFIDENCE: 0.5",
            trust_verbal=0.6,
            trust_score=0.4,
        )
        assert combined == pytest.approx(0.5)
        assert confidence == pytest.approx(0.5)
        assert score is None

        # Both missing → 0.0.
        combined, confidence, score = parse_weighted_signal(
            "", trust_verbal=0.6, trust_score=0.4
        )
        assert combined == 0.0
        assert confidence == 0.0
        assert score is None

        # The helper is also re-exported from the package.
        from moaxy.pipeline import (
            parse_weighted_signal as PkgParseWeightedSignal,
        )

        assert PkgParseWeightedSignal is parse_weighted_signal


# ────────────────────────────────────────────────────────────────────
# Full response surface (config + in-process app) — sanity check
# ────────────────────────────────────────────────────────────────────


class TestDelta5MoaxyConfigWiring:
    """The M5 ReflectionConfig fields land on MoaxyConfig when set in YAML."""

    def test_extra_027_moaxy_config_picks_up_new_fields(self):
        # Construct a full MoaxyConfig that sets order, trust_verbal,
        # trust_score on a route's reflection block. Pydantic must
        # accept the values and surface them.
        route = RouteConfig(
            name="delta5",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="local",
            reflection=ReflectionConfig(
                turns=1,
                order="advise_first",
                trust_verbal=0.7,
                trust_score=0.3,
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="local", adapter="ollama", base_url="http://x"
                )
            ],
            routes=[route],
        )
        assert cfg.routes[0].reflection.order == "advise_first"
        assert cfg.routes[0].reflection.trust_verbal == 0.7
        assert cfg.routes[0].reflection.trust_score == 0.3
        assert cfg.routes[0].advisor.model == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# Module exports
# ────────────────────────────────────────────────────────────────────


class TestDelta5ModuleExports:
    """The M5 helpers are re-exported from the :mod:`moaxy.pipeline`
    package."""

    def test_parse_score_re_exported(self):
        from moaxy.pipeline import parse_score as PkgParseScore

        assert PkgParseScore is parse_score

    def test_parse_weighted_signal_re_exported(self):
        from moaxy.pipeline import (
            parse_weighted_signal as PkgParseWeightedSignal,
        )

        assert PkgParseWeightedSignal is parse_weighted_signal

    def test_parse_advisor_score_re_exported(self):
        from moaxy.pipeline import (
            parse_advisor_score as PkgParseAdvisorScore,
        )
        from moaxy.pipeline.advisor import parse_advisor_score

        assert PkgParseAdvisorScore is parse_advisor_score

    def test_parse_advisor_issues_re_exported(self):
        from moaxy.pipeline import (
            parse_advisor_issues as PkgParseAdvisorIssues,
        )
        from moaxy.pipeline.advisor import parse_advisor_issues

        assert PkgParseAdvisorIssues is parse_advisor_issues
