"""Tests for the pipeline context and event types.

Covers:
- :class:`PipelineContext` constructs with all required fields and
  sensible defaults.
- :class:`PipelineEvent` is a typed dataclass with the documented
  fields.
- :meth:`PipelineContext.append_event` records events in order and
  returns the recorded event.
- The default system prompts in :mod:`moaxy.pipeline.prompts` contain
  the literal substrings the validation contract pins (VAL-PIPE-010).
"""

from __future__ import annotations

import dataclasses

from moaxy.adapters.base import (
    ChatResponse,
    Message,
    Usage,
    UsageAccumulator,
)
from moaxy.models.config import (
    AdvisorConfig,
    ReflectionConfig,
    RouteConfig,
)
from moaxy.models.config import RouteMatch as ConfigRouteMatch
from moaxy.pipeline import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
    PipelineContext,
    PipelineEvent,
)
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT as PROMPTS_ADVISOR,
)
from moaxy.pipeline.prompts import (
    DEFAULT_REFLECT_PROMPT as PROMPTS_REFLECT,
)
from moaxy.routing.matcher import RouteMatch


def _build_route_match() -> RouteMatch:
    """Build a :class:`RouteMatch` for use as a context ``route`` value."""
    config_route = RouteConfig(
        name="reflective-coder",
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases={"coder-pro": "minimax-m3:cloud"},
        fallbacks=["minimax-m2.7:cloud"],
        retry=2,
        reflection=ReflectionConfig(turns=1, early_exit=True, threshold=0.85),
        advisor=AdvisorConfig(model="deepseek-v4-pro:cloud", turns=1),
    )
    return RouteMatch(
        route=config_route,
        original_model="coder-pro",
        resolved_model="minimax-m3:cloud",
        backend="ollama-local",
        path="/v1/chat/completions",
        reflection=config_route.reflection,
        advisor=config_route.advisor,
        fallbacks=list(config_route.fallbacks),
        retry=config_route.retry,
        aliases=dict(config_route.aliases),
    )


# -----------------------------------------------------------------------------
# PipelineEvent
# -----------------------------------------------------------------------------


class TestPipelineEvent:
    """The :class:`PipelineEvent` dataclass shape and defaults."""

    def test_is_a_dataclass(self):
        assert dataclasses.is_dataclass(PipelineEvent)

    def test_required_type_field(self):
        ev = PipelineEvent(type="initial")
        assert ev.type == "initial"

    def test_turn_model_text_are_optional(self):
        ev = PipelineEvent(type="reflect_critique")
        assert ev.turn is None
        assert ev.model is None
        assert ev.text is None

    def test_all_optional_fields_can_be_set(self):
        ev = PipelineEvent(
            type="reflect_revised",
            turn=2,
            model="minimax-m3:cloud",
            text="revised answer",
        )
        assert ev.type == "reflect_revised"
        assert ev.turn == 2
        assert ev.model == "minimax-m3:cloud"
        assert ev.text == "revised answer"

    def test_event_field_names(self):
        names = {f.name for f in dataclasses.fields(PipelineEvent)}
        assert names == {"type", "turn", "model", "text"}


# -----------------------------------------------------------------------------
# PipelineContext
# -----------------------------------------------------------------------------


class TestPipelineContextFields:
    """The :class:`PipelineContext` dataclass exposes the documented fields."""

    def test_is_a_dataclass(self):
        assert dataclasses.is_dataclass(PipelineContext)

    def test_required_fields_present(self):
        names = {f.name for f in dataclasses.fields(PipelineContext)}
        assert names == {
            "request_id",
            "request",
            "route",
            "principal",
            "model_alias_resolved",
            "target_backend",
            "upstream_response",
            "usage",
            "events",
            "original_model",
        }

    def test_default_construction(self):
        ctx = PipelineContext(
            request_id="req-1",
            request={"model": "coder-pro", "messages": []},
            route=_build_route_match(),
        )
        assert ctx.request_id == "req-1"
        assert ctx.request == {"model": "coder-pro", "messages": []}
        assert ctx.route is not None
        assert ctx.principal is None
        assert ctx.model_alias_resolved == ""
        assert ctx.target_backend is None
        assert ctx.upstream_response is None
        assert isinstance(ctx.usage, UsageAccumulator)
        assert ctx.events == []
        assert ctx.original_model == ""

    def test_full_construction(self):
        route = _build_route_match()
        usage = UsageAccumulator()
        usage.add(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        response = ChatResponse(
            id="r1",
            model="minimax-m3:cloud",
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        ctx = PipelineContext(
            request_id="req-1",
            request={"model": "coder-pro", "messages": []},
            route=route,
            principal={"key_id": "k1", "roles": ["admin"]},
            model_alias_resolved="minimax-m3:cloud",
            target_backend="ollama-local",
            upstream_response=response,
            usage=usage,
            events=[PipelineEvent(type="initial", model="minimax-m3:cloud")],
            original_model="coder-pro",
        )
        assert ctx.principal == {"key_id": "k1", "roles": ["admin"]}
        assert ctx.model_alias_resolved == "minimax-m3:cloud"
        assert ctx.target_backend == "ollama-local"
        assert ctx.upstream_response is response
        assert ctx.usage is usage
        assert len(ctx.events) == 1
        assert ctx.original_model == "coder-pro"

    def test_events_default_to_independent_lists(self):
        ctx_a = PipelineContext(request_id="a", request={}, route=None)
        ctx_b = PipelineContext(request_id="b", request={}, route=None)
        ctx_a.append_event("initial", model="m1")
        assert ctx_b.events == []


class TestPipelineContextUsage:
    """The context's :class:`UsageAccumulator` sums usage across calls."""

    def test_default_usage_starts_at_zero(self):
        ctx = PipelineContext(request_id="r", request={}, route=None)
        snap = ctx.usage.snapshot()
        assert snap.prompt_tokens == 0
        assert snap.completion_tokens == 0
        assert snap.total_tokens == 0

    def test_usage_accumulates_across_calls(self):
        ctx = PipelineContext(request_id="r", request={}, route=None)
        ctx.usage.add(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        ctx.usage.add(Usage(prompt_tokens=20, completion_tokens=7, total_tokens=27))
        snap = ctx.usage.snapshot()
        assert snap.prompt_tokens == 30
        assert snap.completion_tokens == 12
        assert snap.total_tokens == 42


class TestPipelineContextAppendEvent:
    """The :meth:`append_event` helper records events in order."""

    def test_append_event_minimal(self):
        ctx = PipelineContext(request_id="r", request={}, route=None)
        ctx.append_event("initial")
        assert len(ctx.events) == 1
        assert ctx.events[0].type == "initial"
        assert ctx.events[0].turn is None
        assert ctx.events[0].model is None
        assert ctx.events[0].text is None

    def test_append_event_returns_event(self):
        ctx = PipelineContext(request_id="r", request={}, route=None)
        ev = ctx.append_event("reflect_critique", turn=1, model="m", text="critique")
        assert ev in ctx.events
        assert ev.type == "reflect_critique"
        assert ev.turn == 1
        assert ev.model == "m"
        assert ev.text == "critique"

    def test_append_event_preserves_order(self):
        ctx = PipelineContext(request_id="r", request={}, route=None)
        ctx.append_event("initial", model="m1")
        ctx.append_event("reflect_critique", turn=0, text="c0")
        ctx.append_event("reflect_revised", turn=0, text="r0")
        ctx.append_event("reflect_early_exit", turn=1)
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_early_exit",
        ]


# -----------------------------------------------------------------------------
# Default system prompts
# -----------------------------------------------------------------------------


class TestDefaultPrompts:
    """The default prompts are non-empty and contain the required substrings.

    These tests pin the contract for VAL-PIPE-010 (and the analogous
    advisor-prompt assertion). Editing the prompt text? Update the
    validator contract in lockstep.
    """

    def test_default_reflect_prompt_contains_reflect_confidence(self):
        assert "REFLECT_CONFIDENCE:" in DEFAULT_REFLECT_PROMPT
        # and the same constant is reachable from the prompts module
        assert "REFLECT_CONFIDENCE:" in PROMPTS_REFLECT

    def test_default_reflect_prompt_is_nonempty_string(self):
        assert isinstance(DEFAULT_REFLECT_PROMPT, str)
        assert DEFAULT_REFLECT_PROMPT.strip() != ""

    def test_default_advisor_prompt_contains_advisor_approve(self):
        assert "ADVISOR_APPROVE" in DEFAULT_ADVISOR_PROMPT
        assert "ADVISOR_APPROVE" in PROMPTS_ADVISOR

    def test_default_advisor_prompt_contains_advisor_revise_colon(self):
        assert "ADVISOR_REVISE:" in DEFAULT_ADVISOR_PROMPT
        assert "ADVISOR_REVISE:" in PROMPTS_ADVISOR

    def test_default_advisor_prompt_is_nonempty_string(self):
        assert isinstance(DEFAULT_ADVISOR_PROMPT, str)
        assert DEFAULT_ADVISOR_PROMPT.strip() != ""

    def test_pipeline_package_re_exports_prompts(self):
        # The pipeline __init__ re-exports both constants.
        assert DEFAULT_REFLECT_PROMPT is PROMPTS_REFLECT
        assert DEFAULT_ADVISOR_PROMPT is PROMPTS_ADVISOR


# -----------------------------------------------------------------------------
# Routing integration
# -----------------------------------------------------------------------------


class TestPipelineContextRouteIntegration:
    """PipelineContext integrates with the routing layer's RouteMatch type."""

    def test_route_is_route_match_instance(self):
        ctx = PipelineContext(
            request_id="r",
            request={},
            route=_build_route_match(),
        )
        assert isinstance(ctx.route, RouteMatch)

    def test_route_can_be_none(self):
        # A request that did not match any route still flows through the
        # pipeline (the server rejects it earlier, but the type allows None).
        ctx = PipelineContext(request_id="r", request={}, route=None)
        assert ctx.route is None

    def test_alias_resolved_field_separate_from_original(self):
        # The contract: original_model echoes the client's value;
        # model_alias_resolved holds the real model name.
        ctx = PipelineContext(
            request_id="r",
            request={"model": "coder-pro"},
            route=_build_route_match(),
            model_alias_resolved="minimax-m3:cloud",
            original_model="coder-pro",
        )
        assert ctx.original_model == "coder-pro"
        assert ctx.model_alias_resolved == "minimax-m3:cloud"
        assert ctx.original_model != ctx.model_alias_resolved
