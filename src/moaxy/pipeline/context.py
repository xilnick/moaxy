"""Typed pipeline context and event types for the moaxy orchestrator.

The orchestrator threads a single :class:`PipelineContext` value object
through every stage (initial generation, reflection turns, advisor
stage, fallback walks). Keeping state in one place makes the request
lifecycle observable end-to-end: the ``events`` list is the trace the
proxy surfaces in ``x-moaxy-*`` response headers and structured logs.

This module is intentionally adapter-agnostic. It depends on:

* :class:`moaxy.adapters.base.UsageAccumulator` — for summed token usage.
* :class:`moaxy.routing.matcher.RouteMatch` — for the routing decision.

The body field is typed as ``dict[str, Any]`` because the request body is
already a generic OpenAI-shaped mapping by the time the pipeline sees
it; tightening it to a Pydantic model would couple the pipeline to a
specific wire shape that may change between client and backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from moaxy.adapters.base import ChatResponse, UsageAccumulator
from moaxy.routing.matcher import RouteMatch


@dataclass
class PipelineEvent:
    """A single observable event in the pipeline lifecycle.

    The orchestrator appends one of these to
    :attr:`PipelineContext.events` for every meaningful step: the
    initial answer, each reflection critique and revision, each advisor
    pass, and any early exits. Only the fields relevant to a given
    event are populated; consumers must treat ``turn``, ``model``, and
    ``text`` as optional.
    """

    type: str
    turn: int | None = None
    model: str | None = None
    text: str | None = None


@dataclass
class PipelineContext:
    """The typed context threaded through every pipeline stage.

    Fields are intentionally mutable: the orchestrator mutates the
    context in place as each stage runs (e.g. ``upstream_response`` is
    replaced on every revision). A request maps 1:1 to a context; the
    context is created at request entry and discarded when the response
    is serialised.

    Attributes:
        request_id: Opaque correlation id propagated to logs and
            ``x-moaxy-request-id``.
        request: The original OpenAI-shaped request body (the
            ``messages``, ``model``, ``stream``, ``temperature``, etc.
            fields).
        route: The :class:`moaxy.routing.matcher.RouteMatch` chosen by
            the route matcher. ``None`` when no route matched; the
            server should reject the request before reaching the
            pipeline in that case.
        principal: The authenticated principal (API key id, role
            claims, etc.) attached by the auth gate. ``None`` when
            auth is disabled or the request is unauthenticated.
        model_alias_resolved: The alias-resolved real model name
            (e.g. ``"minimax-m3:cloud"`` for an alias
            ``coder-pro -> minimax-m3:cloud``). Equal to the client's
            ``request["model"]`` when no alias applies.
        target_backend: The backend name selected by the route (e.g.
            ``"ollama-local"``). ``None`` for routes that resolve at
            runtime (weighted, round-robin).
        upstream_response: The most recent :class:`ChatResponse` from
            the adapter. Replaced on every reflection revision and
            advisor pass; the final value is what gets serialised to
            the client.
        usage: Running token-usage totals across every adapter call in
            the request. The orchestrator adds each call's usage; the
            snapshot is exposed in the final response.
        events: Ordered list of :class:`PipelineEvent` records. The
            server reads this list to populate ``x-moaxy-reflect-turns``,
            ``x-moaxy-reflect-confidence``, ``x-moaxy-advisor-model``,
            and ``x-moaxy-fallbacks-used`` response headers.
        original_model: The model name the client sent, kept verbatim
            for response echo. The final response's ``model`` field
            must echo the original alias, not the resolved name.
    """

    request_id: str
    request: dict[str, Any]
    route: RouteMatch | None
    principal: dict[str, Any] | None = None
    model_alias_resolved: str = ""
    target_backend: str | None = None
    upstream_response: ChatResponse | None = None
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)
    events: list[PipelineEvent] = field(default_factory=list)
    original_model: str = ""

    def append_event(
        self,
        type: str,
        *,
        turn: int | None = None,
        model: str | None = None,
        text: str | None = None,
    ) -> PipelineEvent:
        """Append a :class:`PipelineEvent` and return it for inspection.

        Convenience for the orchestrator: ``ctx.append_event("initial",
        model=model)`` reads more naturally than constructing an event
        and appending it by hand.
        """
        event = PipelineEvent(type=type, turn=turn, model=model, text=text)
        self.events.append(event)
        return event


__all__ = ["PipelineContext", "PipelineEvent"]
