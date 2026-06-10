"""Top-level pipeline orchestrator for the moaxy proxy.

The :class:`Orchestrator` (with its single public coroutine
:meth:`Orchestrator.run`) is the workhorse of the data-plane: it
threads a :class:`~moaxy.pipeline.context.PipelineContext` through the
full per-request LLM loop — initial generation, the optional
self-reflection loop, the optional advisor pass — and accumulates
usage, emits structured :class:`~moaxy.pipeline.context.PipelineEvent`
records, and stamps ``x-moaxy-*`` metadata onto the final
:class:`~moaxy.adapters.base.ChatResponse`.

Algorithm
---------

The orchestrator implements the algorithm described in
``architecture.md``::

    async def run(ctx):
        # 0. Resolve alias and pick the primary model.
        # 1. Initial generation: one call_with_fallbacks.
        # 2. Reflection loop (0..3 turns, sequential):
        #    for each turn:
        #      a. Critique via call_with_fallbacks; emit reflect_critique.
        #      b. If this is the LAST turn AND early_exit is on AND
        #         confidence >= threshold, emit reflect_early_exit
        #         and break (no revision on the last turn).
        #      c. Otherwise do the revision; emit reflect_revised.
        #         If early_exit clears the threshold (and there are
        #         remaining turns) emit reflect_early_exit AFTER the
        #         revision and break.
        # 3. Advisor: at most one call_with_fallbacks against the
        #    configured advisor model. The advisor's input is the
        #    post-reflection answer.
        # 4. Stamp the original alias back into the response.model field.

Each LLM call is wrapped in :func:`moaxy.pipeline.fallback.call_with_fallbacks`
so the per-step retry budget and per-route fallback list apply uniformly
to every site (initial, critique, revision, advisor). When a step succeeds
on the primary model, ``fallbacks_used`` is empty; the orchestrator
combines the per-step lists so the response header reflects every model
the walker actually used past the primary.

Headers and events
------------------

The orchestrator's run() returns the same :class:`PipelineContext` it
was given (the context is mutated in place so the caller can read
``ctx.events``, ``ctx.usage``, ``ctx.upstream_response`` after the call).
It does NOT set ``x-moaxy-*`` response headers directly; the FastAPI
handler reads ``ctx.events`` and :attr:`PipelineContext.usage` to
materialise the response via :func:`build_response_headers`. The
orchestrator is responsible only for the *content* of the headers;
the server is responsible for *setting* them on the HTTP response.
This separation keeps the orchestrator testable in isolation.

Parallelism
-----------

The orchestrator implements the M2 sequential default. Both
``reflection.parallel`` and ``advisor.parallel`` are accepted by the
config (they are M4 toggles), but the default ``parallel: false`` path
runs every step in source order, one after another. The M4 parallel
mode is a follow-on feature; this module ships the sequential
reference implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UsageAccumulator,
)
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.fallback import call_with_fallbacks
from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_reflection_messages,
    build_revision_messages,
)
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
)
from moaxy.pipeline.reflector import parse_confidence

logger = logging.getLogger(__name__)


def _resolved_model_chain(ctx: PipelineContext) -> list[str]:
    """Build the model list to feed :func:`call_with_fallbacks`.

    The primary is the alias-resolved real model; the fallbacks are the
    route's per-route fallback list. The chain is returned in invocation
    order (primary first). The orchestrator never re-applies alias
    resolution to the fallback entries — they are model identifiers
    understood by the backend adapter as written in config.
    """
    route = ctx.route
    assert route is not None  # the server rejects no-route requests before the pipeline
    primary = ctx.model_alias_resolved or route.resolved_model
    fallbacks = list(route.fallbacks)
    return [primary, *fallbacks]


def _advisor_model_chain(ctx: PipelineContext) -> list[str]:
    """Build the model list for the advisor step.

    The advisor's primary is the configured :attr:`AdvisorConfig.model`
    on the matched route. The advisor reuses the route's fallback list
    and retry budget so the per-model fallback policy applies uniformly.
    """
    route = ctx.route
    assert route is not None
    advisor_model = route.advisor.model
    assert advisor_model is not None
    return [advisor_model, *list(route.fallbacks)]


def _reflect_system_prompt(ctx: PipelineContext) -> str:
    """Return the reflector system prompt to use for this request.

    The route's per-route ``reflection.system_prompt`` wins; when it is
    unset, the default :data:`DEFAULT_REFLECT_PROMPT` is used.
    """
    route = ctx.route
    assert route is not None
    return route.reflection.system_prompt or DEFAULT_REFLECT_PROMPT


def _advisor_system_prompt(ctx: PipelineContext) -> str:
    """Return the advisor system prompt to use for this request.

    The route's per-route ``advisor.system_prompt`` wins; when it is
    unset, the default :data:`DEFAULT_ADVISOR_PROMPT` is used.
    """
    route = ctx.route
    assert route is not None
    return route.advisor.system_prompt or DEFAULT_ADVISOR_PROMPT


def _sampling_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Extract the OpenAI sampling parameters from the request body.

    The orchestrator forwards ``temperature``, ``top_p``, ``max_tokens``,
    and any other non-``model``/``messages``/``stream`` field to every
    LLM call so the upstream provider sees the same sampling parameters
    the client sent. (VAL-PIPE-042 pins this contract.)
    """
    return {
        k: v
        for k, v in request.items()
        if k not in {"model", "messages", "stream"}
    }


def _accumulate(usage: UsageAccumulator, response: ChatResponse) -> None:
    """Add the response's token usage to the running accumulator."""
    usage.add(response.usage)


def _text_of(response: ChatResponse) -> str:
    """Return the assistant content of a :class:`ChatResponse`."""
    return response.message.content or ""


def _is_advisor_approval(text: str) -> bool:
    """Return True if the advisor text approves the previous answer.

    The advisor emits ``ADVISOR_APPROVE`` (optionally with trailing
    text/whitespace) when it has nothing to revise. Any other output
    is treated as a revision.
    """
    if not isinstance(text, str) or not text:
        return False
    return "ADVISOR_APPROVE" in text


def _advisor_revised_text(text: str) -> str:
    """Extract the revised text after an ``ADVISOR_REVISE:`` marker.

    The advisor's revised answer lives after the literal ``ADVISOR_REVISE:``
    prefix on the first line. When the marker is missing, the entire
    text is returned unchanged; that case should be rare (a model that
    follows the intent but forgets the prefix), and the orchestrator
    conservatively treats the whole thing as the revised answer.
    """
    if not isinstance(text, str):
        return ""
    marker = "ADVISOR_REVISE:"
    idx = text.find(marker)
    if idx < 0:
        return text.strip()
    return text[idx + len(marker):].lstrip("\n").strip()


def _has_advice_marker(text: str) -> bool:
    """Return True if the advisor's text contains the ``ADVISOR_REVISE:`` marker."""
    if not isinstance(text, str) or not text:
        return False
    return "ADVISOR_REVISE:" in text


class Orchestrator:
    """The per-request LLM orchestrator.

    The orchestrator is stateless from request to request: every
    :meth:`run` invocation mutates the supplied
    :class:`PipelineContext` in place but holds no request-scoped state
    on the instance itself. The class is intentionally lightweight
    (it is constructed once at app startup and re-used across
    requests); tests can construct a fresh instance per test.

    Attributes:
        adapter: The :class:`moaxy.adapters.base.Adapter` that
            :meth:`run` dispatches every LLM call through. The same
            adapter is used for the initial call, every reflection
            step, and the advisor pass; per-model retries and
            fallbacks are layered on top by
            :func:`moaxy.pipeline.fallback.call_with_fallbacks`.
    """

    def __init__(self, adapter: Adapter) -> None:
        self.adapter = adapter

    async def _initial_call(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
    ) -> tuple[ChatResponse, list[str]]:
        """Make the initial-generation LLM call via the fallback walker.

        Returns:
            A ``(response, fallbacks_used)`` tuple. ``response`` is the
            adapter's :class:`ChatResponse`. ``fallbacks_used`` is the
            list of fallback model names the walker actually used
            past the primary.
        """
        assert ctx.route is not None
        kwargs = _sampling_kwargs(ctx.request)
        messages: list[dict[str, Any]] = ctx.request.get("messages", [])
        response, fallbacks_used = await call_with_fallbacks(
            self.adapter,
            models=model_chain,
            retry=ctx.route.retry,
            messages=messages,
            **kwargs,
        )
        return response, fallbacks_used

    async def _reflection_critique(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        current_answer: str,
        turn: int,
    ) -> tuple[ChatResponse, str, float, list[str]]:
        """Run the critique half of one reflection turn.

        Builds the critique message list, dispatches via the fallback
        walker, and parses the confidence off the response text.
        Returns ``(response, critique_text, confidence, fallbacks_used)``.
        """
        assert ctx.route is not None
        kwargs = _sampling_kwargs(ctx.request)
        messages = build_reflection_messages(
            history=history,
            answer=current_answer,
            system_prompt=_reflect_system_prompt(ctx),
        )
        response, fallbacks_used = await call_with_fallbacks(
            self.adapter,
            models=model_chain,
            retry=ctx.route.retry,
            messages=messages,
            **kwargs,
        )
        text = _text_of(response)
        confidence = parse_confidence(text)
        return response, text, confidence, fallbacks_used

    async def _reflection_revision(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        answer: str,
        critique: str,
    ) -> tuple[ChatResponse, list[str]]:
        """Run the revision half of one reflection turn.

        Builds the revision message list (history + critique), dispatches
        via the fallback walker, and returns the new response and the
        fallback models actually used.
        """
        assert ctx.route is not None
        kwargs = _sampling_kwargs(ctx.request)
        messages = build_revision_messages(
            history=history,
            answer=answer,
            critique=critique,
            system_prompt=_reflect_system_prompt(ctx),
        )
        response, fallbacks_used = await call_with_fallbacks(
            self.adapter,
            models=model_chain,
            retry=ctx.route.retry,
            messages=messages,
            **kwargs,
        )
        return response, fallbacks_used

    async def _advisor_call(
        self,
        ctx: PipelineContext,
        *,
        advisor_chain: list[str],
        history: list[dict[str, Any]],
        answer: str,
    ) -> tuple[ChatResponse, list[str]]:
        """Run the advisor pass over the post-reflection answer.

        Builds the advisor message list, dispatches via the fallback
        walker, and returns the response and the fallback models
        actually used.
        """
        assert ctx.route is not None
        kwargs = _sampling_kwargs(ctx.request)
        messages = build_advisor_messages(
            history=history,
            answer=answer,
            system_prompt=_advisor_system_prompt(ctx),
        )
        response, fallbacks_used = await call_with_fallbacks(
            self.adapter,
            models=advisor_chain,
            retry=ctx.route.retry,
            messages=messages,
            **kwargs,
        )
        return response, fallbacks_used

    async def _run_reflection(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        current_answer: str,
    ) -> tuple[str, list[str], int]:
        """Run the 0..3-turn reflection loop and return the final answer.

        Returns:
            A ``(final_answer, total_fallbacks_used, turns_executed)``
            tuple. ``final_answer`` is the assistant text to use as the
            next stage's input (either the initial answer when no
            reflection ran, or the most recent revision when at least
            one turn completed). ``total_fallbacks_used`` aggregates
            every fallback the walker used across the critique and
            revision calls (per the contract, the response header
            reports the union). ``turns_executed`` is the count of
            ``reflect_critique`` events emitted (i.e. the count of
            turns the orchestrator actually attempted).
        """
        assert ctx.route is not None
        reflect_cfg = ctx.route.reflection
        answer = current_answer
        total_fallbacks: list[str] = []
        turns_executed = 0

        for turn in range(reflect_cfg.turns):
            turns_executed += 1
            critique_response, critique_text, confidence, fallbacks_used = (
                await self._reflection_critique(
                    ctx,
                    model_chain=model_chain,
                    history=history,
                    current_answer=answer,
                    turn=turn,
                )
            )
            total_fallbacks.extend(fallbacks_used)
            _accumulate(ctx.usage, critique_response)
            ctx.append_event(
                "reflect_critique",
                turn=turn,
                model=critique_response.model or model_chain[0],
                text=critique_text,
            )
            # Remember the last parsed confidence for the response header.
            ctx.__dict__["last_confidence"] = confidence

            is_last_turn = turn == reflect_cfg.turns - 1
            clears_threshold = (
                reflect_cfg.early_exit and confidence >= reflect_cfg.threshold
            )

            # On the last turn, an early-exit short-circuits the
            # revision: the model has declared itself confident and we
            # trust it. On any earlier turn, we still want a revised
            # answer to feed the next iteration's critique, then we
            # break out of the loop after the revision.
            if is_last_turn and clears_threshold:
                ctx.append_event(
                    "reflect_early_exit",
                    turn=turn,
                    model=critique_response.model or model_chain[0],
                )
                logger.info(
                    "orchestrator: early exit at reflection turn=%d "
                    "(confidence=%.3f >= threshold=%.3f)",
                    turn,
                    confidence,
                    reflect_cfg.threshold,
                )
                break

            revision_response, rev_fallbacks_used = await self._reflection_revision(
                ctx,
                model_chain=model_chain,
                history=history,
                answer=answer,
                critique=critique_text,
            )
            total_fallbacks.extend(rev_fallbacks_used)
            _accumulate(ctx.usage, revision_response)
            answer = _text_of(revision_response)
            ctx.append_event(
                "reflect_revised",
                turn=turn,
                model=revision_response.model or model_chain[0],
                text=answer,
            )

            if clears_threshold:
                ctx.append_event(
                    "reflect_early_exit",
                    turn=turn,
                    model=revision_response.model or model_chain[0],
                )
                logger.info(
                    "orchestrator: early exit at reflection turn=%d "
                    "(confidence=%.3f >= threshold=%.3f) after revision",
                    turn,
                    confidence,
                    reflect_cfg.threshold,
                )
                break

        return answer, total_fallbacks, turns_executed

    async def _run_advisor(
        self,
        ctx: PipelineContext,
        *,
        advisor_chain: list[str],
        history: list[dict[str, Any]],
        answer: str,
    ) -> tuple[str, list[str]]:
        """Run the 0..1-turn advisor pass and return the (possibly revised) answer.

        Returns:
            A ``(final_answer, fallbacks_used)`` tuple. The advisor's
            input is the post-reflection answer; its output either
            approves the previous answer (in which case the input is
            returned unchanged) or revises it (in which case the
            post-``ADVISOR_REVISE:`` text is returned).
        """
        assert ctx.route is not None
        advisor_cfg = ctx.route.advisor
        if advisor_cfg.turns < 1 or not advisor_cfg.model:
            return answer, []

        response, fallbacks_used = await self._advisor_call(
            ctx,
            advisor_chain=advisor_chain,
            history=history,
            answer=answer,
        )
        _accumulate(ctx.usage, response)
        text = _text_of(response)
        ctx.append_event(
            "advisor",
            model=response.model or advisor_cfg.model,
            text=text,
        )

        if _is_advisor_approval(text):
            ctx.append_event(
                "advisor_approve",
                model=response.model or advisor_cfg.model,
            )
            return answer, fallbacks_used

        revised = _advisor_revised_text(text) if _has_advice_marker(text) else text.strip()
        ctx.append_event(
            "advisor_revised",
            model=response.model or advisor_cfg.model,
            text=revised,
        )
        return revised, fallbacks_used

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run the full initial → reflection → advisor pipeline.

        The coroutine mutates ``ctx`` in place:

        * ``ctx.upstream_response`` is set to the final
          :class:`ChatResponse` (or replaced on every revision and
          advisor revise).
        * ``ctx.usage`` is summed across every LLM call.
        * ``ctx.events`` is appended to in execution order.
        * ``ctx.model_alias_resolved`` is set when missing.
        * ``ctx.target_backend`` is set when missing.

        The coroutine also stamps the original alias the client sent
        back into the response's ``model`` field (so the response
        ``body.model == request["model"]`` regardless of which real
        model produced the answer).

        Args:
            ctx: The :class:`PipelineContext` describing the request.
                Must have a non-``None`` ``route`` (the server rejects
                no-route requests before reaching the pipeline), a
                ``request_id``, a ``request`` body with at least
                ``"messages"``, and a ``model_alias_resolved`` (or a
                route that exposes one).

        Returns:
            The same :class:`PipelineContext` for caller convenience.
            The context's :attr:`PipelineContext.upstream_response`
            carries the final assistant message and the
            :attr:`PipelineContext.usage` field carries the summed
            token counts.

        Raises:
            UpstreamError: A permanent (4xx) failure from the adapter
                during any step. Bubbles up unchanged; the caller
                turns this into an HTTP 4xx.
            UpstreamExhaustedError: Every model in the chain has
                exhausted its retry budget. The exception message
                contains the substring ``"all backends failed"`` and
                the caller's error handler turns it into HTTP 502.
            RuntimeError: A pre-condition is violated (no route, no
                adapter, etc.).
        """
        if ctx.route is None:
            raise RuntimeError("PipelineContext.route must be set before run()")
        if self.adapter is None:
            raise RuntimeError("Orchestrator.adapter must be set")

        # Stage 0: alias resolution bookkeeping. The server is expected
        # to populate ctx.model_alias_resolved before the pipeline
        # runs, but be defensive: derive it from the route when missing.
        if not ctx.model_alias_resolved:
            ctx.model_alias_resolved = ctx.route.resolved_model
        if ctx.target_backend is None:
            ctx.target_backend = ctx.route.backend

        original_model = ctx.original_model or ctx.request.get("model", "")
        ctx.original_model = original_model

        model_chain = _resolved_model_chain(ctx)
        history: list[dict[str, Any]] = list(ctx.request.get("messages", []))

        # Stage 1: initial generation.
        initial_response, initial_fallbacks = await self._initial_call(
            ctx, model_chain=model_chain
        )
        _accumulate(ctx.usage, initial_response)
        ctx.upstream_response = initial_response
        current_answer = _text_of(initial_response)
        ctx.append_event(
            "initial",
            model=initial_response.model or model_chain[0],
            text=current_answer,
        )
        fallbacks_used: list[str] = list(initial_fallbacks)

        # Stage 2: reflection loop (0..3 turns, sequential by default).
        if ctx.route.reflection.turns > 0:
            reflect_answer, reflect_fallbacks, _turns_executed = await self._run_reflection(
                ctx,
                model_chain=model_chain,
                history=history,
                current_answer=current_answer,
            )
            current_answer = reflect_answer
            fallbacks_used.extend(reflect_fallbacks)
            # The reflection loop replaces ``ctx.upstream_response`` in
            # place on every revision; the latest revision (or the
            # initial answer, when every turn was early-exited) is
            # what the next stage (advisor) reads. We do not rebuild
            # the response here; the final rebuild happens at Stage 4.

        # Stage 3: advisor (0..1 turn, sequential after reflection).
        if ctx.route.advisor.turns >= 1 and ctx.route.advisor.model:
            advisor_chain = _advisor_model_chain(ctx)
            advisor_answer, advisor_fallbacks = await self._run_advisor(
                ctx,
                advisor_chain=advisor_chain,
                history=history,
                answer=current_answer,
            )
            current_answer = advisor_answer
            fallbacks_used.extend(advisor_fallbacks)
            if ctx.upstream_response is not None:
                ctx.upstream_response = ChatResponse(
                    id=ctx.upstream_response.id,
                    model=ctx.upstream_response.model,
                    message=Message(role="assistant", content=current_answer),
                    usage=ctx.upstream_response.usage,
                    finish_reason=ctx.upstream_response.finish_reason or "stop",
                )

        # Stage 4: stamp the original alias the client sent into the
        # response's ``model`` field. The response handler reads
        # ``ctx.upstream_response.model`` when building the wire body;
        # setting it here keeps the alias echo single-sourced. The
        # ``usage`` field on the response is the *accumulated* snapshot
        # across every LLM call in the pipeline so the final response
        # reflects the total token cost of the request, not just the
        # last call.
        if ctx.upstream_response is not None:
            ctx.upstream_response = ChatResponse(
                id=ctx.upstream_response.id,
                model=original_model,
                message=Message(
                    role=ctx.upstream_response.message.role,
                    content=current_answer,
                ),
                usage=ctx.usage.snapshot(),
                finish_reason=ctx.upstream_response.finish_reason or "stop",
            )

        logger.debug(
            "orchestrator: request_id=%s events=%d fallbacks=%d",
            ctx.request_id,
            len(ctx.events),
            len(fallbacks_used),
        )

        # Stash the aggregated fallback list on the context for the
        # response handler. The dataclass schema does not have this
        # field, so we add it as a runtime attribute. The server's
        # response builder reads ``ctx.fallbacks_used`` when setting
        # the ``x-moaxy-fallbacks-used`` header.
        ctx.__dict__["fallbacks_used"] = fallbacks_used
        return ctx


def build_response_headers(ctx: PipelineContext, *, request_id: str) -> dict[str, str]:
    """Build the ``x-moaxy-*`` response headers for a finished pipeline run.

    The orchestrator populates ``ctx.events`` and ``ctx.usage`` while it
    runs; the server calls this helper after the run completes to
    materialise the final response headers. The header values are
    derived deterministically from the event log so any server can
    reproduce them by walking the events.

    Args:
        ctx: The :class:`PipelineContext` returned by
            :meth:`Orchestrator.run`.
        request_id: The opaque correlation id (also returned in the
            ``x-moaxy-request-id`` header). The server typically reads
            this off the middleware-supplied ``ctx.request_id`` and
            passes it in to keep the helper pure.

    Returns:
        A dict suitable for ``fastapi.responses.JSONResponse(headers=...)``.
        Always contains ``x-moaxy-request-id``. Contains
        ``x-moaxy-alias-resolved`` when a route was matched (the value
        equals the alias-resolved real model name). Contains
        ``x-moaxy-fallbacks-used`` reflecting the aggregated list of
        fallback models the walker actually used. Contains
        ``x-moaxy-reflect-turns``, ``x-moaxy-reflect-confidence``, and
        ``x-moaxy-advisor-model`` when reflection and/or advisor ran.
    """
    headers: dict[str, str] = {"x-moaxy-request-id": request_id}

    if ctx.route is not None:
        headers["x-moaxy-alias-resolved"] = (
            ctx.model_alias_resolved or ctx.route.resolved_model
        )

    fallbacks_used = ctx.__dict__.get("fallbacks_used", [])
    if fallbacks_used:
        import json
        headers["x-moaxy-fallbacks-used"] = json.dumps(list(fallbacks_used))
    else:
        headers["x-moaxy-fallbacks-used"] = "0"

    reflect_turns = sum(1 for e in ctx.events if e.type == "reflect_critique")
    headers["x-moaxy-reflect-turns"] = str(reflect_turns)

    last_confidence = ctx.__dict__.get("last_confidence", 0.0) or 0.0
    headers["x-moaxy-reflect-confidence"] = f"{last_confidence:g}"

    advisor_model = None
    for event in ctx.events:
        if event.type in ("advisor", "advisor_approve", "advisor_revised"):
            if event.model:
                advisor_model = event.model
    if advisor_model:
        headers["x-moaxy-advisor-model"] = advisor_model

    return headers


__all__ = [
    "Orchestrator",
    "build_response_headers",
]
