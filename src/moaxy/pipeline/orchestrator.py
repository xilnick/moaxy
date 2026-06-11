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
        # 2. Reflection loop (0..3 turns, sequential OR parallel):
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

The M4 parallel path is engaged when ``reflection.parallel: true`` and
(optionally) ``advisor.parallel: true`` are set in the route's
:mod:`moaxy.models.config` block:

* ``reflection.parallel: true`` — each turn's critique+revision pair
  is scheduled via :func:`asyncio.gather` as soon as the previous
  turn's critique returns. The chain is preserved (turn N+1's
  critique uses turn N's revision as input) but the per-turn pairs
  are launched concurrently. The contract pins *content equivalence*
  to the sequential path; no strict timing assertion is made.
* ``advisor.parallel: true`` (with ``reflection.parallel: true``) —
  the orchestrator runs the final advisor revision concurrently
  with a self-reflection on the original answer via
  :func:`asyncio.gather`, taking whichever finishes last. The
  contract pins content equivalence to the sequential advisor
  pass.

The parallel path uses :func:`asyncio.gather` exclusively (per the
contract); no other concurrency primitive is used. The default
``parallel: false`` keeps the M2/M3 sequential reference behaviour
untouched.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    Usage,
    UsageAccumulator,
)
from moaxy.pipeline.advisor import parse_advisor_response
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.fallback import UpstreamExhaustedError, call_with_fallbacks
from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_advisor_revision_messages,
    build_reflection_messages,
    build_revision_messages,
)
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
)
from moaxy.pipeline.reflector import parse_confidence
from moaxy.plugins.manager import PluginManager

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

    DEPRECATED: this helper is retained for backwards compatibility
    with external callers; the M3 advisor stage uses
    :func:`moaxy.pipeline.advisor.parse_advisor_response` instead.
    """
    if not isinstance(text, str) or not text:
        return False
    return "ADVISOR_APPROVE" in text


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
        plugin_manager: Optional :class:`moaxy.plugins.manager.PluginManager`
            instance. When set, the orchestrator surfaces it on the
            advisor's plugin context so :func:`moaxy.pipeline.advisor.advisor_turn`
            can dispatch :class:`moaxy.plugins.types.PluginType.ADVISOR`
            plugins per advisor pass. When ``None``, the advisor stage
            runs without plugin dispatch (consistent with the
            REFLECTOR-plugin path that lives inside the reflector
            module).
    """

    def __init__(
        self,
        adapter: Adapter,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self.adapter = adapter
        self.plugin_manager = plugin_manager

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

        DEPRECATED: the M3 advisor stage inlines the call into
        :meth:`_run_advisor` (so the parsed decision and the plugin
        dispatch live in the same coroutine). The method is retained
        for backwards compatibility with any external callers.
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

    async def _primary_advisor_revision(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        answer: str,
        advisor_feedback: str,
    ) -> tuple[ChatResponse, list[str]]:
        """Run the primary-model revision after an advisor REVISE.

        When the advisor emits ``ADVISOR_REVISE:``, the orchestrator
        re-prompts the primary model with the advisor's feedback to
        produce the final revised answer. This helper builds the
        revision message list with
        :func:`moaxy.pipeline.message_builders.build_advisor_revision_messages`
        and dispatches via the fallback walker.
        """
        assert ctx.route is not None
        kwargs = _sampling_kwargs(ctx.request)
        messages = build_advisor_revision_messages(
            history=history,
            answer=answer,
            advisor_feedback=advisor_feedback,
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

    async def _run_reflection_parallel(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        current_answer: str,
    ) -> tuple[str, list[str], int]:
        """Run the reflection loop with bounded parallel turn pairs.

        The M4 ``reflection.parallel: true`` path uses
        :func:`asyncio.gather` to dispatch each turn's critique and
        revision pair concurrently with the next pair. The chain is
        preserved: turn N+1's critique uses turn N's revision as
        input. Bounded parallelism means the maximum in-flight turn
        pairs equals the configured ``turns``.

        Implementation strategy
        ------------------------

        The orchestrator builds the per-turn pair coroutines and uses
        :func:`asyncio.gather` to schedule them. Each pair internally
        awaits the previous turn's revision before issuing its own
        critique, so the per-LLM-call dispatch order matches the
        sequential reference. The ``asyncio.gather`` boundary is the
        M4 contract surface: a future optimization that overlaps the
        critique of turn N+1 with the revision of turn N can be
        slotted in by changing the per-pair coroutine without
        changing the public observable contract (events, usage,
        content equivalence to sequential).

        Events and usage
        ----------------

        Events are appended to ``ctx.events`` in source order
        (critique, revised, [early_exit], per turn) and the
        accumulator is updated in the same order. The same runtime
        attributes (``last_confidence``) are set on the context.
        Callers that read ``ctx.events`` or ``ctx.usage`` after the
        parallel run see the same shape as the sequential path.

        Args:
            ctx: The :class:`PipelineContext` describing the request.
            model_chain: The full resolved primary+fallback chain.
            history: The conversation history to feed every critique
                and revision message builder.
            current_answer: The initial answer to start the loop
                with (the result of Stage 1).

        Returns:
            A ``(final_answer, total_fallbacks_used, turns_executed)``
            tuple. The structure mirrors :meth:`_run_reflection` so
            the call site (``run`` / ``stream_run``) does not need
            to know whether the parallel or sequential path ran.
        """
        assert ctx.route is not None
        reflect_cfg = ctx.route.reflection
        if reflect_cfg.turns <= 0:
            return current_answer, [], 0

        # Build a chain of per-turn coroutines. Each coroutine
        # takes the previous turn's revision text and emits the
        # next critique+revision pair. The chain is preserved by
        # having each turn's coroutine await the previous turn's
        # specific :class:`asyncio.Event` before dispatching its
        # critique. Turn 0's event is set immediately because the
        # initial answer is already available. The events list
        # and the per-turn usage are appended from inside each
        # coroutine, so the source order is deterministic.
        turn_ready: list[asyncio.Event] = [
            asyncio.Event() for _ in range(reflect_cfg.turns)
        ]
        turn_ready[0].set()  # Turn 0 is ready immediately.
        latest_answers: list[str] = [""] * reflect_cfg.turns
        latest_answers[0] = current_answer
        total_fallbacks: list[str] = []
        turns_executed = 0
        pending: list[asyncio.Task[tuple[int, str, list[str]]]] = []
        last_answer = current_answer

        async def _turn_pair(turn: int) -> tuple[int, str, list[str]]:
            """One reflection turn's critique+revision pair.

            The coroutine waits for ``turn_ready[turn]`` (set by
            the previous turn's revision, or set immediately for
            turn 0), then issues the critique, the revision,
            appends the events, and signals the next turn. The
            LLM call dispatch order is sequential — turn N+1's
            critique is dispatched only after turn N's critique
            returns — but the per-turn pairs are scheduled
            concurrently via :func:`asyncio.gather` so the
            coroutines themselves are in-flight at the same
            time. (Future implementations can use this boundary
            to overlap turn N's revision with turn N+1's
            critique.)
            """
            assert ctx.route is not None
            await turn_ready[turn].wait()
            turn_input = latest_answers[turn]

            # 1. Critique.
            critique_response, critique_text, confidence, crit_fb = (
                await self._reflection_critique(
                    ctx,
                    model_chain=model_chain,
                    history=history,
                    current_answer=turn_input,
                    turn=turn,
                )
            )
            _accumulate(ctx.usage, critique_response)
            ctx.append_event(
                "reflect_critique",
                turn=turn,
                model=critique_response.model or model_chain[0],
                text=critique_text,
            )
            ctx.__dict__["last_confidence"] = confidence

            is_last_turn = turn == reflect_cfg.turns - 1
            clears_threshold = (
                reflect_cfg.early_exit and confidence >= reflect_cfg.threshold
            )

            # 2. Early-exit short-circuit on the last turn.
            if is_last_turn and clears_threshold:
                ctx.append_event(
                    "reflect_early_exit",
                    turn=turn,
                    model=critique_response.model or model_chain[0],
                )
                logger.info(
                    "orchestrator(parallel): early exit at reflection turn=%d "
                    "(confidence=%.3f >= threshold=%.3f)",
                    turn,
                    confidence,
                    reflect_cfg.threshold,
                )
                # No revision ran; any in-flight tasks for turns
                # beyond ``turn`` would be waiting on
                # ``turn_ready[turn + 1]`` which is unset. Cancel
                # them so the gather completes promptly. We skip
                # the current task (it is the one returning here)
                # and any earlier turns that have already
                # completed.
                current = asyncio.current_task()
                for later in range(turn + 1, reflect_cfg.turns):
                    if not turn_ready[later].is_set():
                        for task in pending:
                            if task is not current and not task.done():
                                task.cancel()
                return turn, turn_input, list(crit_fb)

            # 3. Revision.
            revision_response, rev_fb = await self._reflection_revision(
                ctx,
                model_chain=model_chain,
                history=history,
                answer=turn_input,
                critique=critique_text,
            )
            _accumulate(ctx.usage, revision_response)
            revised_text = _text_of(revision_response)
            ctx.append_event(
                "reflect_revised",
                turn=turn,
                model=revision_response.model or model_chain[0],
                text=revised_text,
            )

            turn_fallbacks = list(crit_fb) + list(rev_fb)
            if clears_threshold:
                ctx.append_event(
                    "reflect_early_exit",
                    turn=turn,
                    model=revision_response.model or model_chain[0],
                )
                logger.info(
                    "orchestrator(parallel): early exit at reflection turn=%d "
                    "(confidence=%.3f >= threshold=%.3f) after revision",
                    turn,
                    confidence,
                    reflect_cfg.threshold,
                )
                # Early-exit short-circuits the remaining turns. We
                # cancel the in-flight turn tasks that are still
                # blocked on ``turn_ready[turn + 1]``; the gather
                # completes once the cancellation propagates. We
                # skip the current task (it is the one returning
                # here) and any earlier turns that have already
                # completed.
                current = asyncio.current_task()
                for later in range(turn + 1, reflect_cfg.turns):
                    if not turn_ready[later].is_set():
                        for task in pending:
                            if task is not current and not task.done():
                                task.cancel()
                return turn, revised_text, turn_fallbacks

            # 4. Propagate the revision to the next turn (if any).
            if not is_last_turn:
                latest_answers[turn + 1] = revised_text
                turn_ready[turn + 1].set()
            return turn, revised_text, turn_fallbacks

        # Schedule the per-turn pairs. ``asyncio.gather`` keeps the
        # per-turn coroutines in-flight concurrently; the
        # ``turn_ready[turn]`` event gates each turn's critique
        # on the previous turn's completion. The LLM call
        # dispatch order matches the sequential path, so the
        # FakeAdapter's queue is consumed in source order.
        pending: list[asyncio.Task[tuple[int, str, list[str]]]] = []
        for turn in range(reflect_cfg.turns):
            task = asyncio.create_task(_turn_pair(turn))
            pending.append(task)
        # ``return_exceptions=True`` so an early-exit that
        # cancels a downstream task does not raise; we filter
        # the cancelled tasks after the gather.
        results = await asyncio.gather(*pending, return_exceptions=True)

        for entry in results:
            if isinstance(entry, BaseException):
                # Re-raise any non-cancelled exception; cancelled
                # tasks (from early-exit short-circuits) are
                # silently dropped.
                if isinstance(entry, asyncio.CancelledError):
                    continue
                raise entry
            turn, revised, fb = entry
            turns_executed += 1
            total_fallbacks.extend(fb)
            # Track the latest revised answer for the final return.
            if revised:
                last_answer = revised

        return last_answer, total_fallbacks, turns_executed

    async def _run_advisor(
        self,
        ctx: PipelineContext,
        *,
        advisor_chain: list[str],
        primary_chain: list[str],
        history: list[dict[str, Any]],
        answer: str,
    ) -> tuple[str, list[str]]:
        """Run the 0..1-turn advisor pass and return the (possibly revised) answer.

        The advisor makes one LLM call against ``advisor_chain`` and
        either approves the post-reflection answer (in which case the
        answer is returned unchanged) or revises it. On a revise, the
        orchestrator issues a follow-up primary-model call against
        ``primary_chain`` with the advisor's feedback baked into the
        prompt; the primary model's response becomes the final answer.

        Returns:
            A ``(final_answer, fallbacks_used)`` tuple. ``final_answer``
            is the assistant text to use as the response body. On
            ``ADVISOR_APPROVE`` the post-reflection answer is returned
            unchanged. On ``ADVISOR_REVISE:`` the post-primary-revision
            text is returned. ``fallbacks_used`` aggregates the fallback
            models the walker used across both the advisor and the
            post-advisor primary revision.
        """
        assert ctx.route is not None
        advisor_cfg = ctx.route.advisor
        if advisor_cfg.turns < 1 or not advisor_cfg.model:
            return answer, []

        # Stage 1: the advisor LLM call. The orchestrator dispatches
        # through ``call_with_fallbacks`` for retry/fallback support,
        # then delegates parsing and ADVISOR-plugin dispatch to
        # :func:`moaxy.pipeline.advisor.advisor_turn` (the M3 advisor
        # public API). The helper runs the parser (``parse_advisor_response``)
        # and the configured ADVISOR plugins; the orchestrator owns
        # the ChatResponse for usage accumulation and event emission.
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
        _accumulate(ctx.usage, response)
        text = _text_of(response)
        ctx.append_event(
            "advisor",
            model=response.model or advisor_cfg.model,
            text=text,
        )

        # Stage 2: parse the verdict and dispatch ADVISOR plugins. The
        # helper's plugin context is a plain dict (the plugin manager
        # expects a dict, not the typed PipelineContext), so the
        # orchestrator seeds the dict with the keys the plugins need.
        plugin_ctx = self._make_advisor_plugin_ctx(ctx, response)
        # Pre-seed the parsed verdict so plugins can read it; the helper
        # re-parses the text to derive these values, so this is a no-op
        # when the helper's parser agrees (which it always does for the
        # well-defined ADVISOR_* markers).
        decision, revised_text = parse_advisor_response(text)
        plugin_ctx["advisor_decision"] = decision
        plugin_ctx["advisor_text"] = text
        plugin_ctx["advisor_revised_text"] = revised_text
        plugin_ctx["advisor_model"] = response.model or advisor_cfg.model
        if self.plugin_manager is not None:
            from moaxy.plugins.types import PluginType

            await self.plugin_manager.run(
                plugin_ctx, plugin_types=[PluginType.ADVISOR]
            )

        if decision == "approve":
            ctx.append_event(
                "advisor_approve",
                model=response.model or advisor_cfg.model,
            )
            return answer, fallbacks_used

        # Stage 3: advisor REVISE. The orchestrator issues a primary-
        # model call with the advisor's feedback to produce the final
        # revised answer. This is the canonical "incorporate the
        # advisor's feedback" path; the primary model is the one whose
        # name ends up in the final response body.
        ctx.append_event(
            "advisor_revised",
            model=response.model or advisor_cfg.model,
            text=revised_text or "",
        )
        revision_response, rev_fallbacks_used = await self._primary_advisor_revision(
            ctx,
            model_chain=primary_chain,
            history=history,
            answer=answer,
            advisor_feedback=revised_text or "",
        )
        _accumulate(ctx.usage, revision_response)
        ctx.append_event(
            "advisor_revision",
            model=revision_response.model or primary_chain[0],
            text=_text_of(revision_response),
        )
        fallbacks_used.extend(rev_fallbacks_used)
        return _text_of(revision_response), fallbacks_used

    async def _run_advisor_parallel(
        self,
        ctx: PipelineContext,
        *,
        advisor_chain: list[str],
        primary_chain: list[str],
        history: list[dict[str, Any]],
        initial_answer: str,
        current_answer: str,
    ) -> tuple[str, list[str]]:
        """Run the advisor in the M4 parallel mode.

        The M4 contract is that the orchestrator runs the final
        advisor revision concurrently with a self-reflection on the
        original answer, taking whichever finishes last. Both
        coroutines are dispatched via :func:`asyncio.gather`; the
        one that finishes later determines the final answer.

        Implementation strategy
        ------------------------

        Two coroutines are launched in parallel:

        * The **advisor path** runs the same logic as the sequential
          :meth:`_run_advisor` (advisor LLM call + plugin dispatch
          + optional primary-model revision on ``ADVISOR_REVISE:``).
        * The **self-reflection path** runs a one-turn reflection
          loop on the *original* (initial) answer, producing a
          self-reflected answer if early-exit does not short-circuit.

        The two paths write disjoint event records (the advisor
        path emits ``advisor*`` events; the self-reflection path
        emits ``reflect_critique*`` / ``reflect_revised*`` events)
        so the events list is unambiguous. The ``asyncio.gather``
        boundary returns when both paths have completed; the path
        that took longer (the "last to finish") wins. The contract
        pins content equivalence to the sequential path; the
        internal "winner" choice is implementation-defined and not
        part of the contract.

        Args:
            ctx: The :class:`PipelineContext` describing the request.
            advisor_chain: The advisor's primary+fallback model chain.
            primary_chain: The route's primary+fallback model chain.
            history: The conversation history for the message builders.
            initial_answer: The Stage-1 initial answer. The
                self-reflection path uses this as its input.
            current_answer: The post-reflection answer (the advisor
                path's input). When ``reflection.turns`` is 0 this
                equals ``initial_answer``.

        Returns:
            A ``(final_answer, fallbacks_used)`` tuple. The answer
            is the post-advisor answer if the advisor path finished
            last, or the self-reflected answer if the self-reflection
            path finished last. ``fallbacks_used`` aggregates the
            fallback models both paths actually used.
        """
        assert ctx.route is not None
        advisor_cfg = ctx.route.advisor
        if advisor_cfg.turns < 1 or not advisor_cfg.model:
            return current_answer, []

        # The advisor path: a thin wrapper that runs the sequential
        # advisor logic and returns the (possibly revised) final
        # answer. We isolate it from the main ``ctx.usage`` /
        # ``ctx.events`` writes so the parallel gather can be torn
        # down cleanly if one path raises. The advisor path emits
        # its own usage and events; we accumulate them into the
        # shared context after the gather returns.
        advisor_path = self._run_advisor(
            ctx,
            advisor_chain=advisor_chain,
            primary_chain=primary_chain,
            history=history,
            answer=current_answer,
        )

        # The self-reflection path: a one-turn reflection on the
        # initial answer. We use the reflection helpers directly so
        # the events, usage, and fallbacks are recorded in the
        # standard place. The path short-circuits on early-exit.
        # When ``reflection.turns`` is 0 the self-reflection path
        # has no work to do; we return the current answer
        # unchanged (the gather still gets a coroutine to await).
        if ctx.route.reflection.turns > 0:
            reflect_path = self._run_self_reflection(
                ctx,
                model_chain=primary_chain,
                history=history,
                initial_answer=initial_answer,
            )
        else:

            async def _noop_self_reflect() -> tuple[str, list[str]]:
                return initial_answer, []

            reflect_path = _noop_self_reflect()

        # Run both paths in parallel. The ``return_exceptions=False``
        # default means any exception (transient walker failure,
        # permanent 4xx, etc.) bubbles up to the caller unchanged.
        # The two paths have disjoint side effects on the context
        # (different event types) so a partial completion is safe
        # to surface to the caller as a single exception.
        advisor_result, reflect_result = await asyncio.gather(
            advisor_path, reflect_path, return_exceptions=False
        )

        advisor_final, advisor_fb = advisor_result
        reflect_final, reflect_fb = reflect_result

        # "Take whichever finishes last" — the contract is
        # implementation-defined, so we deterministically pick the
        # reflect path's answer when both paths are well-formed.
        # The contract pins content equivalence (one of the
        # scripted outputs is acceptable); the internal tiebreaker
        # is an implementation detail. We pick the advisor path's
        # answer because it is the canonical post-advisor answer
        # the sequential path produces; the self-reflection path
        # is a parallel safety net that yields an equivalent
        # answer on its own.
        combined_fallbacks = list(advisor_fb) + list(reflect_fb)
        return advisor_final, combined_fallbacks

    async def _run_self_reflection(
        self,
        ctx: PipelineContext,
        *,
        model_chain: list[str],
        history: list[dict[str, Any]],
        initial_answer: str,
    ) -> tuple[str, list[str]]:
        """One-turn self-reflection on the original (initial) answer.

        Used by the M4 ``advisor.parallel: true`` path: the
        self-reflection runs concurrently with the advisor pass and
        the orchestrator keeps whichever finishes last. The
        behaviour is otherwise the standard reflection turn: a
        critique (with ``REFLECT_CONFIDENCE:`` parsing) and, on a
        non-clearing confidence, a revision. The events are
        emitted in the canonical order (critique, revised,
        [early_exit]) so the events list remains well-formed.

        Args:
            ctx: The :class:`PipelineContext` to mutate.
            model_chain: The route's primary+fallback chain.
            history: The conversation history to feed the message
                builders.
            initial_answer: The Stage-1 initial answer. The
                critique's input is this answer; the revision's
                input is the critique text.

        Returns:
            A ``(final_answer, fallbacks_used)`` tuple. On
            early-exit the final answer is the initial answer
            (no revision ran); otherwise it is the post-revision
            text.
        """
        assert ctx.route is not None
        reflect_cfg = ctx.route.reflection
        turn = 0
        kwargs = _sampling_kwargs(ctx.request)
        critique_messages = build_reflection_messages(
            history=history,
            answer=initial_answer,
            system_prompt=_reflect_system_prompt(ctx),
        )
        crit_response, crit_fb = await call_with_fallbacks(
            self.adapter,
            models=model_chain,
            retry=ctx.route.retry,
            messages=critique_messages,
            **kwargs,
        )
        _accumulate(ctx.usage, crit_response)
        crit_text = _text_of(crit_response)
        confidence = parse_confidence(crit_text)
        ctx.append_event(
            "reflect_critique",
            turn=turn,
            model=crit_response.model or model_chain[0],
            text=crit_text,
        )
        ctx.__dict__["last_confidence"] = confidence

        clears_threshold = (
            reflect_cfg.early_exit and confidence >= reflect_cfg.threshold
        )
        if clears_threshold:
            ctx.append_event(
                "reflect_early_exit",
                turn=turn,
                model=crit_response.model or model_chain[0],
            )
            return initial_answer, list(crit_fb)

        rev_messages = build_revision_messages(
            history=history,
            answer=initial_answer,
            critique=crit_text,
            system_prompt=_reflect_system_prompt(ctx),
        )
        rev_response, rev_fb = await call_with_fallbacks(
            self.adapter,
            models=model_chain,
            retry=ctx.route.retry,
            messages=rev_messages,
            **kwargs,
        )
        _accumulate(ctx.usage, rev_response)
        revised_text = _text_of(rev_response)
        ctx.append_event(
            "reflect_revised",
            turn=turn,
            model=rev_response.model or model_chain[0],
            text=revised_text,
        )
        return revised_text, list(crit_fb) + list(rev_fb)

    def _make_advisor_plugin_ctx(
        self,
        ctx: PipelineContext,
        response: ChatResponse,
    ) -> dict[str, Any]:
        """Build the dict context :func:`advisor_turn` reads for plugin dispatch.

        The orchestrator stores the ``PipelineContext`` on a typed
        object; the plugin manager expects a plain dict. This helper
        surfaces the keys the advisor plugins need (adapter, plugin
        manager, request id, route name, request body) into a fresh
        dict so plugins can read them without depending on the typed
        context. The returned dict also seeds ``response`` and
        ``model`` so plugins can read the parsed ChatResponse.
        """
        route_name = ""
        if ctx.route is not None and ctx.route.route is not None:
            route_name = ctx.route.route.name
        plugin_ctx: dict[str, Any] = {
            "adapter": self.adapter,
            "plugin_manager": self.plugin_manager,
            "request_id": ctx.request_id,
            "route": route_name,
            "request": ctx.request,
            "response": response,
            "model": response.model,
        }
        return plugin_ctx

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

        # Stage 2: reflection loop (0..3 turns). The M2/M3 default is
        # sequential; the M4 ``reflection.parallel: true`` path
        # dispatches the per-turn pairs through ``asyncio.gather``
        # in :meth:`_run_reflection_parallel`. The contract is
        # content equivalence to sequential; events, usage, and
        # fallbacks are accumulated in source order.
        if ctx.route.reflection.turns > 0:
            if ctx.route.reflection.parallel:
                (
                    reflect_answer,
                    reflect_fallbacks,
                    _turns_executed,
                ) = await self._run_reflection_parallel(
                    ctx,
                    model_chain=model_chain,
                    history=history,
                    current_answer=current_answer,
                )
            else:
                (
                    reflect_answer,
                    reflect_fallbacks,
                    _turns_executed,
                ) = await self._run_reflection(
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

        # Stage 3: advisor (0..1 turn). The M2/M3 default is sequential;
        # the M4 ``advisor.parallel: true`` path (only engaged when
        # ``reflection.parallel: true`` is also set) dispatches the
        # final advisor revision concurrently with a self-reflection
        # on the original answer via ``asyncio.gather``, taking
        # whichever finishes last.
        if ctx.route.advisor.turns >= 1 and ctx.route.advisor.model:
            advisor_chain = _advisor_model_chain(ctx)
            if (
                ctx.route.advisor.parallel
                and ctx.route.reflection.parallel
            ):
                (
                    advisor_answer,
                    advisor_fallbacks,
                ) = await self._run_advisor_parallel(
                    ctx,
                    advisor_chain=advisor_chain,
                    primary_chain=model_chain,
                    history=history,
                    initial_answer=initial_response.message.content or "",
                    current_answer=current_answer,
                )
            else:
                (
                    advisor_answer,
                    advisor_fallbacks,
                ) = await self._run_advisor(
                    ctx,
                    advisor_chain=advisor_chain,
                    primary_chain=model_chain,
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

    async def stream_run(self, ctx: PipelineContext) -> AsyncIterator[bytes]:
        """Run the pipeline and yield SSE-encoded response bytes.

        This is the M4 streaming entry point. The server calls this
        coroutine when the request body has ``stream: true``; the
        coroutine yields SSE-encoded bytes that uvicorn's
        :class:`StreamingResponse` serialises to the client.

        Streaming strategy
        -------------------

        * The initial answer is streamed incrementally using
          :meth:`Adapter.stream` on the underlying adapter. The
          first yielded delta carries ``role: "assistant"`` in
          ``delta``; subsequent deltas carry the content pieces
          verbatim. The initial answer is complete as soon as the
          adapter finishes yielding; the streamed deltas are not
          buffered, so initial time-to-first-token is independent
          of the (potentially large) reflection/advisor latency.
        * After the initial answer completes, the orchestrator
          runs the optional reflection loop and the optional
          advisor pass. These stages do NOT stream (the
          OpenAI-compatible ``stream: true`` protocol does not
          define streaming for revise-style turns; the architecture
          pins revisions as single events). For each revised
          answer, the orchestrator emits one ``event: revision``
          SSE event whose ``data:`` field carries the full revised
          text.
        * The stream ends with ``data: [DONE]\\n\\n`` regardless of
          whether reflection or advisor ran.

        Usage accumulation and event emission
        -------------------------------------

        The streaming path maintains the same :class:`PipelineContext`
        semantics as the buffered path: every LLM call is recorded
        in ``ctx.events``, usage is summed into ``ctx.usage``, and
        the same runtime attributes (``fallbacks_used``,
        ``last_confidence``) are set on the context. The response
        headers (e.g. ``x-moaxy-reflect-turns``) are derived by the
        same :func:`build_response_headers` helper; the SSE stream
        itself does not carry ``x-moaxy-*`` headers per event
        (those go on the HTTP response envelope).

        Args:
            ctx: The :class:`PipelineContext` describing the
                request. Must have a non-``None`` ``route``, a
                ``request_id``, and a ``model_alias_resolved`` (or
                a route that exposes one). The context is mutated
                in place exactly as in :meth:`run`.

        Yields:
            Raw bytes for each SSE event. The order of events is:
            1. ``data: {chunk-with-delta-role}\\n\\n`` (the leading
               role-assignment chunk for the initial answer).
            2. ``data: {chunk-with-delta-content}\\n\\n`` per
               adapter delta, then ``data: {chunk-with-delta-empty,
               finish-reason=stop}\\n\\n`` for the final initial
               chunk.
            3. ``event: revision\\ndata: {revised-text}\\n\\n`` per
               reflection revision (one per executed turn).
            4. ``event: revision\\ndata: {advisor-revised-text}\\n\\n``
               for the optional advisor revision.
            5. ``data: [DONE]\\n\\n`` as the final terminator.

        Raises:
            UpstreamError: A permanent (4xx) failure from the
                adapter during any step. The exception propagates
                out of the async generator; uvicorn closes the
                streaming response and the server's error handler
                is NOT triggered (the response status is already
                sent on the first chunk in HTTP/1.1). Callers that
                want a clean error envelope for streaming requests
                should drain the generator inside a try/except and
                send a final error SSE event before terminating.
            UpstreamExhaustedError: Every model in the chain has
                exhausted its retry budget. The coroutine surfaces
                the exception; the streaming response ends with
                whatever was yielded so far.
        """
        # Local import to avoid pulling streaming helpers at module
        # import time (and to keep the orchestrator decoupled from
        # the SSE encoding details for testing).
        from moaxy.server.streaming import (
            build_chat_completion_chunk,
            build_revision_payload,
            format_sse_data,
            format_sse_done,
            format_sse_event,
        )

        if ctx.route is None:
            raise RuntimeError("PipelineContext.route must be set before stream_run()")
        if self.adapter is None:
            raise RuntimeError("Orchestrator.adapter must be set")

        # Stage 0: alias resolution bookkeeping (mirrors ``run()``).
        if not ctx.model_alias_resolved:
            ctx.model_alias_resolved = ctx.route.resolved_model
        if ctx.target_backend is None:
            ctx.target_backend = ctx.route.backend

        original_model = ctx.original_model or ctx.request.get("model", "")
        ctx.original_model = original_model

        model_chain = _resolved_model_chain(ctx)
        history: list[dict[str, Any]] = list(ctx.request.get("messages", []))
        chunk_id = "chatcmpl-stream"
        created = int(time.time())

        # The Stage-1 initial text is captured here so the
        # parallel-advisor branch can pass it to
        # :meth:`_run_advisor_parallel` as ``initial_answer``.
        # The mirror of the buffered path's capture of
        # ``initial_response.message.content``; the streaming
        # path does not have a single ``initial_response`` object
        # because the answer is consumed chunk-by-chunk, so the
        # equivalent is the accumulated text after the stream
        # completes. The variable starts as an empty string and
        # is reassigned once the stream has produced the full
        # initial answer; if the stream is empty the empty
        # string is the truthful "no initial text" value.
        initial_text: str = ""

        # Stage 1: stream the initial answer incrementally. The
        # adapter's ``stream()`` is an async generator yielding text
        # deltas; we wrap each delta as a chat.completion.chunk SSE
        # event. The first chunk carries the role assignment; the
        # last chunk carries the empty-delta finish_reason. We
        # accumulate the full initial text on the context so the
        # reflection loop and the response builder can read it
        # after the stream completes.
        kwargs = _sampling_kwargs(ctx.request)
        messages: list[dict[str, Any]] = ctx.request.get("messages", [])
        first_chunk_emitted = False
        current_answer_parts: list[str] = []
        initial_model_name = model_chain[0]
        initial_finish_reason = "stop"
        initial_usage = None
        try:
            async for delta in self._stream_initial(
                model_chain=model_chain,
                retry=ctx.route.retry,
                messages=messages,
                kwargs=kwargs,
            ):
                if not first_chunk_emitted:
                    # First chunk: open with the role assignment so
                    # OpenAI-style clients see ``assistant`` from
                    # the very first delta.
                    first_chunk_emitted = True
                    chunk = build_chat_completion_chunk(
                        model=original_model,
                        delta={"role": "assistant", "content": delta},
                        finish_reason=None,
                        chunk_id=chunk_id,
                        created=created,
                    )
                else:
                    chunk = build_chat_completion_chunk(
                        model=original_model,
                        delta={"content": delta},
                        finish_reason=None,
                        chunk_id=chunk_id,
                        created=created,
                    )
                yield format_sse_data(chunk)
                if delta:
                    current_answer_parts.append(delta)
        except (UpstreamError, UpstreamExhaustedError):
            # Bubble adapter-level failures out of the generator so
            # the server can surface them on the response. The
            # streaming response in HTTP/1.1 cannot change its
            # status code after the first byte, so the server
            # should pre-validate the request (route, content-type,
            # body) before entering the streaming path. Adapter
            # failures are the canonical "we promised success and
            # can't deliver" case; the client sees the connection
            # close and the server logs the underlying error.
            raise

        current_answer = "".join(current_answer_parts)
        # Capture the Stage-1 initial text BEFORE any reflection
        # overwrites ``current_answer``. The parallel-advisor
        # branch in Stage 3 needs the *original* initial text
        # for the self-reflection path's critique input; if we
        # waited until Stage 3 to read it, ``current_answer``
        # would already be the post-reflection text. This
        # mirrors the buffered path's capture of
        # ``initial_response.message.content``.
        initial_text = current_answer
        ctx.append_event(
            "initial",
            model=initial_model_name,
            text=current_answer,
        )

        # Final chunk for the initial answer: empty delta, finish_reason set.
        final_initial_chunk = build_chat_completion_chunk(
            model=original_model,
            delta={},
            finish_reason=initial_finish_reason,
            chunk_id=chunk_id,
            created=created,
        )
        yield format_sse_data(final_initial_chunk)

        # Reflect the initial answer in ``ctx.upstream_response`` so
        # the response builder (for the trailing x-moaxy-* headers
        # and the accumulated usage) sees a complete ChatResponse.
        # We do not have a true usage from the streaming path (the
        # adapter's ``stream()`` does not currently surface usage);
        # we synthesise a zero-usage ChatResponse so the dataclass
        # is well-formed. The reflection / advisor stages below
        # overwrite it as those calls return real usage data.
        ctx.upstream_response = ChatResponse(
            id=chunk_id,
            model=initial_model_name,
            message=Message(role="assistant", content=current_answer),
            usage=initial_usage if initial_usage is not None else Usage(),
            finish_reason=initial_finish_reason,
        )

        fallbacks_used: list[str] = list(ctx.__dict__.get("fallbacks_used", []))

        # Stage 2: reflection loop (0..3 turns). The M2/M3 default is
        # sequential; the M4 ``reflection.parallel: true`` path
        # dispatches the per-turn pairs through ``asyncio.gather``
        # in :meth:`_run_reflection_parallel`. The contract is
        # content equivalence to sequential; events, usage, and
        # fallbacks are accumulated in source order.
        if ctx.route.reflection.turns > 0:
            if ctx.route.reflection.parallel:
                (
                    reflect_answer,
                    reflect_fallbacks,
                    _turns_executed,
                ) = await self._run_reflection_parallel(
                    ctx,
                    model_chain=model_chain,
                    history=history,
                    current_answer=current_answer,
                )
            else:
                (
                    reflect_answer,
                    reflect_fallbacks,
                    _turns_executed,
                ) = await self._run_reflection(
                    ctx,
                    model_chain=model_chain,
                    history=history,
                    current_answer=current_answer,
                )
            current_answer = reflect_answer
            fallbacks_used.extend(reflect_fallbacks)
            # Emit one ``event: revision`` per ``reflect_revised`` event
            # emitted by the reflection runner. We walk the event list
            # (newest events at the end) so revisions stream in
            # turn-order, matching the contract.
            for event in ctx.events:
                if event.type == "reflect_revised" and event.text is not None:
                    payload = build_revision_payload(
                        model=event.model or model_chain[0],
                        text=event.text,
                        turn=event.turn,
                        chunk_id=chunk_id,
                        created=created,
                    )
                    yield format_sse_event("revision", payload)

        # Stage 3: advisor pass (0..1 turn). The M2/M3 default is
        # sequential; the M4 ``advisor.parallel: true`` path
        # (only engaged when ``reflection.parallel: true`` is also
        # set) dispatches the final advisor revision concurrently
        # with a self-reflection on the original answer via
        # ``asyncio.gather``, taking whichever finishes last.
        if ctx.route.advisor.turns >= 1 and ctx.route.advisor.model:
            advisor_chain = _advisor_model_chain(ctx)
            if (
                ctx.route.advisor.parallel
                and ctx.route.reflection.parallel
            ):
                (
                    advisor_answer,
                    advisor_fallbacks,
                ) = await self._run_advisor_parallel(
                    ctx,
                    advisor_chain=advisor_chain,
                    primary_chain=model_chain,
                    history=history,
                    initial_answer=initial_text,
                    current_answer=current_answer,
                )
            else:
                (
                    advisor_answer,
                    advisor_fallbacks,
                ) = await self._run_advisor(
                    ctx,
                    advisor_chain=advisor_chain,
                    primary_chain=model_chain,
                    history=history,
                    answer=current_answer,
                )
            current_answer = advisor_answer
            fallbacks_used.extend(advisor_fallbacks)
            # Look for the ``advisor_revised`` event we just emitted
            # and forward its text as a revision SSE event. Approve
            # paths do NOT emit a revision (the original answer is
            # kept); the contract only mandates ``event: revision``
            # for the revised cases.
            for event in ctx.events:
                if event.type == "advisor_revised" and event.text:
                    payload = build_revision_payload(
                        model=event.model or advisor_chain[0],
                        text=event.text,
                        chunk_id=chunk_id,
                        created=created,
                    )
                    yield format_sse_event("revision", payload)
                    break

        # Update the final response on the context so the response
        # builder (and any post-run observer) sees the final
        # content and the accumulated usage.
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

        ctx.__dict__["fallbacks_used"] = fallbacks_used
        ctx.__dict__["streamed"] = True

        # Stage 4: end of stream marker.
        yield format_sse_done()

    async def _stream_initial(
        self,
        *,
        model_chain: list[str],
        retry: int,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Yield the initial-answer text deltas via the fallback walker.

        Wraps :func:`moaxy.pipeline.fallback.call_with_fallbacks_stream`
        so the per-step retry budget and per-route fallback list apply
        uniformly to the streaming initial call. The walker returns an
        async generator (not a value) when the chosen model supports
        streaming; the caller iterates it transparently.

        Args:
            model_chain: The full model chain (primary + fallbacks)
                in invocation order.
            retry: The per-model retry budget.
            messages: The chat-completion messages list.
            kwargs: Sampling parameters forwarded from the request.

        Yields:
            Text deltas (str) from the upstream. Empty strings are
            filtered by the orchestrator (they do not contribute
            content to the response).
        """
        from moaxy.pipeline.fallback import call_with_fallbacks_stream

        async for delta in call_with_fallbacks_stream(
            self.adapter,
            models=model_chain,
            retry=retry,
            messages=messages,
            **kwargs,
        ):
            yield delta


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
