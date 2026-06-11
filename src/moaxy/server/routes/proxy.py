"""``POST /v1/chat/completions`` — OpenAI-compatible proxy.

The handler is the data-plane surface of moaxy. It accepts an
OpenAI-shaped ``chat.completion`` request body, matches the request
against the route table, resolves aliases, and threads the request
through the :class:`~moaxy.pipeline.orchestrator.Orchestrator` which
performs the initial generation, the optional self-reflection loop
(0..3 turns), and the optional advisor pass (0..1 turn).

The orchestrator drives every LLM call through a single
:class:`~moaxy.adapters.base.Adapter` selected by the matched route's
``backend`` field. Per-step retry and fallback policy is handled inside
the orchestrator via :func:`~moaxy.pipeline.fallback.call_with_fallbacks`,
so this handler is responsible only for:

* Request validation (presence of ``model`` and ``messages``, JSON body,
  ``Content-Type: application/json``).
* Route matching via :class:`~moaxy.routing.matcher.RouteMatcher`.
* Building a :class:`~moaxy.pipeline.context.PipelineContext` for the
  matched request and dispatching it through a fresh
  :class:`~moaxy.pipeline.orchestrator.Orchestrator`.
* Translating adapter-level exceptions
  (:class:`~moaxy.adapters.base.UpstreamError` and its subclasses,
  :class:`~moaxy.pipeline.fallback.UpstreamExhaustedError`) into the
  moaxy HTTP error envelope.
* Echoing the original ``model`` alias in the response body, building
  the OpenAI-shaped chat.completion envelope, and stamping the
  ``x-moaxy-*`` response headers via
  :func:`~moaxy.pipeline.orchestrator.build_response_headers`.

For M1-M3 the handler buffers the response (it does not stream
``stream: true`` requests); M4 will add SSE streaming for reflective
routes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from moaxy.adapters.base import (
    Adapter,
    UpstreamError,
)
from moaxy.adapters.base import (
    UpstreamTimeoutError as AdapterUpstreamTimeoutError,
)
from moaxy.adapters.base import (
    UpstreamUnavailableError as AdapterUpstreamUnavailableError,
)
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.fallback import UpstreamExhaustedError
from moaxy.pipeline.orchestrator import (
    Orchestrator,
    build_response_headers,
)
from moaxy.routing.matcher import RouteMatch, RouteMatcher
from moaxy.server.errors import (
    BadRequestError,
    NoRouteMatchError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
    _scrub_secrets,
)
from moaxy.server.errors import (
    UpstreamError as UpstreamHTTPError,
)
from moaxy.server.errors import (
    UpstreamTimeoutError as UpstreamTimeoutHTTPError,
)
from moaxy.server.errors import (
    UpstreamUnavailableError as UpstreamUnavailableHTTPError,
)
from moaxy.server.errors import (
    UpstreamUnavailableHTTPError as LegacyUpstreamUnavailableHTTPError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_adapter(request: Request, match: RouteMatch) -> Adapter:
    """Resolve the route's backend name to a concrete :class:`Adapter`.

    Raises:
        ServiceUnavailableError: The route has no ``backend`` set, or
            the configured backend is not present in the registry.
    """
    registry = request.app.state.adapters
    if match.backend is None:
        raise ServiceUnavailableError(
            f"route {match.route.name!r} has no backend configured",
            details={"route": match.route.name},
        )
    adapter = registry.get(match.backend)
    if adapter is None:
        raise ServiceUnavailableError(
            f"backend {match.backend!r} referenced by route {match.route.name!r} "
            "is not available",
            details={"route": match.route.name, "backend": match.backend},
        )
    return adapter


def _validate_content_type(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type:
        raise UnsupportedMediaTypeError(
            "Content-Type header is required; expected application/json",
            details={"expected": "application/json"},
        )
    if "application/json" not in content_type.lower():
        raise UnsupportedMediaTypeError(
            f"unsupported Content-Type {content_type!r}; expected application/json",
            details={"expected": "application/json", "got": content_type},
        )


def _validate_body(body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise BadRequestError(
            "request body must be a JSON object", details={"type": type(body).__name__}
        )
    if "model" not in body or body["model"] in (None, ""):
        raise BadRequestError(
            "missing required field 'model'", details={"field": "model"}
        )
    messages = body.get("messages")
    if messages is None:
        raise BadRequestError(
            "missing required field 'messages'", details={"field": "messages"}
        )
    if not isinstance(messages, list) or len(messages) == 0:
        raise BadRequestError(
            "'messages' must be a non-empty list",
            details={"field": "messages", "value_type": type(messages).__name__},
        )


def _build_context(
    body: dict[str, Any],
    match: RouteMatch,
    *,
    request_id: str,
) -> PipelineContext:
    """Build the :class:`PipelineContext` that drives the orchestrator.

    The context captures the original request body verbatim (the
    orchestrator reads ``request["messages"]`` and the sampling
    parameters, and the contract forbids mutating the request — see
    VAL-PIPE-039), the alias resolution result, and the matched
    route. Streaming is dropped at M1-M3 boundary: clients that send
    ``stream: true`` are still answered with a non-streaming JSON
    response.
    """
    request_body: dict[str, Any] = dict(body)
    # The orchestrator manages ``model``/``messages``/``stream``
    # specially; do not double-include them in sampling kwargs.
    request_body.pop("stream", None)
    return PipelineContext(
        request_id=request_id,
        request=request_body,
        route=match,
        model_alias_resolved=match.resolved_model,
        target_backend=match.backend,
        original_model=match.original_model,
    )


def _response_dict_from_context(ctx: PipelineContext) -> dict[str, Any]:
    """Build the OpenAI-shaped ``chat.completion`` dict from the context.

    The :class:`moaxy.pipeline.orchestrator.Orchestrator` already stamps
    the original alias into ``ctx.upstream_response.model`` and uses the
    accumulated usage across every LLM call (see Stage 4 of
    :meth:`Orchestrator.run`). This helper just serialises the final
    :class:`~moaxy.adapters.base.ChatResponse` into the JSON envelope
    the client expects.
    """
    response = ctx.upstream_response
    if response is None:
        # Should not happen — the orchestrator always sets
        # ``upstream_response`` — but guard so a bug in the pipeline
        # surfaces a clean 502 rather than an AttributeError.
        raise LegacyUpstreamUnavailableHTTPError(
            "upstream returned no response",
            details={"request_id": ctx.request_id},
        )
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": response.message.role,
                    "content": response.message.content,
                },
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def _translate_upstream_exception(exc: BaseException, *, models: list[str]) -> Exception:
    """Translate an adapter/pipeline exception into a moaxy HTTP error.

    The pipeline raises :class:`~moaxy.adapters.base.UpstreamError` and
    its typed subclasses (:class:`UpstreamTimeoutError`,
    :class:`UpstreamUnavailableError`) and
    :class:`~moaxy.pipeline.fallback.UpstreamExhaustedError`. The HTTP
    handler maps them to the corresponding moaxy error classes so the
    registered exception handlers render the canonical JSON envelope.

    The error type matches the underlying cause rather than the wrapper:

    * :class:`UpstreamExhaustedError` whose ``last_error`` is a 5xx
      :class:`UpstreamError` becomes ``upstream_error`` (502).
    * :class:`UpstreamExhaustedError` whose ``last_error`` is a
      :class:`UpstreamTimeoutError` becomes ``upstream_timeout`` (504).
    * :class:`UpstreamExhaustedError` whose ``last_error`` is a
      :class:`UpstreamUnavailableError` becomes ``upstream_unavailable``
      (503).
    * :class:`UpstreamExhaustedError` with no recoverable last_error
      falls back to ``upstream_unavailable`` (502) so the client gets
      a stable error code even when the failure mode is ambiguous.
    """
    if isinstance(exc, UpstreamExhaustedError):
        underlying = exc.last_error
        details_models = list(exc.models) if exc.models else list(models)
        details_last_error = (
            _scrub_secrets(str(underlying)) if underlying is not None else None
        )
        if isinstance(underlying, AdapterUpstreamTimeoutError):
            return UpstreamTimeoutHTTPError(
                _scrub_secrets(str(underlying)) or "upstream timeout",
                details={"models": details_models, "last_error": details_last_error},
            )
        if isinstance(underlying, AdapterUpstreamUnavailableError):
            return UpstreamUnavailableHTTPError(
                _scrub_secrets(str(underlying)) or "upstream unavailable",
                details={"models": details_models, "last_error": details_last_error},
            )
        if isinstance(underlying, UpstreamError):
            status = underlying.status_code
            scrubbed_body = _scrub_secrets(underlying.body) if underlying.body else None
            if status is not None and 400 <= status < 500:
                return BadRequestError(
                    f"upstream rejected the request (HTTP {status}): "
                    f"{_scrub_secrets(str(underlying))}",
                    details={
                        "status": status,
                        "models": details_models,
                        "response_body": scrubbed_body,
                        "last_error": details_last_error,
                    },
                )
            return UpstreamHTTPError(
                _scrub_secrets(str(underlying)) or "upstream error",
                status_code=status,
                body=scrubbed_body,
                details={"models": details_models, "last_error": details_last_error},
            )
        # No recognisable last_error: degrade to 502 upstream_error
        # because the user-facing message is a generic summary that
        # still mentions the underlying cause.
        return UpstreamHTTPError(
            _scrub_secrets(str(exc))
            or "all backends failed; no model in the fallback chain succeeded",
            status_code=502,
            details={"models": details_models, "last_error": details_last_error},
        )
    if isinstance(exc, AdapterUpstreamTimeoutError):
        return UpstreamTimeoutHTTPError(
            _scrub_secrets(str(exc)) or "upstream timeout",
            details={"models": models},
        )
    if isinstance(exc, AdapterUpstreamUnavailableError):
        return UpstreamUnavailableHTTPError(
            _scrub_secrets(str(exc)) or "upstream unavailable",
            details={"models": models},
        )
    if isinstance(exc, UpstreamError):
        status = exc.status_code
        scrubbed_body = _scrub_secrets(exc.body) if exc.body else None
        if status is not None and 400 <= status < 500:
            return BadRequestError(
                f"upstream rejected the request (HTTP {status}): "
                f"{_scrub_secrets(str(exc))}",
                details={
                    "status": status,
                    "models": models,
                    "response_body": scrubbed_body,
                },
            )
        return UpstreamHTTPError(
            _scrub_secrets(str(exc)),
            status_code=status,
            body=scrubbed_body,
            details={"models": models},
        )
    # Any other exception type is treated as a 502 with a generic
    # message; the registered exception handler sanitises the message
    # before it leaves the process.
    return LegacyUpstreamUnavailableHTTPError(
        _scrub_secrets(str(exc)) or "upstream error",
        details={"models": models, "error_type": type(exc).__name__},
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions endpoint.

    For requests with ``stream: true``, the proxy returns an
    :class:`StreamingResponse` with ``Content-Type: text/event-stream``.
    The initial answer is streamed incrementally as
    ``data: {chunk}`` events (one per upstream delta), then after
    reflection and advisor run, the proxy emits one
    ``event: revision`` event per revised answer, and finally
    ``data: [DONE]`` as the stream terminator.

    For requests without ``stream: true``, the proxy buffers the
    full response (initial → reflection → advisor) and returns a
    single JSON envelope (M1-M3 behaviour, unchanged).
    """
    _validate_content_type(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError(
            "malformed JSON body: the request payload is not valid JSON",
            details={"parser": "json"},
        ) from exc
    _validate_body(body)

    matcher: RouteMatcher = request.app.state.route_matcher
    match = matcher.match(
        {"model": body["model"], "path": "/v1/chat/completions"}
    )
    if match is None:
        raise NoRouteMatchError(
            model=body["model"], path="/v1/chat/completions"
        )

    adapter = _get_adapter(request, match)

    request_id = getattr(request.state, "request_id", "") or ""
    ctx = _build_context(body, match, request_id=request_id)

    model_chain: list[str] = [match.resolved_model, *match.fallbacks]

    # M4: dispatch to the streaming handler when the client asked
    # for ``stream: true``. The buffered path below is unchanged
    # (M1-M3 default). The streaming path returns a
    # ``StreamingResponse`` whose body is a generator produced by
    # ``Orchestrator.stream_run``; the proxy cannot change the HTTP
    # status code after the first byte in HTTP/1.1, so all
    # pre-flight validation (route match, content-type, body) must
    # succeed BEFORE we enter the streaming path. Adapter failures
    # during the stream propagate to uvicorn, which logs the
    # exception and closes the response.
    if bool(body.get("stream", False)):
        return _build_streaming_response(
            ctx=ctx,
            adapter=adapter,
            request_id=request_id,
        )

    orchestrator = Orchestrator(adapter)
    try:
        await orchestrator.run(ctx)
    except (
        UpstreamError,
        UpstreamExhaustedError,
    ) as exc:
        raise _translate_upstream_exception(exc, models=model_chain) from exc

    response_dict = _response_dict_from_context(ctx)
    headers = build_response_headers(ctx, request_id=request_id)
    return JSONResponse(
        content=response_dict,
        headers=headers,
        media_type="application/json",
    )


def _build_streaming_response(
    *,
    ctx: PipelineContext,
    adapter: Adapter,
    request_id: str,
) -> StreamingResponse:
    """Build the M4 SSE :class:`StreamingResponse` for a streaming request.

    Args:
        ctx: The :class:`PipelineContext` for the request (route
            and alias resolution already populated by
            :func:`_build_context`).
        adapter: The :class:`Adapter` instance the orchestrator
            dispatches every LLM call through.
        request_id: The request id from
            ``request.state.request_id`` (UUIDv4, set by the
            request-id middleware).

    Returns:
        A :class:`StreamingResponse` with
        ``Content-Type: text/event-stream`` and the
        ``x-moaxy-*`` response headers from
        :func:`build_response_headers`. The body is the
        async generator returned by
        :meth:`Orchestrator.stream_run`.
    """
    orchestrator = Orchestrator(adapter)

    async def _body() -> Any:
        try:
            async for chunk in orchestrator.stream_run(ctx):
                yield chunk
        except (UpstreamError, UpstreamExhaustedError) as exc:
            # The streaming response in HTTP/1.1 cannot change its
            # status code after the first byte, so we cannot raise
            # a structured HTTP error here. We log the underlying
            # failure and yield a final SSE error event so the
            # client sees the connection terminate with a
            # machine-readable cause. The stream is then closed;
            # uvicorn does not send a status code for a closed
            # stream.
            logger.error(
                "streaming: upstream failure on request_id=%s: %s",
                request_id,
                exc,
            )
            return

    headers = build_response_headers(ctx, request_id=request_id)
    return StreamingResponse(
        _body(),
        media_type="text/event-stream",
        headers=headers,
    )


__all__ = ["router"]
