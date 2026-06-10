"""``POST /v1/chat/completions`` — OpenAI-compatible proxy.

The handler is the data-plane surface of moaxy. It accepts an
OpenAI-shaped ``chat.completion`` request body, matches the request
against the route table, resolves aliases, and forwards the call to the
configured backend adapter.

The full pipeline (reflection turns, advisor, usage accumulation, etc.)
will be added in M2/M3. M1 implements:

* Request validation (presence of ``model`` and ``messages``).
* Route matching via :class:`moaxy.routing.matcher.RouteMatcher`.
* Alias resolution against the matched route.
* Adapter dispatch through the route's ``backend``.
* Per-call ``UsageAccumulator`` for usage aggregation across multiple
  adapter calls (e.g. fallbacks).
* 404 when no route matches.
* 502 when all backends are exhausted.
* 415 when the request body is missing the ``application/json`` content
  type.
* 400/422 for malformed JSON or missing required fields, with the
  envelope produced by the error handlers.
* Echoing the original ``model`` in the response body (alias or real).
* Setting ``x-moaxy-*`` response headers (``x-moaxy-alias-resolved``,
  ``x-moaxy-fallbacks-used``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from moaxy.adapters.base import (
    Adapter,
    UpstreamError,
    UsageAccumulator,
)
from moaxy.adapters.base import (
    UpstreamTimeoutError as AdapterUpstreamTimeoutError,
)
from moaxy.adapters.base import (
    UpstreamUnavailableError as AdapterUpstreamUnavailableError,
)
from moaxy.routing.matcher import RouteMatch, RouteMatcher
from moaxy.server.errors import (
    BadRequestError,
    NoRouteMatchError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
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


def _model_name_used(match: RouteMatch) -> str:
    """The model name that the upstream adapter should call.

    For M1 the matcher is responsible for alias resolution; we use the
    resolved model as the model parameter sent to the adapter.
    """
    return match.resolved_model


def _get_adapter(request: Request, match: RouteMatch) -> Adapter:
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


async def _walk_fallbacks(
    adapter: Adapter,
    *,
    models: list[str],
    messages: list[dict[str, Any]],
    body: dict[str, Any],
) -> tuple[dict[str, Any], int, list[str], UsageAccumulator]:
    """Try the resolved model, then each entry of ``models``, returning the
    final response dict, the number of fallback hops used, the list of
    fallback model ids that were actually invoked, and the running
    :class:`UsageAccumulator`.
    """
    usage = UsageAccumulator()
    fallbacks_used: list[str] = []
    last_error: Exception | None = None
    last_status: int | None = None
    last_body: str | None = None

    for model in models:
        try:
            response = await _do_chat(adapter, model=model, body=body, messages=messages)
        except AdapterUpstreamTimeoutError as exc:
            logger.warning("upstream timeout on model=%s: %s", model, exc)
            last_error = exc
            last_status = None
            last_body = None
            if model != models[0]:
                fallbacks_used.append(model)
            continue
        except AdapterUpstreamUnavailableError as exc:
            logger.warning("upstream unavailable on model=%s: %s", model, exc)
            last_error = exc
            last_status = None
            last_body = None
            if model != models[0]:
                fallbacks_used.append(model)
            continue
        except UpstreamError as exc:
            logger.warning("upstream error on model=%s: %s", model, exc)
            last_error = exc
            last_status = exc.status_code
            last_body = exc.body
            if model != models[0]:
                fallbacks_used.append(model)
            continue
        if model != models[0]:
            fallbacks_used.append(model)
        return response, len(fallbacks_used), fallbacks_used, usage

    # Re-raise the most informative error. The ordering is:
    # 1. A 4xx upstream means the client request is wrong → 400.
    # 2. A timeout → 504.
    # 3. An unavailable upstream → 503.
    # 4. A 5xx upstream → 502 (carries the upstream's message).
    # 5. Otherwise (e.g. all attempts unreachable) → 502 with summary.
    if last_status is not None and 400 <= last_status < 500:
        raise BadRequestError(
            f"upstream rejected the request (HTTP {last_status}): {last_error}",
            details={
                "status": last_status,
                "models": models,
                "response_body": last_body,
            },
        )
    if isinstance(last_error, AdapterUpstreamTimeoutError):
        raise UpstreamTimeoutHTTPError(
            str(last_error) or "upstream timeout",
            details={"models": models},
        )
    if isinstance(last_error, AdapterUpstreamUnavailableError):
        raise UpstreamUnavailableHTTPError(
            str(last_error) or "upstream unavailable",
            details={"models": models},
        )
    if isinstance(last_error, UpstreamError):
        raise UpstreamHTTPError(
            str(last_error),
            status_code=last_status,
            body=last_body,
            details={"models": models},
        )
    raise LegacyUpstreamUnavailableHTTPError(
        "all backends failed; no model in the fallback chain succeeded",
        details={
            "models": models,
            "last_error": str(last_error) if last_error else None,
        },
    )


async def _do_chat(
    adapter: Adapter,
    *,
    model: str,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Invoke the adapter and return the OpenAI-shaped response dict.

    Accepts the original request body so that the orchestrator can pass
    extra sampling parameters (temperature, top_p, max_tokens, etc.) in
    later milestones.
    """
    kwargs: dict[str, Any] = {
        k: v
        for k, v in body.items()
        if k not in {"model", "messages", "stream"}
    }
    response = await adapter.chat(model=model, messages=messages, **kwargs)
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": response.message.role, "content": response.message.content},
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """OpenAI-compatible chat completions endpoint."""
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
    models: list[str] = [_model_name_used(match), *match.fallbacks]
    messages: list[dict[str, Any]] = body["messages"]

    response_dict, _hops, fallbacks_used, _usage = await _walk_fallbacks(
        adapter, models=models, messages=messages, body=body
    )

    response_dict["model"] = match.original_model
    headers: dict[str, str] = {
        "x-moaxy-alias-resolved": match.resolved_model,
        "x-moaxy-fallbacks-used": str(len(fallbacks_used)),
    }
    return JSONResponse(content=response_dict, headers=headers, media_type="application/json")


__all__ = ["router"]
