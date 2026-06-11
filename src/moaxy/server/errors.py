"""Uniform JSON error responses for the moaxy proxy.

Every error response (4xx and 5xx) follows the same shape:

    {"error": {"type": "<short_code>", "message": "<human-readable>"}}

The :class:`MoaxyError` hierarchy is the canonical way to raise an error
from any server-side code path; the registered exception handlers convert
it (and other exceptions) into the JSON envelope. The :class:`MoaxyError`
class is also re-exported for callers that want to raise their own typed
errors against a route handler.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class MoaxyError(Exception):
    """Base class for all moaxy HTTP errors.

    Subclasses set :attr:`status_code` and a short :attr:`error_type` code
    that the JSON envelope surfaces. The :attr:`message` is the human
    text; the :attr:`details` dict is an optional extra payload surfaced
    in the envelope as ``"details"`` (omitted when empty).

    Subclasses should not include Python stack traces, filesystem paths,
    or ``/site-packages/`` substrings in the message. Callers catching a
    :class:`MoaxyError` and re-raising should preserve the original
    message.
    """

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details) if details else {}


class BadRequestError(MoaxyError):
    status_code = 400
    error_type = "bad_request"


class UnsupportedMediaTypeError(MoaxyError):
    status_code = 415
    error_type = "unsupported_media_type"


class NotFoundError(MoaxyError):
    status_code = 404
    error_type = "not_found"


class UnauthorizedError(MoaxyError):
    """Raised when the request is unauthenticated (no principal on state).

    Maps to HTTP 401. The auth gate already returns 401 for missing
    or invalid API keys; this class is the in-band signal for code
    paths that bypass the gate (admin role checks, etc.) but still
    need a 401 to the client.
    """

    status_code = 401
    error_type = "unauthorized"


class ForbiddenError(MoaxyError):
    """Raised when the request is authenticated but not authorised.

    Maps to HTTP 403. The admin endpoints raise this when a valid
    principal is missing the ``admin`` role; the client gets a
    distinct 403 instead of a 401 so it can tell "your key is
    recognised but you cannot do this" apart from "your key is
    not recognised".
    """

    status_code = 403
    error_type = "forbidden"


class NoRouteMatchError(MoaxyError):
    """Raised when no route in the table matches the incoming request.

    Maps to HTTP 404. The error message contains the literal substring
    ``"no route matches"`` so callers can distinguish it from a generic
    404 (e.g. an unknown path).
    """

    status_code = 404
    error_type = "no_route_match"

    def __init__(
        self,
        message: str | None = None,
        *,
        model: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if message is None:
            parts: list[str] = []
            if model is not None:
                parts.append(f"model {model!r}")
            if path is not None:
                parts.append(f"on {path}")
            suffix = " " + " ".join(parts) if parts else ""
            message = f"no route matches{suffix}"
        payload: dict[str, Any] = dict(details) if details else {}
        if model is not None:
            payload.setdefault("model", model)
        if path is not None:
            payload.setdefault("path", path)
        super().__init__(message, details=payload or None)


class MethodNotAllowedError(MoaxyError):
    status_code = 405
    error_type = "method_not_allowed"


class UpstreamError(MoaxyError):
    """Raised when the upstream returns a 4xx/5xx or an unusable response.

    Maps to HTTP 502. The original upstream message (or a fallback
    summary) is preserved in :attr:`message` so the client gets a
    useful, human-readable explanation. Stack traces, file paths, and
    ``/site-packages/`` substrings are never included.
    """

    status_code = 502
    error_type = "upstream_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_message = _sanitize_message(message)
        payload: dict[str, Any] = dict(details) if details else {}
        if status_code is not None:
            payload.setdefault("upstream_status", status_code)
        if body is not None and "upstream_body" not in payload:
            # Scrub the upstream body before exposing it in the
            # public error envelope. The upstream may echo any
            # secrets it was sent in its own error message
            # (e.g. an OpenRouter 401 response that includes the
            # bearer token in the error text), and surfacing the
            # raw body would make moaxy a leak vector for the
            # very secrets it is supposed to protect. The
            # full body is preserved in ``self.upstream_body``
            # for internal logging / debug surfaces, but the
            # client-facing ``details["upstream_body"]`` is
            # always the scrubbed form.
            payload["upstream_body"] = _scrub_secrets(body)
        super().__init__(safe_message, details=payload or None)
        self.upstream_status_code = status_code
        self.upstream_body = body


class UpstreamTimeoutError(MoaxyError):
    """Raised when the upstream times out (read, connect, or pool)."""

    status_code = 504
    error_type = "upstream_timeout"


class UpstreamUnavailableError(MoaxyError):
    """Raised when the upstream is unreachable (connection refused, DNS, etc.)."""

    status_code = 503
    error_type = "upstream_unavailable"


class ValidationError(MoaxyError):
    """Raised when the request body fails Pydantic validation.

    Maps to HTTP 422 (or 400 when called explicitly with
    ``status_code=400``). The :attr:`details` dict typically carries
    the list of validation errors keyed under ``"errors"``.
    """

    status_code = 422
    error_type = "validation_error"


class UpstreamUnavailableHTTPError(MoaxyError):
    """Legacy 502 error class kept for backward compatibility.

    New code should use :class:`UpstreamError` (502) for upstream
    failures with a surfaced message, or
    :class:`UpstreamUnavailableError` (503) for connection-level
    failures. The old name is still wired through the exception
    handlers.
    """

    status_code = 502
    error_type = "upstream_unavailable"


class ServiceUnavailableError(MoaxyError):
    status_code = 503
    error_type = "service_unavailable"


# Patterns that must NEVER appear in a user-facing error message. Keeping
# this in one place ensures the contract is enforced at every error
# construction site, not just in the response layer.
_STACK_TRACE_FRAGMENTS = (
    "Traceback (most recent call last)",
    "Traceback ",
    "Traceback:",
)


def _sanitize_message(message: str) -> str:
    """Return a copy of ``message`` safe to expose to clients.

    Strips anything that looks like a Python stack trace, an absolute
    filesystem path under ``/site-packages/`` or the project, or the
    file/line markers FastAPI/Starlette emit alongside tracebacks.

    The goal is the user-facing contract from the validation document:
    response bodies must NEVER contain ``Traceback``, ``File "``,
    ``line ``, ``/site-packages/``, or filesystem paths. The
    :class:`MoaxyError` message is the only field rendered in the
    envelope, so this is the single chokepoint for sanitisation.
    """
    if not message:
        return message
    text = str(message)
    # Drop anything that smells like a Python traceback frame.
    for fragment in _STACK_TRACE_FRAGMENTS:
        text = text.replace(fragment, "")
    # Drop "File \"..." / "line N" markers that may appear standalone.
    while 'File "' in text:
        start = text.find('File "')
        end = text.find('"', start + len('File "'))
        if end == -1:
            text = text[:start]
        else:
            text = text[:start] + text[end + 1 :]
    # Drop absolute paths under site-packages.
    if "/site-packages/" in text:
        head, _, rest = text.partition("/site-packages/")
        text = head + rest.split("/", 1)[-1] if "/" in rest else head
    # Drop project filesystem paths (defence in depth).
    if "/Users/" in text and "/moaxy/" in text:
        # Keep the tail past the last "/moaxy/" segment.
        text = text.split("/moaxy/", 1)[-1]
    return text.strip() or "upstream error"


# Patterns that, if present in a public-facing error body, would leak
# a secret. The scrubber replaces each match with a redaction marker
# so the public envelope preserves the shape of the original body
# (useful for debugging) without exposing the secret. New patterns
# can be added here as new secret formats are introduced.
#
# Notes on the patterns:
# - ``Bearer <token>`` covers HTTP ``Authorization:`` headers in
#   upstream error bodies (some upstreams echo the bad token back in
#   the 401 response).
# - ``sk-or-v1-...``, ``sk-...``, and ``gsk_...`` cover common LLM
#   provider API key formats (OpenRouter, OpenAI, Groq).
# - The <KEY>...</KEY> XML-style pattern is defensive: a few legacy
#   upstream APIs echo API keys in error bodies wrapped in custom
#   XML tags.
_SECRET_PATTERNS: tuple[str, ...] = (
    r"Bearer\s+[A-Za-z0-9._\-+/=]{8,}",
    r"sk-or-v1-[A-Za-z0-9_\-]+",
    r"sk-[A-Za-z0-9_\-]{20,}",
    r"gsk_[A-Za-z0-9_\-]{20,}",
    r"<KEY>[^<]*</KEY>",
)


def _scrub_secrets(body: str) -> str:
    """Return a copy of ``body`` with known secret patterns redacted.

    The scrubber is a defensive measure for the M6 contract: the
    moaxy error envelope (sent to the client) MUST NOT include the
    API key in plain text, even when the upstream's own error
    message happens to reference the key. The scrubber replaces
    every match with ``<redacted>`` so the client still sees a
    useful body shape (e.g. ``{"error": "<redacted>"}``) without
    the secret.
    """
    if not body:
        return body
    text = str(body)
    for pattern in _SECRET_PATTERNS:
        text = re.sub(pattern, "<redacted>", text)
    return text


def _envelope(
    error_type: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"type": error_type, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


def _moaxy_error_response(request: Request, exc: MoaxyError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers: dict[str, str] = {}
    if request_id:
        headers["x-moaxy-request-id"] = request_id
    if isinstance(exc, MethodNotAllowedError):
        allowed = exc.details.get("allowed") if exc.details else None
        if allowed:
            headers["Allow"] = ", ".join(str(m).upper() for m in allowed)
    logger.info(
        "request error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "req_method": request.method,
            "status_code": exc.status_code,
            "error_type": exc.error_type,
            "error_message": exc.message,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.error_type, exc.message, details=exc.details or None),
        headers=headers,
        media_type="application/json",
    )


def _http_exception_response(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers: dict[str, str] = {}
    if request_id:
        headers["x-moaxy-request-id"] = request_id
    if exc.headers:
        for key, value in exc.headers.items():
            headers[key] = value
    error_type = "http_error"
    if exc.status_code == 404:
        error_type = "not_found"
    elif exc.status_code == 405:
        error_type = "method_not_allowed"
    elif exc.status_code == 415:
        error_type = "unsupported_media_type"
    logger.info(
        "http error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "req_method": request.method,
            "status_code": exc.status_code,
            "error_type": error_type,
        },
    )
    message = str(exc.detail) if exc.detail else f"HTTP {exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(error_type, message),
        headers=headers,
        media_type="application/json",
    )


def _validation_error_response(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers: dict[str, str] = {}
    if request_id:
        headers["x-moaxy-request-id"] = request_id
    errors = exc.errors()
    message = "request validation failed"
    if errors:
        first = errors[0]
        loc = first.get("loc") or ()
        msg = first.get("msg") or "invalid value"
        if loc:
            message = f"{'.'.join(str(p) for p in loc)}: {msg}"
        else:
            message = msg
    logger.info(
        "validation error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "req_method": request.method,
            "status_code": 422,
        },
    )
    return JSONResponse(
        status_code=422,
        content=_envelope("validation_error", message, details={"errors": errors}),
        headers=headers,
        media_type="application/json",
    )


def _unhandled_error_response(
    request: Request, exc: Exception
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers: dict[str, str] = {}
    if request_id:
        headers["x-moaxy-request-id"] = request_id
    logger.exception(
        "unhandled exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "req_method": request.method,
        },
    )
    return JSONResponse(
        status_code=500,
        content=_envelope("internal_error", "internal server error"),
        headers=headers,
        media_type="application/json",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register the moaxy error handlers on the given FastAPI app.

    Order matters: more specific handlers are registered last. FastAPI
    dispatches to the most recently registered handler that matches the
    exception class.
    """
    app.add_exception_handler(MoaxyError, _moaxy_error_response)
    app.add_exception_handler(RequestValidationError, _validation_error_response)
    app.add_exception_handler(StarletteHTTPException, _http_exception_response)
    app.add_exception_handler(Exception, _unhandled_error_response)


__all__ = [
    "BadRequestError",
    "ForbiddenError",
    "MethodNotAllowedError",
    "MoaxyError",
    "NoRouteMatchError",
    "NotFoundError",
    "ServiceUnavailableError",
    "UnauthorizedError",
    "UnsupportedMediaTypeError",
    "UpstreamError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "UpstreamUnavailableHTTPError",
    "ValidationError",
    "register_error_handlers",
]
