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


class MethodNotAllowedError(MoaxyError):
    status_code = 405
    error_type = "method_not_allowed"


class UpstreamUnavailableHTTPError(MoaxyError):
    status_code = 502
    error_type = "upstream_unavailable"


class ServiceUnavailableError(MoaxyError):
    status_code = 503
    error_type = "service_unavailable"


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
    "MethodNotAllowedError",
    "MoaxyError",
    "NotFoundError",
    "ServiceUnavailableError",
    "UnsupportedMediaTypeError",
    "UpstreamUnavailableHTTPError",
    "register_error_handlers",
]
