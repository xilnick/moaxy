"""Server-level middleware: request_id, timing, structured JSON logging.

Three independent Starlette middlewares compose into a single
:class:`ServerMiddleware` mounted on the FastAPI app:

* :class:`RequestIdMiddleware` — assigns every request a UUIDv4 and
  stores it on ``request.state.request_id``; copies it into the response
  header ``x-moaxy-request-id``. Two successive requests always see two
  distinct ids.
* :class:`TimingMiddleware` — records ``time.monotonic()`` deltas on
  ``request.state.start_time`` and emits an ``x-moaxy-time-ms`` response
  header.
* :class:`StructuredLoggingMiddleware` — emits one JSON log line per
  request, carrying the request id, method, path, status, and elapsed
  milliseconds. The same log line includes the ``request_id`` so it can
  be correlated with the response header.

The middlewares are stacked: the request id is attached FIRST so the
timing and logging middlewares can read it from ``request.state``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "x-moaxy-request-id"
TIMING_HEADER = "x-moaxy-time-ms"
REQUEST_ID_ATTR = "request_id"
START_TIME_ATTR = "start_time"

_logger = logging.getLogger("moaxy.server.access")


def _new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a UUIDv4 to every request and echo it in the response.

    The id is also attached to ``request.state.request_id`` so downstream
    middlewares and route handlers can include it in their own log
    lines. An inbound ``x-request-id`` header is preserved when present
    (so an upstream proxy or test can override the id), but a fresh
    UUIDv4 is used when no header is supplied.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        inbound = request.headers.get("x-request-id") or request.headers.get(
            REQUEST_ID_HEADER
        )
        request_id = inbound or _new_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Record the start time and expose the elapsed milliseconds in a header."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.start_time = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - request.state.start_time) * 1000.0
        response.headers[TIMING_HEADER] = f"{elapsed_ms:.3f}"
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Emit a single JSON log line per request when the response is finalised.

    The log line carries the request id (from
    ``request.state.request_id``), the method, path, status code, and
    elapsed milliseconds. The same request id is present in the
    response header and the log entry, providing end-to-end correlation.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = getattr(request.state, "start_time", None) or time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            _logger.error(
                "request failed",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "req_method": request.method,
                    "req_path": request.url.path,
                    "status_code": 500,
                    "elapsed_ms": round(elapsed_ms, 3),
                },
            )
            raise

        elapsed_ms = (time.monotonic() - start) * 1000.0
        log_record = {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        message = json.dumps(log_record, sort_keys=True)
        if response.status_code >= 500:
            _logger.error(message)
        elif response.status_code >= 400:
            _logger.warning(message)
        else:
            _logger.info(message)
        return response


__all__ = [
    "REQUEST_ID_ATTR",
    "REQUEST_ID_HEADER",
    "RequestIdMiddleware",
    "START_TIME_ATTR",
    "StructuredLoggingMiddleware",
    "TIMING_HEADER",
    "TimingMiddleware",
]
