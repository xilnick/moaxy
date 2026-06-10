"""Tests for uniform error handling in the moaxy server.

These tests cover the error class hierarchy and the registered exception
handlers that translate every error into the canonical envelope:

    {"error": {"type": "<short_code>", "message": "<human-readable>"}}

The contract: every 4xx/5xx response has that body shape, the
``Content-Type`` is ``application/json``, and the body NEVER contains
``Traceback``, ``File "``, ``line ``, ``/site-packages/``, or any
filesystem path under ``/Users/.../moaxy/...``. Unhandled exceptions
are logged with the request id and stack trace, but the client only
sees a generic 500 JSON.

Custom error types covered here:

* :class:`NoRouteMatchError` → 404 with ``"no route matches"`` in the
  message.
* :class:`UpstreamError` → 502 with the upstream's error message.
* :class:`UpstreamTimeoutError` → 504.
* :class:`UpstreamUnavailableError` → 503.
* :class:`ValidationError` → 400/422.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from moaxy.models.config import (
    AdapterConfig,
    MoaxyConfig,
    RouteConfig,
)
from moaxy.models.config import (
    RouteMatch as ConfigRouteMatch,
)
from moaxy.server.app import create_app
from moaxy.server.errors import (
    MoaxyError,
    NoRouteMatchError,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    ValidationError,
    register_error_handlers,
)
from tests.conftest import (
    make_config,
    make_json_response,
    make_ollama_adapter,
)

# Patterns that must NEVER appear in an error response body.
LEAK_PATTERNS = (
    "Traceback",
    'File "',
    "line ",
    "/site-packages/",
)


def _is_path_under_project(value: str) -> bool:
    """Return True if ``value`` looks like a project filesystem path."""
    return "/Users/" in value and "/moaxy/" in value


# ────────────────────────────────────────────────────────────────────
# Error class hierarchy
# ────────────────────────────────────────────────────────────────────


class TestErrorClassHierarchy:
    """The custom error types have the right HTTP status and type code."""

    def test_no_route_match_error_is_404(self):
        exc = NoRouteMatchError("no route matches model 'x' on /v1/chat/completions")
        assert exc.status_code == 404
        assert exc.error_type == "no_route_match"
        assert "no route matches" in exc.message

    def test_upstream_error_is_502(self):
        exc = UpstreamError("upstream returned HTTP 500: bad")
        assert exc.status_code == 502
        assert exc.error_type == "upstream_error"
        assert "upstream" in exc.message.lower()

    def test_upstream_timeout_error_is_504(self):
        exc = UpstreamTimeoutError("upstream timeout talking to 127.0.0.1:11434")
        assert exc.status_code == 504
        assert exc.error_type == "upstream_timeout"

    def test_upstream_unavailable_error_is_503(self):
        exc = UpstreamUnavailableError("upstream unavailable: cannot connect")
        assert exc.status_code == 503
        assert exc.error_type == "upstream_unavailable"

    def test_validation_error_is_422_or_400(self):
        exc = ValidationError("request validation failed")
        assert exc.status_code in (400, 422)
        assert exc.error_type == "validation_error"

    def test_all_custom_errors_subclass_moaxy_error(self):
        for cls in (
            NoRouteMatchError,
            UpstreamError,
            UpstreamTimeoutError,
            UpstreamUnavailableError,
            ValidationError,
        ):
            err = cls("x")
            assert isinstance(err, MoaxyError)
            assert isinstance(err, Exception)

    def test_details_round_trip(self):
        exc = NoRouteMatchError(
            "no route matches model 'x' on /v1/chat/completions",
            details={"model": "x", "path": "/v1/chat/completions"},
        )
        assert exc.details == {"model": "x", "path": "/v1/chat/completions"}


# ────────────────────────────────────────────────────────────────────
# Response envelope shape on every error
# ────────────────────────────────────────────────────────────────────


class TestResponseEnvelope:
    """All error responses are JSON shaped like {error: {type, message}}."""

    @pytest.mark.asyncio
    async def test_no_route_match_envelope(self):
        app = create_app(config=MoaxyConfig(backends=[], routes=[]))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "missing-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert body["error"]["type"] == "no_route_match"
        assert "no route matches" in body["error"]["message"]
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_400_envelope_on_bad_request(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_404_envelope_on_unknown_path(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/nonexistent")
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_405_envelope(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/health")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]


# ────────────────────────────────────────────────────────────────────
# No stack-trace leaks (VAL-HTTP-025, VAL-CROSS-015)
# ────────────────────────────────────────────────────────────────────


class TestNoStackTraceLeaks:
    """No response body contains Python stack-trace fragments."""

    async def _post(self, app, path, *, json=None, content=None, headers=None):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                path,
                json=json,
                content=content,
                headers=headers or {"Content-Type": "application/json"},
            )

    @pytest.mark.asyncio
    async def test_400_does_not_leak(self):
        app = create_app(config=make_config())
        response = await self._post(
            app,
            "/v1/chat/completions",
            content=b"{not-json",
        )
        text = response.text
        for pat in LEAK_PATTERNS:
            assert pat not in text, f"leaked {pat!r} in body: {text[:200]}"
        assert not _is_path_under_project(text)

    @pytest.mark.asyncio
    async def test_404_does_not_leak(self):
        app = create_app(config=make_config())
        response = await self._post(
            app,
            "/v1/chat/completions",
            json={
                "model": "missing-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        text = response.text
        for pat in LEAK_PATTERNS:
            assert pat not in text
        assert not _is_path_under_project(text)

    @pytest.mark.asyncio
    async def test_405_does_not_leak(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/health")
        text = response.text
        for pat in LEAK_PATTERNS:
            assert pat not in text
        assert not _is_path_under_project(text)

    @pytest.mark.asyncio
    async def test_500_does_not_leak(self):
        """An unhandled exception surfaces as a clean 500 JSON."""
        app = FastAPI()
        register_error_handlers(app)

        @app.get("/boom")
        async def boom():
            raise RuntimeError("explode: /Users/secret/path/file.py line 42")

        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/boom")
        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "internal_error"
        # No traceback fragments and no filesystem path.
        text = response.text
        for pat in LEAK_PATTERNS:
            assert pat not in text, f"leaked {pat!r} in body: {text[:200]}"
        assert "/Users/" not in text or "/moaxy/" not in text
        assert "explode" not in text  # internal message is suppressed

    @pytest.mark.asyncio
    async def test_500_logs_request_id_and_stack_trace(self, caplog):
        """Unhandled exceptions log the request id and stack trace."""
        from moaxy.server.middleware import RequestIdMiddleware

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)
        register_error_handlers(app)

        @app.get("/boom")
        async def boom():
            raise RuntimeError("explode")

        with caplog.at_level(logging.ERROR):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.get(
                    "/boom", headers={"x-request-id": "fixed-id-123"}
                )
        assert response.status_code == 500
        # The log line carries the request id.
        matching = [
            record for record in caplog.records
            if getattr(record, "request_id", None) == "fixed-id-123"
        ]
        assert matching, "expected a log line with the request id"
        # The log message includes the exception info (traceback is in
        # the formatted log record).
        assert any(record.exc_info is not None for record in matching)


# ────────────────────────────────────────────────────────────────────
# Adapter-layer exception → JSON error mapping
# ────────────────────────────────────────────────────────────────────


class TestAdapterExceptionMapping:
    """Adapter-layer exceptions bubble up as the expected HTTP envelope."""

    @pytest.mark.asyncio
    async def test_adapter_upstream_error_returns_502(self):
        from moaxy.adapters.registry import AdapterRegistry

        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(
                {"error": {"message": "upstream is on fire"}}, status_code=500
            )

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 502
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "upstream_error"
        # The upstream's error message is surfaced.
        assert "upstream is on fire" in body["error"]["message"]
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_adapter_upstream_timeout_returns_504(self):
        from moaxy.adapters.registry import AdapterRegistry

        async def handler(_request: httpx.Request) -> httpx.Response:
            import httpx as _httpx

            raise _httpx.TimeoutException("read timed out")

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 504
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "upstream_timeout"
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_adapter_upstream_unavailable_returns_503(self):
        from moaxy.adapters.registry import AdapterRegistry

        async def handler(_request: httpx.Request) -> httpx.Response:
            import httpx as _httpx

            raise _httpx.ConnectError("connection refused")

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 503
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "upstream_unavailable"
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_adapter_upstream_4xx_returns_400(self):
        """A 4xx from the upstream surfaces as a 400 client error."""
        from moaxy.adapters.registry import AdapterRegistry

        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(
                {"error": {"message": "model not found"}}, status_code=404
            )

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        # The proxy converts the upstream 4xx into a clean bad-request
        # error; the upstream's message is included in details (if any)
        # but the body never contains a stack trace.
        text = response.text
        for pat in LEAK_PATTERNS:
            assert pat not in text


# ────────────────────────────────────────────────────────────────────
# Pydantic ValidationError
# ────────────────────────────────────────────────────────────────────


class TestValidationError:
    """A Pydantic ValidationError surfaces as a 400/422 JSON body."""

    @pytest.mark.asyncio
    async def test_pydantic_validation_error_returns_400_or_422(self):
        """Built-in FastAPI request validation returns 422 with the envelope."""
        app = FastAPI()
        register_error_handlers(app)

        from pydantic import BaseModel

        class Payload(BaseModel):
            model: str
            messages: list[str]

        @app.post("/validate")
        async def validate(payload: Payload) -> dict[str, str]:
            return {"ok": "yes"}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # messages: [] is empty, but Pydantic v2 default is fine;
            # instead we send a wrong type to trigger ValidationError.
            response = await client.post(
                "/validate", json={"model": "x", "messages": "not-a-list"}
            )
        assert response.status_code == 422
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "validation_error"
        # The validation details are surfaced.
        assert "details" in body["error"] or "errors" in body["error"]
        assert response.headers["content-type"].startswith("application/json")

    def test_validation_error_class_is_constructible(self):
        exc = ValidationError(
            "field 'messages': must be a non-empty list",
            details={"field": "messages"},
        )
        assert exc.status_code in (400, 422)
        assert exc.error_type == "validation_error"
        assert "messages" in exc.message


# ────────────────────────────────────────────────────────────────────
# Generic Exception handler
# ────────────────────────────────────────────────────────────────────


class TestUnhandledExceptionHandler:
    """The generic Exception handler returns a clean 500 JSON."""

    def test_register_handlers_includes_exception(self):
        app = FastAPI()
        register_error_handlers(app)
        # The base Exception handler is registered.
        assert Exception in app.exception_handlers
        assert MoaxyError in app.exception_handlers


# ────────────────────────────────────────────────────────────────────
# End-to-end: route match failure maps to 404 + no_route_match
# ────────────────────────────────────────────────────────────────────


class TestNoRouteMatchEndToEnd:
    """A request whose model matches no route returns 404 + no_route_match."""

    @pytest.mark.asyncio
    async def test_unknown_model_returns_no_route_match(self):
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(
                        model="known-model", path="/v1/chat/completions"
                    ),
                    backend="b",
                )
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown-xyz",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "no_route_match"
        assert "no route matches" in body["error"]["message"]
        assert "unknown-xyz" in body["error"]["message"]


# ────────────────────────────────────────────────────────────────────
# Request id propagation on error responses
# ────────────────────────────────────────────────────────────────────


class TestRequestIdOnErrors:
    """Every error response carries the x-moaxy-request-id header."""

    @pytest.mark.asyncio
    async def test_request_id_on_500(self):
        from moaxy.server.middleware import RequestIdMiddleware

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)
        register_error_handlers(app)

        @app.get("/boom")
        async def boom():
            raise RuntimeError("explode")

        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/boom")
        assert "x-moaxy-request-id" in response.headers
        assert response.headers["x-moaxy-request-id"]

    @pytest.mark.asyncio
    async def test_request_id_on_404_no_route(self):
        cfg = MoaxyConfig(backends=[], routes=[])
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "missing",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert "x-moaxy-request-id" in response.headers
        assert response.headers["x-moaxy-request-id"]
