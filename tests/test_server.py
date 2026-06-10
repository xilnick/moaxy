"""Tests for the moaxy FastAPI app factory and route handlers.

Covers:
- create_app() factory wires config, adapters, and route matcher.
- GET /health returns 200 with the canonical body and request id.
- GET /v1/models returns the OpenAI-shaped data array including aliases.
- POST /v1/chat/completions routes through the orchestrator and adapter.
- 4xx/5xx error envelopes (empty messages, missing model, missing
  Content-Type, malformed JSON, unknown path, wrong method).
- Content-Type is application/json for every response.
- x-moaxy-request-id is present and distinct across requests.
- Server bind (verified via lsof in a separate integration test).
"""

from __future__ import annotations

import asyncio
import re

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.adapters.registry import AdapterRegistry
from moaxy.models.config import (
    AdapterConfig,
    RouteConfig,
)
from moaxy.models.config import (
    RouteMatch as ConfigRouteMatch,
)
from moaxy.server.app import create_app
from moaxy.server.errors import (
    BadRequestError,
    MethodNotAllowedError,
    MoaxyError,
    NotFoundError,
    register_error_handlers,
)
from moaxy.server.middleware import (
    REQUEST_ID_HEADER,
    TIMING_HEADER,
)
from tests.conftest import (
    make_config,
    make_json_response,
    make_ollama_adapter,
    make_ollama_payload,
)

# ────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────


class TestCreateApp:
    """create_app() builds a FastAPI app with config, adapters, and routes."""

    def test_create_app_with_no_args(self):
        app = create_app()
        assert app.title == "moaxy"
        assert hasattr(app.state, "config")
        assert hasattr(app.state, "adapters")
        assert hasattr(app.state, "route_matcher")

    def test_create_app_with_explicit_config(self):
        cfg = make_config()
        app = create_app(config=cfg)
        assert app.state.config is cfg
        assert app.state.adapters is not None

    def test_create_app_with_custom_adapters(self):
        cfg = make_config()
        registry = AdapterRegistry()
        app = create_app(config=cfg, adapters=registry)
        assert app.state.adapters is registry

    def test_create_app_uses_route_matcher_for_routes(self):
        cfg = make_config()
        app = create_app(config=cfg)
        matcher = app.state.route_matcher
        assert matcher.routes == cfg.routes

    def test_create_app_preserves_plugins_dir(self):
        cfg = make_config()
        app = create_app(config=cfg, plugins_dir="custom-plugins")
        assert app.state.plugins_dir == "custom-plugins"

    def test_create_app_uses_default_plugins_dir(self):
        cfg = make_config()
        app = create_app(config=cfg)
        assert app.state.plugins_dir == cfg.plugins.plugins_dir


# ────────────────────────────────────────────────────────────────────
# Health endpoint (VAL-HTTP-001, VAL-HTTP-002)
# ────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /health returns 200 with a stable JSON body."""

    @pytest.mark.asyncio
    async def test_health_returns_200_with_status_ok(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body == {"status": "ok"}
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_health_content_type_is_application_json(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        ct = response.headers.get("content-type", "")
        assert ct.startswith("application/json")

    @pytest.mark.asyncio
    async def test_health_includes_request_id_header(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert REQUEST_ID_HEADER in response.headers
        assert len(response.headers[REQUEST_ID_HEADER]) > 0

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self):
        """VAL-HTTP-002: /health returns 200 even without credentials.

        M1 has no auth gate, so the assertion is simply that ``/health``
        works without sending any credentials — it is not on the data
        plane, so M1 does not enforce auth anywhere.
        """
        from moaxy.models.config import AuthConfig, MoaxyConfig

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                )
            ],
            auth=AuthConfig(enabled=True),
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200


# ────────────────────────────────────────────────────────────────────
# /v1/models endpoint (VAL-HTTP-003, VAL-HTTP-004)
# ────────────────────────────────────────────────────────────────────


class TestModelsEndpoint:
    """GET /v1/models returns an OpenAI-shaped model list."""

    @pytest.mark.asyncio
    async def test_models_returns_empty_data_when_no_routes(self):
        from moaxy.models.config import MoaxyConfig

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)

    @pytest.mark.asyncio
    async def test_models_includes_aliases(self):
        from moaxy.models.config import MoaxyConfig

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        body = response.json()
        ids = {m["id"] for m in body["data"]}
        assert "coder-pro" in ids
        assert "minimax-m3:cloud" in ids

    @pytest.mark.asyncio
    async def test_models_data_entries_have_object_field(self):
        from moaxy.models.config import MoaxyConfig

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"alias-1": "model-1"},
                )
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        for entry in response.json()["data"]:
            assert entry["object"] == "model"
            assert isinstance(entry["id"], str) and entry["id"]


# ────────────────────────────────────────────────────────────────────
# /v1/chat/completions endpoint (VAL-HTTP-005, 006, 007)
# ────────────────────────────────────────────────────────────────────


class TestChatCompletionsEndpoint:
    """POST /v1/chat/completions forwards to the configured adapter."""

    @pytest.mark.asyncio
    async def test_chat_completions_returns_200_with_choices(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(make_ollama_payload(content="PONG"))

        adapter = make_ollama_adapter(handler)
        cfg = make_config()
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "minimax-m2.7:cloud"
        assert len(body["choices"]) >= 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "PONG"

    @pytest.mark.asyncio
    async def test_chat_completions_includes_full_usage(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(
                make_ollama_payload(
                    content="ok",
                    prompt_tokens=12,
                    completion_tokens=4,
                    total_tokens=16,
                )
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
        body = response.json()
        assert "usage" in body
        assert body["usage"]["prompt_tokens"] == 12
        assert body["usage"]["completion_tokens"] == 4
        assert body["usage"]["total_tokens"] == 16
        assert (
            body["usage"]["total_tokens"]
            >= max(body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"])
        )

    @pytest.mark.asyncio
    async def test_chat_completions_includes_canonical_envelope(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(make_ollama_payload())

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
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        body = response.json()
        assert "id" in body and body["id"]
        assert body["object"] == "chat.completion"
        assert "created" in body and isinstance(body["created"], int)
        assert "model" in body
        assert "choices" in body and isinstance(body["choices"], list)
        assert "usage" in body and isinstance(body["usage"], dict)

    @pytest.mark.asyncio
    async def test_chat_completions_echoes_alias(self):
        from moaxy.models.config import MoaxyConfig

        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(make_ollama_payload(model="minimax-m3:cloud"))

        adapter = make_ollama_adapter(handler)
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        registry = AdapterRegistry({"b": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        body = response.json()
        assert body["model"] == "coder-pro"
        assert response.headers.get("x-moaxy-alias-resolved") == "minimax-m3:cloud"


# ────────────────────────────────────────────────────────────────────
# Error paths (VAL-HTTP-008, 009, 011, 012, 013, 014, 026)
# ────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    """The 4xx/5xx error envelopes for the data plane."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "minimax-m2.7:cloud", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "messages" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_model_returns_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "model" in body["error"]["message"].lower()
        assert response.headers.get("content-type", "").startswith("application/json")
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_missing_content_type_returns_415(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b'{"model":"x","messages":[{"role":"user","content":"hi"}]}',
            )
        assert response.status_code == 415
        body = response.json()
        assert "error" in body
        assert (
            "content-type" in body["error"]["message"].lower()
            or "content_type" in body["error"]["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_text_plain_content_type_returns_415(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b'{"model":"x","messages":[]}',
                headers={"Content-Type": "text/plain"},
            )
        assert response.status_code == 415
        body = response.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_malformed_json_returns_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_get_on_chat_completions_returns_405(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/chat/completions")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "method_not_allowed"
        allow = response.headers.get("allow") or response.headers.get("Allow")
        assert allow and "POST" in allow.upper()

    @pytest.mark.asyncio
    async def test_post_on_health_returns_405(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/health")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/nonexistent")
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert response.headers.get("content-type", "").startswith("application/json")

    @pytest.mark.asyncio
    async def test_no_route_matches_returns_404(self):
        from moaxy.models.config import MoaxyConfig

        cfg = MoaxyConfig(backends=[], routes=[])
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "some-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert "some-model" in body["error"]["message"]


# ────────────────────────────────────────────────────────────────────
# Response headers (VAL-HTTP-016, 017)
# ────────────────────────────────────────────────────────────────────


class TestResponseHeaders:
    """All responses carry the standard headers."""

    @pytest.mark.asyncio
    async def test_request_id_distinct_per_request(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/health")
            r2 = await client.get("/health")
        id1 = r1.headers[REQUEST_ID_HEADER]
        id2 = r2.headers[REQUEST_ID_HEADER]
        assert id1 and id2
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_request_id_is_uuid_hex(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        rid = r.headers[REQUEST_ID_HEADER]
        assert re.fullmatch(r"[0-9a-f]{32}", rid), f"unexpected id format: {rid!r}"

    @pytest.mark.asyncio
    async def test_request_id_present_on_error_responses(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        assert REQUEST_ID_HEADER in r.headers
        assert r.headers[REQUEST_ID_HEADER]

    @pytest.mark.asyncio
    async def test_content_type_application_json_on_success(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        assert r.headers.get("content-type", "").startswith("application/json")

    @pytest.mark.asyncio
    async def test_content_type_application_json_on_error(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        assert r.headers.get("content-type", "").startswith("application/json")

    @pytest.mark.asyncio
    async def test_timing_header_present(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        assert TIMING_HEADER in r.headers
        assert float(r.headers[TIMING_HEADER]) >= 0.0

    @pytest.mark.asyncio
    async def test_inbound_request_id_header_is_preserved(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health", headers={"x-request-id": "abc-123"})
        assert r.headers[REQUEST_ID_HEADER] == "abc-123"


# ────────────────────────────────────────────────────────────────────
# Stack-trace leak prevention (VAL-HTTP-025)
# ────────────────────────────────────────────────────────────────────


class TestNoStackTraceLeaks:
    """No response body contains a Python stack trace."""

    LEAK_PATTERNS = ("Traceback", 'File "', "line ", "/site-packages/")

    @pytest.mark.asyncio
    async def test_400_does_not_leak_traceback(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                content=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
        text = r.text
        for pat in self.LEAK_PATTERNS:
            assert pat not in text, f"leaked {pat!r} in body: {text[:200]}"

    @pytest.mark.asyncio
    async def test_404_does_not_leak_traceback(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/nonexistent")
        text = r.text
        for pat in self.LEAK_PATTERNS:
            assert pat not in text, f"leaked {pat!r} in body: {text[:200]}"

    @pytest.mark.asyncio
    async def test_405_does_not_leak_traceback(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/health")
        text = r.text
        for pat in self.LEAK_PATTERNS:
            assert pat not in text, f"leaked {pat!r} in body: {text[:200]}"


# ────────────────────────────────────────────────────────────────────
# Error envelope (uniform error shape)
# ────────────────────────────────────────────────────────────────────


class TestErrorEnvelope:
    """All errors are shaped like ``{"error": {"type", "message"}}``."""

    @pytest.mark.asyncio
    async def test_envelope_on_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        body = r.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]

    @pytest.mark.asyncio
    async def test_envelope_on_404(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/nonexistent")
        body = r.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]

    @pytest.mark.asyncio
    async def test_envelope_on_405(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/health")
        body = r.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]


# ────────────────────────────────────────────────────────────────────
# Middleware unit tests
# ────────────────────────────────────────────────────────────────────


class TestRequestIdMiddleware:
    """The request id middleware attaches a UUIDv4 to every request."""

    @pytest.mark.asyncio
    async def test_assigns_uuid(self):
        app = create_app(config=make_config())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/health")
        rid = r.headers[REQUEST_ID_HEADER]
        assert re.fullmatch(r"[0-9a-f]{32}", rid)

    @pytest.mark.asyncio
    async def test_unique_per_request(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            ids = set()
            for _ in range(5):
                r = await client.get("/health")
                ids.add(r.headers[REQUEST_ID_HEADER])
        assert len(ids) == 5


class TestTimingMiddleware:
    """The timing middleware records elapsed milliseconds."""

    @pytest.mark.asyncio
    async def test_timing_header_format(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        value = r.headers[TIMING_HEADER]
        float(value)


# ────────────────────────────────────────────────────────────────────
# Errors module unit tests
# ────────────────────────────────────────────────────────────────────


class TestErrorsModule:
    """The errors module exposes the right classes and helpers."""

    def test_moaxy_error_status_code_default(self):
        e = MoaxyError("x")
        assert e.status_code == 500
        assert e.error_type == "internal_error"

    def test_bad_request_error(self):
        e = BadRequestError("nope")
        assert e.status_code == 400
        assert e.error_type == "bad_request"

    def test_not_found_error(self):
        e = NotFoundError("nope")
        assert e.status_code == 404
        assert e.error_type == "not_found"

    def test_method_not_allowed_error(self):
        e = MethodNotAllowedError("nope")
        assert e.status_code == 405
        assert e.error_type == "method_not_allowed"

    def test_register_error_handlers_is_callable(self):
        from fastapi import FastAPI

        app = FastAPI()
        register_error_handlers(app)
        assert any(True for _ in app.exception_handlers)


# ────────────────────────────────────────────────────────────────────
# Concurrent request isolation (VAL-HTTP-024)
# ────────────────────────────────────────────────────────────────────


class TestConcurrentRequests:
    """Two concurrent requests get distinct request ids and responses."""

    @pytest.mark.asyncio
    async def test_two_concurrent_requests_have_distinct_ids(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(make_ollama_payload())

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1, r2 = await asyncio.gather(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m1",
                        "messages": [{"role": "user", "content": "a"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m1",
                        "messages": [{"role": "user", "content": "b"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
            )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert (
            r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]
        )


# (no module-level helper required; tests build their configs inline)
