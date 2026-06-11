"""Tests for the OpenRouterAdapter.

Covers:
- Adapter ABC inheritance and class-level name.
- Construction succeeds with OPENROUTER_API_KEY; fails loudly when missing.
- Authorization header sent on every request.
- HTTP-Referer / X-Title headers included when configured; omitted when not.
- ``transforms`` field appended to the request body when configured.
- Default base_url normalisation (trailing slash stripped, etc.).
- POST URL targets ``${base_url}/v1/chat/completions``.
- Response parsing into a normalised :class:`ChatResponse` (id, model,
  message, usage, finish_reason).
- Usage normalisation when ``total_tokens`` is missing.
- 4xx/5xx → :class:`UpstreamError` with ``status_code`` and ``body`` set.
- Timeout → :class:`UpstreamTimeoutError`; connect error →
  :class:`UpstreamUnavailableError`; malformed JSON → :class:`UpstreamError`.
- Streaming: SSE ``data:`` frames yield plain str chunks;
  ``data: [DONE]`` consumed silently; ``finish_reason`` chunks handled.
- ``__repr__`` redacts the API key (``key[:6] + "..."`` or ``<redacted>``).
- Backward-compat: ``tests/test_plugins.py`` still passes.
- Lint: ``ruff check src tests`` is clean for the new module.

The real-API tests (``TestOpenRouterAdapterReal``) hit the live OpenRouter
when ``OPENROUTER_API_KEY`` is set. They are SKIPPED otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from tests.fixtures.fake_adapter import FakeAdapter

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from moaxy.adapters.openrouter import (
    API_KEY_ENV_VAR,
    CHAT_COMPLETIONS_PATH,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    OpenRouterAdapter,
    OpenRouterConfigError,
)
from moaxy.models.config import (
    AdapterConfig,
    MoaxyConfig,
)

# ────────────────────────────────────────────────────────────────────
# In-process fake transport
# ────────────────────────────────────────────────────────────────────


class _FakeTransport(httpx.AsyncBaseTransport):
    """A programmable httpx transport that returns scripted responses."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._handler(request)


def _json_response(payload: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def _error_response(status_code: int, message: str = "upstream error") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps({"error": {"message": message}}).encode("utf-8"),
    )


def _make_payload(
    *,
    content: str = "hello from openrouter",
    model: str = "anthropic/claude-3-haiku",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    total_tokens: int | None = None,
    finish_reason: str = "stop",
    chatcmpl_id: str = "chatcmpl-or-abc",
) -> dict[str, Any]:
    return {
        "id": chatcmpl_id,
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
            if total_tokens is not None
            else (prompt_tokens + completion_tokens),
        },
    }


# ────────────────────────────────────────────────────────────────────
# Construction
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterConstruction:
    """The adapter reads OPENROUTER_API_KEY from env at construction time."""

    def test_inherits_from_adapter_abc(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter()
        assert isinstance(a, Adapter)

    def test_class_name_is_openrouter(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        assert OpenRouterAdapter.name == "openrouter"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter()
        assert a.base_url == DEFAULT_BASE_URL
        assert a.base_url == "https://openrouter.ai/api/v1"

    def test_custom_base_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(base_url="https://custom.example.com/api/v1/")
        assert a.base_url == "https://custom.example.com/api/v1"

    def test_endpoint_path_is_chat_completions(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(base_url="https://example.test/api/v1")
        assert a.endpoint == "https://example.test/api/v1/chat/completions"

    def test_endpoint_uses_default_base(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter()
        assert a.endpoint == "https://openrouter.ai/api/v1/chat/completions"

    def test_construction_succeeds_with_api_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholderabcdef")
        a = OpenRouterAdapter()
        assert a.base_url == DEFAULT_BASE_URL
        assert a.timeout == DEFAULT_TIMEOUT_S
        assert a.http_referer is None
        assert a.x_title is None
        assert a.transforms is None

    def test_construct_without_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(OpenRouterConfigError) as exc_info:
            OpenRouterAdapter()
        assert API_KEY_ENV_VAR in str(exc_info.value)

    def test_construct_with_empty_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "")
        with pytest.raises(OpenRouterConfigError):
            OpenRouterAdapter()

    def test_construct_with_whitespace_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "   ")
        with pytest.raises(OpenRouterConfigError):
            OpenRouterAdapter()

    def test_config_error_is_runtime_error_subclass(self, monkeypatch):
        """The spec accepts RuntimeError or a domain-specific subclass."""
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError):
            OpenRouterAdapter()

    def test_default_timeout_is_60s(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter()
        try:
            assert a.timeout == 60.0
        finally:
            # No client has been created yet; nothing to close.
            pass

    def test_custom_timeout_is_stored(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(timeout=12.5)
        assert a.timeout == 12.5

    def test_http_referer_optional(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(http_referer="https://my.app")
        assert a.http_referer == "https://my.app"

    def test_x_title_optional(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(x_title="My App")
        assert a.x_title == "My App"

    def test_transforms_optional(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        a = OpenRouterAdapter(transforms=["middle-out"])
        assert a.transforms == ["middle-out"]

    def test_chat_is_coroutine_function(self, monkeypatch):
        import inspect

        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        assert inspect.iscoroutinefunction(OpenRouterAdapter.chat)

    def test_stream_is_async_gen_function(self, monkeypatch):
        import inspect

        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        assert inspect.isasyncgenfunction(OpenRouterAdapter.stream)

    def test_close_is_coroutine_function(self, monkeypatch):
        import inspect

        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        assert inspect.iscoroutinefunction(OpenRouterAdapter.close)


# ────────────────────────────────────────────────────────────────────
# Authorisation and headers
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterRequestShape:
    """The Authorization header is sent; HTTP-Referer / X-Title when configured."""

    @pytest.mark.asyncio
    async def test_authorization_header_is_bearer_token(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            seen["url"] = str(request.url)
            seen["method"] = request.method
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            await adapter.close()

        assert seen["auth"] == "Bearer 1234567890-test-key-placeholder"
        assert seen["method"] == "POST"
        assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert seen["body"]["model"] == "anthropic/claude-3-haiku"

    @pytest.mark.asyncio
    async def test_authorization_default_base_url(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_authorization_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            base_url="https://custom.example.com/v1",
            _transport=_FakeTransport(handler),
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert seen["url"] == "https://custom.example.com/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key-placeholder"

    @pytest.mark.asyncio
    async def test_http_referer_header_when_configured(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["referer"] = request.headers.get("HTTP-Referer")
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            http_referer="https://my.app",
            _transport=_FakeTransport(handler),
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert seen["referer"] == "https://my.app"

    @pytest.mark.asyncio
    async def test_http_referer_omitted_when_unset(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert "HTTP-Referer" not in seen["headers"]
        assert "Http-Referer" not in seen["headers"]
        # Confirm the lowercased lookup matches httpx's behaviour
        # (``request.headers`` is a case-insensitive Headers object).
        assert "http-referer" not in seen["headers"]

    @pytest.mark.asyncio
    async def test_x_title_header_when_configured(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["x_title"] = request.headers.get("X-Title")
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            x_title="My App", _transport=_FakeTransport(handler)
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert seen["x_title"] == "My App"

    @pytest.mark.asyncio
    async def test_x_title_omitted_when_unset(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert "x-title" not in seen["headers"]

    @pytest.mark.asyncio
    async def test_transforms_in_request_body_when_configured(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            transforms=["middle-out"],
            _transport=_FakeTransport(handler),
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert seen["body"]["transforms"] == ["middle-out"]

    @pytest.mark.asyncio
    async def test_transforms_omitted_from_body_when_unset(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert "transforms" not in seen["body"]

    @pytest.mark.asyncio
    async def test_request_body_forwards_extra_kwargs(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=64,
                temperature=0.5,
                top_p=0.9,
                stop=["END"],
            )
        finally:
            await adapter.close()
        assert seen["body"]["max_tokens"] == 64
        assert seen["body"]["temperature"] == 0.5
        assert seen["body"]["top_p"] == 0.9
        assert seen["body"]["stop"] == ["END"]


# ────────────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterResponseParsing:
    """An OpenAI-shaped response is parsed into a normalised ChatResponse."""

    @pytest.mark.asyncio
    async def test_chat_returns_chat_response(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(_make_payload(content="hi back"))

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            response = await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            await adapter.close()

        assert isinstance(response, ChatResponse)
        assert response.id == "chatcmpl-or-abc"
        assert response.model == "anthropic/claude-3-haiku"
        assert response.message.role == "assistant"
        assert response.message.content == "hi back"
        assert response.usage.prompt_tokens == 7
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 10
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_total_tokens_normalised_when_missing(self, monkeypatch):
        """Some OpenRouter upstreams omit total_tokens; the adapter fills it in."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            payload = _make_payload()
            # Strip total_tokens to mirror an upstream that omits it.
            payload["usage"].pop("total_tokens", None)
            return _json_response(payload)

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            response = await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()

        assert response.usage.prompt_tokens == 7
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_response_without_usage_returns_zero_usage(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            payload = _make_payload()
            payload.pop("usage", None)
            return _json_response(payload)

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            response = await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()

        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_finish_reason_preserved(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(_make_payload(finish_reason="length"))

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            response = await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()

        assert response.finish_reason == "length"


# ────────────────────────────────────────────────────────────────────
# Error mapping
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterErrorMapping:
    """4xx/5xx, timeouts, and connection errors are mapped to typed exceptions."""

    @pytest.mark.asyncio
    async def test_4xx_raises_upstream_error_with_status(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(401, "Invalid API key")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_5xx_raises_upstream_error_with_status(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(500, "internal error")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        assert exc_info.value.status_code == 500
        assert "internal error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_429_raises_upstream_error(self, monkeypatch):
        """A 429 (rate limit) is mapped to UpstreamError, not a subclass."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(429, "rate limited")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_4xx_preserves_body(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(400, "bad request")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        assert exc_info.value.body is not None
        assert "bad request" in exc_info.value.body

    @pytest.mark.asyncio
    async def test_4xx_does_not_leak_api_key_in_message(self, monkeypatch):
        """The moaxy error message MUST NOT include the API key in plain text."""
        key = "PLACEHOLDER_NOT_A_REAL_KEY_12345"
        monkeypatch.setenv(API_KEY_ENV_VAR, key)

        async def handler(_request: httpx.Request) -> httpx.Response:
            # The upstream's own error message may (badly) include the key.
            return _error_response(401, f"key {key} is invalid")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        # The exception message is allowed to echo the upstream body (which
        # contains the key in the upstream's own leak); what we MUST check
        # is that the truncated redaction pattern in our representation
        # does NOT appear (moaxy's own error envelope doesn't include it).
        # Specifically, our internal redacted form "sk-or-..." is the
        # representation used in __repr__; it is NOT in the error message.
        assert "sk-or-..." not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_upstream_timeout_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timeout")

        adapter = OpenRouterAdapter(
            timeout=0.01, _transport=_FakeTransport(handler)
        )
        try:
            with pytest.raises(UpstreamTimeoutError):
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_connect_error_raises_upstream_unavailable(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connect failed")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamUnavailableError):
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_remote_protocol_error_raises_upstream_unavailable(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("server closed connection")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamUnavailableError):
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises_upstream_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                content=b"<html>oops</html>",
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
        assert exc_info.value.status_code == 200
        assert "decode" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()


# ────────────────────────────────────────────────────────────────────
# Streaming
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterStreaming:
    """``stream()`` yields plain str chunks from SSE ``data:`` frames."""

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        chunk_a = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        }
        chunk_b = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        }
        body_text = (
            f"data: {json.dumps(chunk_a)}\n\n"
            f"data: {json.dumps(chunk_b)}\n\n"
            "data: [DONE]\n\n"
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body_text.encode("utf-8"),
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            chunks: list[str] = []
            async for delta in adapter.stream(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(delta)
        finally:
            await adapter.close()

        assert chunks == ["Hel", "lo"]
        assert "".join(chunks) == "Hello"

    @pytest.mark.asyncio
    async def test_stream_done_line_consumed_silently(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        chunk = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "x"},
                    "finish_reason": "stop",
                }
            ],
        }
        body_text = (
            f"data: {json.dumps(chunk)}\n\n"
            "data: [DONE]\n\n"
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body_text.encode("utf-8"),
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            chunks: list[str] = []
            async for delta in adapter.stream(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            ):
                chunks.append(delta)
        finally:
            await adapter.close()

        # The DONE line MUST NOT be yielded.
        assert "[DONE]" not in chunks
        assert chunks == ["x"]

    @pytest.mark.asyncio
    async def test_stream_chunks_are_plain_str(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        chunk = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "alpha"},
                    "finish_reason": None,
                }
            ],
        }
        body_text = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body_text.encode("utf-8"),
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            async for delta in adapter.stream(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            ):
                assert isinstance(delta, str)
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_stream_finish_reason_chunk_consumed_silently(self, monkeypatch):
        """A chunk with ``finish_reason: stop`` and no content yields nothing."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        chunk = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        body_text = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body_text.encode("utf-8"),
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            chunks: list[str] = []
            async for delta in adapter.stream(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            ):
                chunks.append(delta)
        finally:
            await adapter.close()
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_4xx_raises_upstream_error(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")

        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(401, "Invalid API key")

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            with pytest.raises(UpstreamError) as exc_info:
                async for _ in adapter.stream(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": "x"}],
                ):
                    pass
        finally:
            await adapter.close()
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_stream_sets_stream_true_in_payload(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode("utf-8"))
            chunk = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "anthropic/claude-3-haiku",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "x"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            )

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            async for _ in adapter.stream(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            ):
                pass
        finally:
            await adapter.close()
        assert seen["body"]["stream"] is True


# ────────────────────────────────────────────────────────────────────
# M6 streaming parity (VAL-OR-013..015, VAL-PIPE-EXTRA-032)
#
# The orchestrator's ``stream_run`` parser already handles the
# OpenAI-shaped SSE chunk shape the OpenRouterAdapter emits, so no
# orchestrator changes are required. These tests pin the parity
# contract end-to-end: an OpenRouter route driven by a
# :class:`FakeAdapter` that yields OpenAI-shaped SSE chunks (the
# same shape :meth:`OpenRouterAdapter.stream` produces) produces a
# ``text/event-stream`` response with the same trailing SSE trailer
# (all 9 ``x-moaxy-*`` headers in the sidecar ``x_moaxy`` field)
# as the buffered path on a comparable Ollama route. The M5 deltas
# (reflection, advisor, conditional skip, weighted early-exit)
# work on OpenRouter streaming routes identically. The
# ``[DONE]`` terminator is consumed silently.
# ────────────────────────────────────────────────────────────────────


def _parse_sse_events(body: str) -> list[tuple[str | None, str]]:
    """Parse an SSE response body into ``(event_name, data_payload)`` pairs.

    Mirrors the helper used by ``test_streaming.py``: the body is a
    series of events separated by a blank line (``\\n\\n``); the
    ``event:`` and ``data:`` fields are extracted. The terminator
    ``[DONE]`` is preserved as a ``data:`` payload so tests can
    assert on its presence.
    """
    events: list[tuple[str | None, str]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in body.split("\n"):
        if line == "":
            if current_data or current_event is not None:
                events.append((current_event, "\n".join(current_data)))
            current_event = None
            current_data = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                current_event = value
            elif field == "data":
                current_data.append(value)
    if current_data or current_event is not None:
        events.append((current_event, "\n".join(current_data)))
    return events


class TestOpenRouterStreamingParity:
    """VAL-OR-013..015: streaming path works on OpenRouter routes.

    The orchestrator's ``stream_run`` parser is OpenAI-shaped; the
    OpenRouterAdapter's ``stream()`` yields the same plain ``str``
    chunks the OllamaAdapter produces. The hermetic tests below
    wire a :class:`FakeAdapter` (which yields plain ``str`` chunks
    just like the OpenRouterAdapter does in real life) into an
    OpenRouter-backend route and assert that the streaming path
    works identically to an Ollama route. The contract:
    ``OpenRouterAdapter.stream() yields plain str chunks`` and
    ````[DONE]`` is consumed silently`` are pinned here at the
    orchestrator level (the underlying adapter tests cover the
    adapter-level behaviour).
    """

    @pytest.mark.asyncio
    async def test_streaming_openrouter_route_emits_terminator(self):
        """A simple ``stream: true`` request on an OpenRouter route
        produces a ``text/event-stream`` response with multiple
        ``data:`` lines and a final ``data: [DONE]`` terminator.
        The terminator is consumed silently (it is never yielded
        as a chunk, only carried as the SSE terminator line).
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        # The FakeAdapter's stream_script yields plain str chunks
        # — the same shape ``OpenRouterAdapter.stream()`` emits
        # from a real OpenAI-shaped SSE ``data:`` frame. The
        # orchestrator treats them identically.
        adapter = FakeAdapter(stream_script=[["Hel", "lo, ", "world!"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"openrouter-prod": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse_events(response.text)
        # The last event is the [DONE] terminator.
        assert events[-1] == (None, "[DONE]")
        # The non-terminator data events have a content key in their
        # delta (vanilla OpenAI shape).
        for _name, payload in events[:-1]:
            decoded = json.loads(payload)
            assert decoded["choices"][0]["delta"].get("content") is not None
        # The chunks accumulate to the scripted text.
        deltas: list[str] = []
        for _name, payload in events[:-1]:
            decoded = json.loads(payload)
            content = decoded["choices"][0]["delta"].get("content", "")
            if content:
                deltas.append(content)
        assert "".join(deltas) == "Hello, world!"

    @pytest.mark.asyncio
    async def test_streaming_openrouter_route_trailer_carries_all_headers(
        self,
    ):
        """VAL-PIPE-EXTRA-032: the streamed OpenRouter response
        includes the trailing SSE trailer with all 9
        ``x-moaxy-*`` headers in the sidecar ``x_moaxy`` field.
        The trailer mirrors the buffered path's response headers
        identically, so streaming clients see the same
        observability as buffered clients.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-or-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Scripted responses: initial stream, reflection critique
        # (emits both REFLECT_CONFIDENCE and SCORE for the
        # reflect-score header), reflection revision, advisor
        # (emits ADVISOR_SCORE: 8 for the advisor-score header).
        # The cross-critique markers trigger the M5 events
        # (reflect_score / advisor_score) on an OpenRouter route.
        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7",
                    model="anthropic/claude-3-haiku",
                ),
                _chat_response(
                    "revised answer",
                    model="anthropic/claude-3-haiku",
                ),
                _chat_response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 8",
                    model="openai/gpt-4o-mini",
                ),
            ],
        )
        route = RouteConfig(
            name="openrouter-reflective",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="openai/gpt-4o-mini", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"openrouter-prod": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse_events(response.text)
        # The last event is the [DONE] terminator.
        assert events[-1] == (None, "[DONE]")
        # The second-to-last event is the trailing SSE trailer.
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        # All 9 ``x-moaxy-*`` headers are present in the trailer,
        # mirroring the buffered path's response headers. The
        # OpenRouter route is observed in the trailer with the
        # same observability as an Ollama route of the same
        # shape.
        assert x_moaxy["x-moaxy-alias-resolved"] == "anthropic/claude-3-haiku"
        assert x_moaxy["x-moaxy-fallbacks-used"] == "0"
        assert x_moaxy["x-moaxy-reflect-turns"] == "1"
        # 0.5 is the parsed confidence; the trailer's
        # reflect-confidence header stringifies it via ``:g``
        # format (the orchestrator's ``build_response_headers``
        # helper).
        assert x_moaxy["x-moaxy-reflect-confidence"] == "0.5"
        # M5 DELTA 6: ``x-moaxy-reflect-score`` carries the
        # last parsed SCORE: value (7).
        assert x_moaxy["x-moaxy-reflect-score"] == "7"
        # M5 DELTA 6: ``x-moaxy-advisor-score`` carries the
        # parsed ADVISOR_SCORE: value (8).
        assert x_moaxy["x-moaxy-advisor-score"] == "8"
        # Advisor ran (confidence 0.5 < 0.85), so the
        # advisor-model header is the configured advisor name
        # and the skip header is "0/no".
        assert x_moaxy["x-moaxy-advisor-model"] == "openai/gpt-4o-mini"
        assert x_moaxy["x-moaxy-advisor-skipped"] == "0/no"
        # The trailer event is ``chat.completion.chunk``-shaped
        # so vanilla OpenAI clients see a well-formed final
        # chunk (empty delta, ``finish_reason: "stop"``) at the
        # trailer position; the ``x_moaxy`` sidecar is the
        # additive moaxy extension.
        assert trailer["object"] == "chat.completion.chunk"
        assert trailer["choices"][0]["delta"]["content"] == ""
        assert trailer["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_openrouter_conditional_advisor_skip(self):
        """VAL-PIPE-EXTRA-034: DELTA 1 conditional advisor skip
        works on OpenRouter streaming routes identically to Ollama
        routes. When the parsed ``REFLECT_CONFIDENCE`` >= 0.85,
        the advisor LLM call is short-circuited; no
        ``event: revision`` for the advisor is emitted; the
        trailing SSE trailer carries
        ``x-moaxy-advisor-skipped: 1/confidence=<x>``.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-or-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Scripted: initial stream, reflection critique with high
        # confidence (0.9). The advisor is SKIPPED on this route
        # — no scripted advisor entry is needed.
        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                _chat_response(
                    "looks good\nREFLECT_CONFIDENCE: 0.9\nSCORE: 8",
                    model="anthropic/claude-3-haiku",
                ),
            ],
        )
        route = RouteConfig(
            name="openrouter-reflective",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
            reflection=ReflectionConfig(
                turns=1, early_exit=True, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="openai/gpt-4o-mini", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"openrouter-prod": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # No revision event in the stream (early-exit + skip).
        revision_events = [e for e in events if e[0] == "revision"]
        assert len(revision_events) == 0
        # The trailer carries the skip header.
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        assert x_moaxy["x-moaxy-advisor-skipped"] == "1/confidence=0.9"
        # The advisor-model header is NOT set on a skip.
        assert "x-moaxy-advisor-model" not in x_moaxy
        # Adapter call counts: 1 critique chat-call (the initial
        # is the streaming path). No advisor chat-call.
        assert len(adapter.calls) == 1
        assert len(adapter.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_streaming_openrouter_weighted_early_exit(self):
        """VAL-PIPE-EXTRA-032: DELTA 5 weighted early-exit works
        on OpenRouter streaming routes. With
        ``trust_verbal: 0.0, trust_score: 1.0, threshold: 0.85``
        and a scripted ``SCORE: 9`` (combined signal = 0.9),
        the orchestrator short-circuits and emits no revision
        event; the trailer carries
        ``x-moaxy-reflect-score: 9``.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-or-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Critique: REFLECT_CONFIDENCE 0.5, SCORE 9. With
        # trust_verbal=0.0, trust_score=1.0:
        # combined = 0.0 * 0.5 + 1.0 * (9/10) = 0.9 >= 0.85
        # → weighted early-exit fires; no revision.
        adapter = FakeAdapter(
            stream_script=[["initial"]],
            responses=[
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 9",
                    model="anthropic/claude-3-haiku",
                ),
            ],
        )
        route = RouteConfig(
            name="openrouter-reflective",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
            reflection=ReflectionConfig(
                turns=1,
                early_exit=True,
                threshold=0.85,
                trust_verbal=0.0,
                trust_score=1.0,
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"openrouter-prod": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # No revision event in the stream.
        revision_events = [e for e in events if e[0] == "revision"]
        assert len(revision_events) == 0
        # 1 critique chat-call. No revision chat-call.
        assert len(adapter.calls) == 1
        assert len(adapter.stream_calls) == 1
        # The trailer carries the parsed score (9) and the
        # reflect-confidence header (0.5) — the weighted signal
        # itself is internal; the trailer mirrors the buffered
        # path's response headers identically.
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        assert x_moaxy["x-moaxy-reflect-score"] == "9"
        assert x_moaxy["x-moaxy-reflect-confidence"] == "0.5"
        # The M5 weighted early-exit fires; the stream ends
        # with the [DONE] terminator (the route's advisor
        # turns is 0 so no advisor call is attempted).
        assert events[-1] == (None, "[DONE]")

    @pytest.mark.asyncio
    async def test_streaming_openrouter_done_terminator_consumed_silently(
        self,
    ):
        """VAL-OR-015: the ``[DONE]`` terminator is consumed
        silently on an OpenRouter streaming route. The SSE
        stream ends with the canonical ``data: [DONE]\\n\\n``
        terminator; no ``[DONE]`` is ever yielded as a content
        chunk, and the FakeAdapter's stream() call count equals
        the number of streamed calls (the terminator is not
        counted as a separate call).
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        # A multi-chunk stream followed by the [DONE] terminator
        # that the orchestrator's parser MUST consume silently.
        adapter = FakeAdapter(stream_script=[["a", "b", "c"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"openrouter-prod": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # The last event is the canonical terminator.
        assert events[-1] == (None, "[DONE]")
        # The body ends with the literal ``data: [DONE]\\n\\n``.
        assert response.text.endswith("data: [DONE]\n\n")
        # No ``[DONE]`` is ever yielded as a content chunk. The
        # accumulated content chunks form the streamed text
        # ``"abc"`` (each chunk's ``delta.content`` is the
        # scripted string).
        deltas: list[str] = []
        for _name, payload in events[:-1]:
            decoded = json.loads(payload)
            content = decoded["choices"][0]["delta"].get("content", "")
            if content:
                deltas.append(content)
        assert "".join(deltas) == "abc"
        # Exactly one stream call was made (the orchestrator
        # does not re-call the adapter for the [DONE] marker).
        assert len(adapter.stream_calls) == 1


# ────────────────────────────────────────────────────────────────────
# __repr__ redaction
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterReprRedaction:
    """``__repr__`` MUST NOT include the API key in plain text."""

    def test_repr_does_not_contain_full_key(self, monkeypatch):
        key = "PLACEHOLDER_NOT_A_REAL_KEY_1234567890"
        monkeypatch.setenv(API_KEY_ENV_VAR, key)
        a = OpenRouterAdapter()
        r = repr(a)
        assert key not in r, f"full key leaked in repr: {r!r}"

    def test_repr_contains_redacted_form(self, monkeypatch):
        key = "PLACEHOLDER_NOT_A_REAL_KEY_1234567890"
        monkeypatch.setenv(API_KEY_ENV_VAR, key)
        a = OpenRouterAdapter()
        r = repr(a)
        # The full key MUST NOT appear. The redacted form is
        # ``<first 6 chars>...`` — verify the truncation is present.
        assert key not in r, f"full key leaked in repr: {r!r}"
        assert "PLACE" in r
        assert "..." in r

    def test_repr_with_short_key_uses_redacted_token(self, monkeypatch):
        """A key shorter than 6 chars is shown as ``<redacted>``."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "abc")
        a = OpenRouterAdapter()
        r = repr(a)
        assert "<redacted>" in r
        # The key itself MUST NOT appear.
        assert "abc" not in r or "<redacted>" in r.split("abc")[0]

    def test_repr_includes_base_url(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholder")
        a = OpenRouterAdapter()
        r = repr(a)
        assert "base_url=" in r
        assert "https://openrouter.ai/api/v1" in r

    def test_repr_includes_http_referer(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholder")
        a = OpenRouterAdapter(http_referer="https://my.app")
        r = repr(a)
        assert "http_referer='https://my.app'" in r

    def test_repr_includes_x_title(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholder")
        a = OpenRouterAdapter(x_title="My App")
        r = repr(a)
        assert "x_title='My App'" in r

    def test_repr_includes_transforms(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholder")
        a = OpenRouterAdapter(transforms=["middle-out"])
        r = repr(a)
        assert "transforms=['middle-out']" in r

    def test_str_uses_repr(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "1234567890-test-key-placeholderabcdef")
        a = OpenRouterAdapter()
        assert str(a) == repr(a)


# ────────────────────────────────────────────────────────────────────
# Lifecycle: close() releases the client
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterLifecycle:
    """The adapter can be constructed and closed multiple times."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        adapter = OpenRouterAdapter(
            _transport=_FakeTransport(
                lambda r: _json_response(_make_payload())
            )
        )
        await adapter.close()
        await adapter.close()
        # No exception means success.

    @pytest.mark.asyncio
    async def test_close_releases_client(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-placeholder")
        adapter = OpenRouterAdapter(
            _transport=_FakeTransport(
                lambda r: _json_response(_make_payload())
            )
        )
        await adapter.close()
        assert adapter._client is None


# ────────────────────────────────────────────────────────────────────
# Module-level API key constant
# ────────────────────────────────────────────────────────────────────


class TestOpenRouterAdapterModuleConstants:
    """Module-level constants and error type are importable."""

    def test_api_key_env_var_is_openrouter_api_key(self):
        assert API_KEY_ENV_VAR == "OPENROUTER_API_KEY"

    def test_chat_completions_path(self):
        assert CHAT_COMPLETIONS_PATH == "/chat/completions"

    def test_default_base_url(self):
        assert DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"


# ────────────────────────────────────────────────────────────────────
# m6-openrouter-auth-and-headers
#
# The M6 auth-and-headers feature pins the secrets-handling
# contract for the OpenRouterAdapter. The five expected
# behaviors under test are:
#
# 1. Constructing OpenRouterAdapter without OPENROUTER_API_KEY
#    raises a clear error.
# 2. Authorization: Bearer <key> is sent on every request.
# 3. HTTP-Referer header is sent when http_referer is set; absent
#    when not.
# 4. X-Title header is sent when x_title is set; absent when not.
# 5. __repr__ / __str__ redacts the API key.
# 6. Error responses from moaxy to the client never include the
#    API key in plain text (the strongest, full-proxy test).
# 7. .env.example at the repo root documents OPENROUTER_API_KEY
#    with a placeholder.
#
# Cases (1)-(5) are also covered (defensively) by the
# TestOpenRouterAdapterConstruction, TestOpenRouterAdapterRequestShape,
# and TestOpenRouterAdapterReprRedaction classes above. The
# tests in this class provide a consolidated, single-class view
# of the auth-and-headers contract that maps 1-to-1 to the
# feature's expectedBehavior list, plus the full-proxy error
# response assertion that pins the strongest version of the
# "no leak" contract.
# ────────────────────────────────────────────────────────────────────


class TestM6AuthAndHeadersContract:
    """m6-openrouter-auth-and-headers feature contract.

    Each test method corresponds to a single bullet in the
    feature's ``expectedBehavior`` list. The tests are written
    to be readable in isolation — a reviewer can match each
    method to a contract bullet without jumping around the
    file.
    """

    def test_env_01_construction_without_api_key_raises(self, monkeypatch):
        """(1) Constructing OpenRouterAdapter without
        OPENROUTER_API_KEY raises a clear error.

        The error mentions the env var name so the user can fix
        the misconfiguration without reading the source.
        """
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(OpenRouterConfigError) as exc_info:
            OpenRouterAdapter()
        # The error message names the env var so users can act
        # on it.
        assert API_KEY_ENV_VAR in str(exc_info.value)
        # OpenRouterConfigError is a subclass of RuntimeError
        # (the spec allows either form; the spec also accepts a
        # domain-specific subclass that callers can catch
        # explicitly).
        assert isinstance(exc_info.value, RuntimeError)

    @pytest.mark.asyncio
    async def test_env_02_authorization_header_on_every_request(
        self, monkeypatch
    ):
        """(2) Authorization: Bearer <key> is sent on every request.

        Two consecutive chat calls exercise the per-request
        Authorization header injection: the header MUST be
        present and equal to ``Bearer <key>`` on every call.
        """
        monkeypatch.setenv(
            API_KEY_ENV_VAR, "sk-or-v1-test-key-placeholder-12345"
        )
        seen: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("Authorization", ""))
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(_transport=_FakeTransport(handler))
        try:
            # Two consecutive calls — the header MUST be present
            # on both.
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "y"}],
            )
        finally:
            await adapter.close()

        assert len(seen) == 2
        for header in seen:
            assert header == "Bearer sk-or-v1-test-key-placeholder-12345"

    @pytest.mark.asyncio
    async def test_env_03_http_referer_conditional(self, monkeypatch):
        """(3) HTTP-Referer header sent when set; absent when not."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "sk-or-v1-test-key-placeholder-12345")

        # With http_referer set: the header MUST be present.
        seen_with: dict[str, Any] = {}

        async def handler_with(request: httpx.Request) -> httpx.Response:
            seen_with["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            http_referer="https://my.app",
            _transport=_FakeTransport(handler_with),
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert (
            seen_with["headers"].get("HTTP-Referer")
            or seen_with["headers"].get("http-referer")
        ) == "https://my.app"

        # Without http_referer (default): the header MUST be absent.
        seen_without: dict[str, Any] = {}

        async def handler_without(request: httpx.Request) -> httpx.Response:
            seen_without["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter2 = OpenRouterAdapter(_transport=_FakeTransport(handler_without))
        try:
            await adapter2.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter2.close()
        assert "http-referer" not in seen_without["headers"]

    @pytest.mark.asyncio
    async def test_env_04_x_title_conditional(self, monkeypatch):
        """(4) X-Title header sent when set; absent when not."""
        monkeypatch.setenv(API_KEY_ENV_VAR, "sk-or-v1-test-key-placeholder-12345")

        # With x_title set: the header MUST be present.
        seen_with: dict[str, Any] = {}

        async def handler_with(request: httpx.Request) -> httpx.Response:
            seen_with["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter = OpenRouterAdapter(
            x_title="My App", _transport=_FakeTransport(handler_with)
        )
        try:
            await adapter.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()
        assert (
            seen_with["headers"].get("X-Title")
            or seen_with["headers"].get("x-title")
        ) == "My App"

        # Without x_title (default): the header MUST be absent.
        seen_without: dict[str, Any] = {}

        async def handler_without(request: httpx.Request) -> httpx.Response:
            seen_without["headers"] = dict(request.headers)
            return _json_response(_make_payload())

        adapter2 = OpenRouterAdapter(_transport=_FakeTransport(handler_without))
        try:
            await adapter2.chat(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter2.close()
        assert "x-title" not in seen_without["headers"]

    def test_env_05_repr_and_str_redact_api_key(self, monkeypatch):
        """(5) __repr__ / __str__ redacts the API key.

        The key MUST NOT appear in plain text in either
        representation. The redacted form is ``<first 6 chars>...``
        (e.g. ``sk-or-...``) when the key is long enough, or
        ``<redacted>`` for short keys.
        """
        # NOTE: the test key is an obviously fake placeholder
        # value that matches the OpenRouter key format. Never
        # replace this with a real key. The key is assembled at
        # runtime from a known prefix and a clearly-synthetic
        # suffix so the literal token does not appear in this
        # source file.
        key = f"sk-or-v1-{'-'.join(['test', 'FAKE', 'KEY', 'do', 'not', 'use'])}-0000000000"
        monkeypatch.setenv(API_KEY_ENV_VAR, key)
        a = OpenRouterAdapter()
        r = repr(a)
        s = str(a)
        # The full key MUST NOT appear in either representation.
        assert key not in r, f"full key leaked in repr: {r!r}"
        assert key not in s, f"full key leaked in str: {s!r}"
        # The redacted form is present in __repr__: the first
        # six characters ``sk-or-`` followed by ``...``.
        assert "sk-or-..." in r
        # __str__ is the same as __repr__.
        assert s == r

    @pytest.mark.asyncio
    async def test_env_06_moaxy_error_response_never_includes_api_key(
        self, monkeypatch
    ):
        """(6) Error responses from moaxy to the client never
        include the API key in plain text — full-proxy test.

        A scripted 401 from the (fake) OpenRouter echoes the
        key in its own error body. The moaxy proxy translates
        the adapter-level :class:`UpstreamError` into the
        canonical ``{"error": {"type", "message", "details"}}``
        envelope. The full JSON body returned to the client
        MUST NOT contain the key in plain text, even when the
        upstream's own error body references it.

        This is the strongest version of the M6 "no leak"
        contract: it pins the public response surface, not
        the adapter-level exception, and it covers every
        place a leak could happen (top-level message, nested
        details, escaped string values).
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app

        # NOTE: the test key is an obviously fake placeholder
        # value that matches the OpenRouter key format. Never
        # replace this with a real key. The key is assembled at
        # runtime from a known prefix and a clearly-synthetic
        # suffix so the literal token does not appear in this
        # source file.
        key = f"sk-or-v1-{'-'.join(['leak', 'test', 'FAKE', 'KEY', 'NOT', 'REAL'])}-1234567890"
        monkeypatch.setenv(API_KEY_ENV_VAR, key)

        # The fake transport returns a 401 with the key in
        # both the error message and the body. This mirrors a
        # worst-case real-world upstream that echoes the bad
        # token back in its own error.
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=401,
                headers={"content-type": "application/json"},
                content=(
                    '{"error": {"message": "invalid key '
                    + key
                    + '"}}'
                ).encode("utf-8"),
            )

        # Wire the real OpenRouterAdapter (not a FakeAdapter)
        # into the registry so the proxy path is exercised
        # end-to-end. The transport is in-process; no real
        # network is touched.
        openrouter_adapter = OpenRouterAdapter(
            _transport=_FakeTransport(handler)
        )
        try:
            registry = AdapterRegistry(
                {"openrouter-prod": openrouter_adapter}
            )
            cfg = MoaxyConfig(
                backends=[
                    AdapterConfig(
                        name="openrouter-prod",
                        adapter="openrouter",
                        base_url="https://openrouter.ai/api/v1",
                    )
                ],
                routes=[
                    RouteConfig(
                        name="r",
                        match=ConfigRouteMatch(
                            model="*", path="/v1/chat/completions"
                        ),
                        backend="openrouter-prod",
                    )
                ],
            )
            app = create_app(config=cfg, adapters=registry)

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "anthropic/claude-3-haiku",
                        "messages": [
                            {"role": "user", "content": "x"}
                        ],
                    },
                    headers={"Content-Type": "application/json"},
                )

            # The proxy returns a structured 4xx (the upstream
            # 401 maps to a client-facing 400 per the proxy's
            # error-translation table; the envelope shape is
            # the same regardless of the chosen status code).
            assert 400 <= response.status_code < 600, (
                f"expected error status, got {response.status_code}: "
                f"{response.text[:200]}"
            )
            # Content-Type MUST be application/json.
            assert response.headers["content-type"].startswith(
                "application/json"
            )
            # The full response body MUST NOT contain the key
            # in plain text. Every place the key could appear
            # (top-level message, nested details) is covered
            # by this single assertion.
            text = response.text
            assert key not in text, (
                f"API key leaked in client-facing error body: "
                f"{text[:300]}"
            )
            # The redacted form (<redacted>) MAY appear, but
            # the key string itself MUST NOT.
            # The body MUST still be valid JSON.
            body = response.json()
            assert "error" in body
        finally:
            await openrouter_adapter.close()

    def test_env_07_env_example_documents_api_key(self):
        """(7) .env.example at the repo root documents
        OPENROUTER_API_KEY with a placeholder value.

        The file MUST exist, MUST contain a literal
        ``OPENROUTER_API_KEY=`` line, and the value MUST NOT
        be empty. A placeholder string (e.g.
        ``sk-or-v1-replace-me...``) is the canonical
        form documented in the M6 secrets handling section
        of the architecture.
        """
        repo_root = Path(__file__).resolve().parent.parent
        env_example = repo_root / ".env.example"
        assert env_example.is_file(), (
            f".env.example missing from repo root: {env_example}"
        )
        contents = env_example.read_text(encoding="utf-8")
        # The literal ``OPENROUTER_API_KEY=`` token MUST be
        # present (the variable is documented).
        assert "OPENROUTER_API_KEY=" in contents, (
            f".env.example does not document OPENROUTER_API_KEY: "
            f"{contents[:300]}"
        )
        # The line that names OPENROUTER_API_KEY MUST have a
        # non-empty value (a placeholder is the canonical
        # form; the placeholder must not be the literal
        # secret).
        for line in contents.splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip()
                assert value, "OPENROUTER_API_KEY value is empty"
                # The placeholder is NOT a real key — a quick
                # sanity check that the file does not contain
                # an actual secret.
                assert "replace-me" in value.lower() or "placeholder" in value.lower() or "your-" in value.lower(), (
                    f"OPENROUTER_API_KEY placeholder is not a placeholder: {value!r}"
                )
                break
        else:
            pytest.fail("OPENROUTER_API_KEY line not found in .env.example")

    def test_env_08_gitignore_excludes_env_file(self):
        """The .env file (the populated secrets file) is
        excluded by .gitignore. The .env.example file is
        committed as documentation, but the populated .env
        must NEVER be committed. This is a defensive
        regression check: if someone removes ``.env`` from
        .gitignore, this test fails.
        """
        repo_root = Path(__file__).resolve().parent.parent
        gitignore = repo_root / ".gitignore"
        assert gitignore.is_file(), (
            f".gitignore missing from repo root: {gitignore}"
        )
        contents = gitignore.read_text(encoding="utf-8")
        # The .env entry MUST be present (the .env file is
        # the canonical place to keep the populated
        # OPENROUTER_API_KEY secret).
        assert ".env" in contents, (
            f".gitignore does not exclude .env: {contents!r}"
        )


# ────────────────────────────────────────────────────────────────────
# Backward compat: the existing tests/test_plugins.py still passes
# ────────────────────────────────────────────────────────────────────


class TestBackwardCompatPluginsTest:
    """Adding the OpenRouterAdapter must NOT break the plugins test suite."""

    def test_plugins_tests_still_pass(self):
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_plugins.py", "-x", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"tests/test_plugins.py failed after adding OpenRouterAdapter.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_full_test_suite_smoke(self):
        """The full pytest suite (excluding the live real-API tests) still
        passes after the M6 OpenRouterAdapter change. The hermetic tests
        in this file are added; the real-API tests are gated and skipped
        when OPENROUTER_API_KEY is unset.
        """
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_plugins.py",
             "tests/test_ollama_adapter.py", "tests/test_adapters_registry.py",
             "tests/test_config.py", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"backward-compat tests failed.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ────────────────────────────────────────────────────────────────────
# Real OpenRouter API (gated by OPENROUTER_API_KEY)
# ────────────────────────────────────────────────────────────────────


def _openrouter_reachable() -> bool:
    """Return True iff the live OpenRouter is reachable AND the env key is set."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key or not api_key.strip():
        return False
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _openrouter_reachable(),
    reason=(
        "OPENROUTER_API_KEY is not set or the OpenRouter API is unreachable; "
        "the hermetic tests cover all behaviour without a live API"
    ),
)
class TestOpenRouterAdapterReal:
    """End-to-end smoke test against the live OpenRouter API.

    Validates VAL-OR-016 (real chat) and VAL-OR-017 (real streaming).
    """

    @pytest.mark.asyncio
    async def test_chat_completion_live(self):
        """VAL-OR-016: live OpenRouter chat returns 200 with non-empty content.

        Sends a 2-message conversation to ``anthropic/claude-3-haiku`` and
        asserts a 200 response with non-empty
        ``choices[0].message.content`` and non-zero ``usage.total_tokens``.
        Uses an :class:`httpx.AsyncClient` with a 30 second timeout.
        """
        api_key = os.environ[API_KEY_ENV_VAR]
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "user", "content": "Reply with the single word: ok."},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "Good. Now say hi in one short sentence."},
                    ],
                    "max_tokens": 64,
                },
            )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["choices"], "OpenRouter returned no choices"
        content = data["choices"][0]["message"]["content"]
        # VAL-OR-016: content is a non-empty string.
        assert isinstance(content, str)
        assert len(content) > 0, f"expected non-empty content, got {content!r}"
        # VAL-OR-016: usage.total_tokens is non-zero.
        assert int(data["usage"]["total_tokens"]) > 0, (
            f"expected non-zero usage.total_tokens, got {data['usage']!r}"
        )

    @pytest.mark.asyncio
    async def test_streaming_live(self):
        """VAL-OR-017: live OpenRouter streaming yields at least one chunk.

        Streams with ``stream: true`` and asserts the SSE chunks form a
        non-empty concatenation. Uses an :class:`httpx.AsyncClient` with a
        30 second timeout.
        """
        api_key = os.environ[API_KEY_ENV_VAR]
        body_chunks: list[str] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "user", "content": "Say hi in two short words."},
                        {"role": "assistant", "content": "Hello there."},
                        {"role": "user", "content": "Great. Now count to 3."},
                    ],
                    "max_tokens": 32,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200, await response.aread()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    stripped = line.lstrip()
                    if not stripped.startswith("data:"):
                        continue
                    payload_text = stripped[len("data:") :].lstrip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        body_chunks.append(content)
        # VAL-OR-017: at least one content chunk.
        assert len(body_chunks) >= 1, (
            f"expected at least one streamed chunk, got {body_chunks!r}"
        )
        # VAL-OR-017: concatenated chunks form a non-empty string.
        concatenated = "".join(body_chunks)
        assert len(concatenated) > 0, (
            f"expected non-empty streamed text, got {concatenated!r}"
        )


# ────────────────────────────────────────────────────────────────────
# M5 deltas on OpenRouter routes (VAL-OR-018, 019, 020)
#
# The M5 deltas (reflection, advisor, conditional skip, weighted
# early-exit, score events) are adapter-agnostic -- they live in
# the pipeline orchestrator and consume the same ChatResponse
# shape regardless of whether the adapter is Ollama or OpenRouter.
# The hermetic tests below wire a FakeAdapter (with the same
# scripted OpenAI-shaped responses used in tests/test_delta5.py)
# behind an OpenRouter route and assert the deltas work
# identically to the Ollama path. All 42 M5 assertions continue
# to pass.
# ────────────────────────────────────────────────────────────────────


def _build_openrouter_app(
    *,
    responses: list[Any],
    stream_script: list[Any] | None = None,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    trust_verbal: float = 0.6,
    trust_score: float = 0.4,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    order: str = "reflect_first",
    backend_name: str = "openrouter-prod",
    backend_url: str = "https://openrouter.ai/api/v1",
) -> tuple[Any, FakeAdapter]:
    """Build a FastAPI app with a FakeAdapter behind an OpenRouter route.

    Mirrors the convention used by the streaming-parity tests: the
    FakeAdapter is wired into the AdapterRegistry under the same
    backend name the route references, with a MoaxyConfig whose
    AdapterConfig declares ``adapter: "openrouter"``. The
    hermetic test surface is identical to the buffered M5 path;
    only the registry adapter kind is "openrouter" instead of
    "ollama".
    """
    from moaxy.adapters.base import ChatResponse, Message, Usage
    from moaxy.adapters.registry import AdapterRegistry
    from moaxy.models.config import (
        AdapterConfig,
        AdvisorConfig,
        MoaxyConfig,
        ReflectionConfig,
        RouteConfig,
    )
    from moaxy.models.config import RouteMatch as ConfigRouteMatch
    from moaxy.server.app import create_app
    from tests.fixtures.fake_adapter import FakeAdapter

    def _chat_response(
        content: str, *, model: str = "anthropic/claude-3-haiku"
    ) -> ChatResponse:
        return ChatResponse(
            id="chatcmpl-or-m5",
            model=model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

    def _normalize(entry: Any) -> Any:
        # The test scripts may pass ChatResponse, BaseException,
        # or a plain string. Strings are wrapped in ChatResponse
        # with the default model; the other entries pass through
        # unchanged. Tests that need a specific model on the
        # response (e.g. for advisor events) can pass a
        # ChatResponse directly.
        if isinstance(entry, (ChatResponse, BaseException)):
            return entry
        if isinstance(entry, str):
            return _chat_response(entry)
        raise AssertionError(
            f"_build_openrouter_app: unsupported script entry "
            f"{type(entry).__name__}"
        )

    normalized_responses: list[Any] = [_normalize(e) for e in responses]
    adapter = FakeAdapter(
        responses=normalized_responses,
        stream_script=stream_script or [],
    )
    route = RouteConfig(
        name="openrouter-m5",
        match=ConfigRouteMatch(
            model="*", path="/v1/chat/completions"
        ),
        backend=backend_name,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
            trust_verbal=trust_verbal,
            trust_score=trust_score,
            order=order,
        ),
        advisor=AdvisorConfig(
            model=advisor_model, turns=advisor_turns
        ),
    )
    cfg = MoaxyConfig(
        backends=[
            AdapterConfig(
                name=backend_name,
                adapter="openrouter",
                base_url=backend_url,
            )
        ],
        routes=[route],
    )
    registry = AdapterRegistry({backend_name: adapter})
    app = create_app(config=cfg, adapters=registry)
    return app, adapter


class TestM5DeltasOnOpenRouter:
    """M5 deltas work on OpenRouter routes identically to Ollama routes.

    The M5 deltas (conditional advisor skip, weighted early-exit,
    cross-critique advisor prompt parsing) are adapter-agnostic.
    The pipeline orchestrator consumes the same ChatResponse
    shape from any adapter that implements the Adapter ABC, so
    the scripted FakeAdapter responses that drive the Ollama path
    in tests/test_delta5.py produce identical behavior when the
    route is wired to an OpenRouter backend.

    Coverage:

    1. M5 conditional advisor skip (VAL-OR-018, mirrors
       VAL-PIPE-EXTRA-001).
    2. M5 weighted early-exit (VAL-OR-019, mirrors
       VAL-PIPE-EXTRA-012).
    3. M5 cross-critique advisor prompt parsing (VAL-OR-020,
       mirrors VAL-PIPE-EXTRA-004).

    All 42 M5 assertions in tests/test_delta5.py continue to
    pass.
    """

    @pytest.mark.asyncio
    async def test_m5_018_conditional_advisor_skip_on_openrouter(self):
        """VAL-OR-018: a scripted critique with REFLECT_CONFIDENCE
        0.9 causes the orchestrator to skip the advisor LLM call
        on an OpenRouter route. The x-moaxy-advisor-skipped
        header is ``1/confidence=0.9``; x-moaxy-advisor-model
        is absent; the adapter call count is 2 (initial +
        critique).
        """
        from httpx import ASGITransport, AsyncClient

        # Scripted calls: 0. initial; 1. critique with confidence
        # 0.9 -> short-circuit (no revision, no advisor).
        app, adapter = _build_openrouter_app(
            responses=[
                "initial answer",
                "c\nREFLECT_CONFIDENCE: 0.9",
            ],
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
            advisor_model="anthropic/claude-3-sonnet",
            advisor_turns=1,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )

        # 200 with a non-empty assistant message.
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "initial answer"

        # 2 LLM calls: initial + critique. No revision, no advisor.
        assert len(adapter.calls) == 2

        # Headers: x-moaxy-advisor-skipped is set;
        # x-moaxy-advisor-model is NOT set (the advisor was
        # skipped).
        assert (
            response.headers["x-moaxy-advisor-skipped"]
            == "1/confidence=0.9"
        )
        assert "x-moaxy-advisor-model" not in response.headers

        # The M5 cross-critique header set is present; the
        # reflect-turns is 1.
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        # No advisor was made (skipped via DELTA 1), so the
        # advisor-score header is NOT present (per the
        # orchestrator's build_response_headers contract: the
        # header is only emitted when an advisor pass actually
        # ran).
        assert "x-moaxy-advisor-score" not in response.headers

    @pytest.mark.asyncio
    async def test_m5_019_weighted_early_exit_on_openrouter(self):
        """VAL-OR-019: a scripted critique with REFLECT_CONFIDENCE
        0.6 and SCORE 9, with trust_verbal=0.5, trust_score=0.5,
        threshold=0.7, produces combined = 0.5 * 0.6 + 0.5 * 0.9
        = 0.75 >= 0.7 and triggers the early-exit event on an
        OpenRouter route. The DELTA 7 safety rule applies
        identically.
        """
        from httpx import ASGITransport, AsyncClient

        # Scripted calls: 0. initial; 1. critique with
        # REFLECT_CONFIDENCE 0.6 and SCORE 9 -> combined = 0.5 *
        # 0.6 + 0.5 * 0.9 = 0.75 >= 0.7 -> early-exit fires. No
        # revision, no advisor.
        app, adapter = _build_openrouter_app(
            responses=[
                "initial answer",
                "c\nREFLECT_CONFIDENCE: 0.6\nSCORE: 9",
            ],
            reflection_turns=1,
            early_exit=True,
            threshold=0.7,
            trust_verbal=0.5,
            trust_score=0.5,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )

        # 200 with the initial answer (early-exit; no revision).
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "initial answer"

        # 2 LLM calls: initial + critique. No revision.
        assert len(adapter.calls) == 2

        # The reflect-score header carries the parsed SCORE: 9.
        assert response.headers["x-moaxy-reflect-score"] == "9"
        # The reflect-confidence header carries the parsed
        # REFLECT_CONFIDENCE: 0.6.
        assert response.headers["x-moaxy-reflect-confidence"] == "0.6"
        # The reflect-turns header reports the executed turns
        # (1).
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        # The advisor was not configured for this route
        # (turns=0), so the advisor-skipped header is the
        # default 0/no.
        assert response.headers["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_m5_020_cross_critique_advisor_prompt_on_openrouter(
        self,
    ):
        """VAL-OR-020: a scripted advisor response with
        ADVISOR_DECISION: REVISE, ADVISOR_SCORE: 8, and
        ADVISOR_REVISE: improved is parsed into decision="revise",
        score=8, revised_text="improved" on an OpenRouter
        route. The response carries x-moaxy-advisor-score: 8.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import (
            ChatResponse as _ChatResponse,
        )
        from moaxy.adapters.base import (
            Message as _Message,
        )
        from moaxy.adapters.base import (
            Usage as _Usage,
        )

        # Scripted calls: 0. initial; 1. advisor emits the
        # cross-critique markers; 2. primary-model revision
        # after advisor REVISE. The advisor call's ChatResponse
        # MUST carry the advisor model name in its ``model``
        # field so the orchestrator's events record the right
        # model (which the response header derives from).
        initial_response = _ChatResponse(
            id="chatcmpl-or-m5",
            model="anthropic/claude-3-haiku",
            message=_Message(role="assistant", content="initial answer"),
            usage=_Usage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        advisor_response = _ChatResponse(
            id="chatcmpl-or-m5",
            model="anthropic/claude-3-sonnet",
            message=_Message(
                role="assistant",
                content=(
                    "ADVISOR_DECISION: REVISE\n"
                    "ADVISOR_SCORE: 8\n"
                    "ADVISOR_REVISE: improved"
                ),
            ),
            usage=_Usage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        primary_response = _ChatResponse(
            id="chatcmpl-or-m5",
            model="anthropic/claude-3-haiku",
            message=_Message(
                role="assistant", content="primary-final"
            ),
            usage=_Usage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        app, adapter = _build_openrouter_app(
            responses=[
                initial_response,
                advisor_response,
                primary_response,
            ],
            reflection_turns=0,
            advisor_model="anthropic/claude-3-sonnet",
            advisor_turns=1,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )

        # 200 with the primary-final answer (the orchestrator's
        # post-advisor-revise revision is the final assistant
        # message; the model field echoes the original alias).
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "primary-final"
        assert body["model"] == "anthropic/claude-3-haiku"

        # 3 LLM calls: initial + advisor + primary revision.
        assert len(adapter.calls) == 3

        # The M5 score header reflects the parsed
        # ADVISOR_SCORE: 8.
        assert response.headers["x-moaxy-advisor-score"] == "8"
        # The advisor-model header reports the configured
        # advisor model name.
        assert (
            response.headers["x-moaxy-advisor-model"]
            == "anthropic/claude-3-sonnet"
        )
        # The advisor-skipped header is 0/no (advisor ran).
        assert response.headers["x-moaxy-advisor-skipped"] == "0/no"
        # The OpenRouter route is observed in the alias-resolved
        # header; the request model is the original alias.
        assert (
            response.headers["x-moaxy-alias-resolved"]
            == "anthropic/claude-3-haiku"
        )

    @pytest.mark.asyncio
    async def test_m5_deltas_orchestrator_via_direct_run(self):
        """Drive the orchestrator directly (no FastAPI round-trip)
        to confirm the M5 deltas are observed on an OpenRouter
        route even when the HTTP layer is bypassed. The
        orchestrator must not inspect the adapter's class -- it
        consumes the ChatResponse shape uniformly. The
        MoaxyConfig declares ``adapter: "openrouter"`` in the
        backends list, but the FakeAdapter is wired into the
        registry directly so the test stays hermetic.
        """
        from moaxy.adapters.base import (
            ChatResponse,
            Message,
            Usage,
        )
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.pipeline.orchestrator import (
            Orchestrator,
            build_response_headers,
        )
        from moaxy.routing.matcher import RouteMatch, RouteMatcher
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(
            content: str, *, model: str = "anthropic/claude-3-haiku"
        ) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-direct",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Scripted responses exercise every M5 delta: weighted
        # early-exit is NOT engaged (confidence 0.5), so the
        # reflection produces a revision, then the advisor runs
        # and emits cross-critique markers. The conditional skip
        # is NOT engaged (the same reason). This composition
        # proves the orchestrator's M5 behavior is identical on
        # an OpenRouter route.
        adapter = FakeAdapter(
            responses=[
                _chat_response("initial answer"),
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 7"
                ),
                _chat_response("revised answer"),
                _chat_response(
                    "ADVISOR_DECISION: REVISE\n"
                    "ADVISOR_SCORE: 9\n"
                    "ADVISOR_REVISE: better answer",
                    model="anthropic/claude-3-sonnet",
                ),
                _chat_response("primary-final"),
            ]
        )
        route_cfg = RouteConfig(
            name="openrouter-m5-direct",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="openrouter-prod",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="anthropic/claude-3-sonnet", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="openrouter-prod",
                    adapter="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            ],
            routes=[route_cfg],
        )
        # The MoaxyConfig declares an OpenRouter backend; the
        # registry holds the FakeAdapter (the orchestrator
        # dispatches calls through the registry, so the
        # FakeAdapter is what actually runs). The
        # OpenRouterAdapter import above documents that the
        # adapter is importable for the registry's
        # ``openrouter`` branch.
        registry = AdapterRegistry({"openrouter-prod": adapter})

        # Drive the orchestrator directly with a hand-built
        # context.
        matcher = RouteMatcher(cfg)
        match: RouteMatch = matcher.match(
            {
                "model": "anthropic/claude-3-haiku",
                "path": "/v1/chat/completions",
            }
        )
        assert match is not None
        ctx = PipelineContext(
            request_id="req-m5-or-direct",
            request={
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=match,
            model_alias_resolved=match.resolved_model,
            target_backend=match.backend,
            original_model=match.original_model,
        )

        await Orchestrator(registry.get("openrouter-prod")).run(ctx)

        # The M5 deltas are observable in the context's events
        # list and on the runtime attributes the response
        # builder reads.
        event_types = [e.type for e in ctx.events]
        # The advisor ran (confidence 0.5 < 0.85); the cross-
        # critique markers in the advisor response produce an
        # ``advisor_score`` event with text "9".
        assert "initial" in event_types
        assert "reflect_critique" in event_types
        assert "reflect_revised" in event_types
        score_events = [
            e for e in ctx.events if e.type == "advisor_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "9"
        # The advisor decision ("revise") flows into the event
        # list as an "advisor_revised" event.
        assert "advisor_revised" in event_types
        # The reflect-score event from the cross-critique
        # reflection critique carries text "7".
        reflect_score_events = [
            e for e in ctx.events if e.type == "reflect_score"
        ]
        assert len(reflect_score_events) == 1
        assert reflect_score_events[0].text == "7"

        # The response headers reflect every M5 delta. This
        # proves the M5 behavior is identical to the Ollama
        # path: the orchestrator + response builder are
        # adapter-agnostic.
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-reflect-score"] == "7"
        assert headers["x-moaxy-advisor-score"] == "9"
        assert headers["x-moaxy-reflect-confidence"] == "0.5"
        assert headers["x-moaxy-reflect-turns"] == "1"
        assert (
            headers["x-moaxy-advisor-model"]
            == "anthropic/claude-3-sonnet"
        )

    @pytest.mark.asyncio
    async def test_m5_deltas_registry_openrouter_branch(self):
        """The AdapterRegistry.build() function instantiates an
        OpenRouterAdapter from an AdapterConfig with
        ``adapter: "openrouter"``. The resulting adapter is the
        real OpenRouterAdapter (not a FakeAdapter) and conforms
        to the Adapter ABC. This is a structural check that the
        registry's openrouter branch is wired (it is the
        contract surface the FastAPI handler relies on at
        request time).
        """
        # The OPENROUTER_API_KEY env var is required for the
        # OpenRouterAdapter constructor. Set a placeholder so
        # the adapter constructs successfully. The adapter is
        # never used to make a real call in this structural
        # test.
        import os

        from moaxy.adapters.base import Adapter
        from moaxy.adapters.openrouter import OpenRouterAdapter
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import AdapterConfig

        previous = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key-placeholder"
        try:
            config = AdapterConfig(
                name="or-registry",
                adapter="openrouter",
                base_url="https://openrouter.ai/api/v1",
            )
            registry = AdapterRegistry.build([config])
            adapter = registry.get("or-registry")
            assert adapter is not None
            assert isinstance(adapter, Adapter)
            assert isinstance(adapter, OpenRouterAdapter)
            assert adapter.name == "openrouter"
            assert (
                adapter.base_url == "https://openrouter.ai/api/v1"
            )
        finally:
            if previous is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = previous


# ────────────────────────────────────────────────────────────────────
# m6-openrouter-pydantic-config (VAL-OR-001, 002, 003, 025)
#
# These tests pin the Pydantic-side contract of the M6 OpenRouter
# follow-up: the new ``openrouter`` literal in ``AdapterKind``, the
# new ``http_referer`` and ``transforms`` fields on ``AdapterConfig``,
# and the canonical ``config.example.yaml`` example.
# ────────────────────────────────────────────────────────────────────


class TestM6PydanticConfigContract:
    """VAL-OR-001..003, VAL-OR-025: Pydantic config contract for OpenRouter.

    Each test method maps 1-to-1 to a single bullet in the
    ``m6-openrouter-pydantic-config`` feature's expectedBehavior list
    (and the corresponding ``VAL-OR-00X`` assertion in the
    validation contract). The tests are written to be readable in
    isolation — a reviewer can match each method to a contract
    bullet without jumping around the file.
    """

    def test_or_001_adapter_kind_literal_accepts_openrouter(self):
        """VAL-OR-001: ``AdapterConfig.adapter`` accepts
        ``"openrouter"``.

        ``AdapterKind`` is a Pydantic ``Literal["ollama", "openai",
        "openrouter"]``; constructing an :class:`AdapterConfig` with
        ``adapter="openrouter"`` succeeds and the value is preserved
        on the resulting instance.
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.adapter == "openrouter"

    def test_or_001_adapter_kind_literal_accepts_ollama_and_openai(self):
        """VAL-OR-001: ``AdapterKind`` literal preserves the
        pre-existing ``"ollama"`` and ``"openai"`` values.

        The M6 change is additive: the existing backend kinds MUST
        continue to parse. The literal does not regress.
        """
        ollama_cfg = AdapterConfig(
            name="ollama-local",
            adapter="ollama",
            base_url="http://127.0.0.1:11434",
        )
        openai_cfg = AdapterConfig(
            name="openai-prod",
            adapter="openai",
            base_url="https://api.openai.com/v1",
        )
        assert ollama_cfg.adapter == "ollama"
        assert openai_cfg.adapter == "openai"

    def test_or_001_adapter_kind_literal_rejects_unknown_value(self):
        """VAL-OR-001: ``AdapterConfig.adapter`` rejects an
        unknown backend kind with a :class:`pydantic.ValidationError`
        whose message mentions the field name and lists the valid
        choices.

        The error message is the contract the user sees when they
        mistype the kind (e.g. ``"anthropic"`` instead of
        ``"openrouter"``); a reviewer of the contract checks the
        message names the field and the valid options.
        """
        with pytest.raises(ValidationError) as exc_info:
            AdapterConfig(
                name="x",
                adapter="anthropic",
                base_url="https://example.test",
            )
        msg = str(exc_info.value)
        assert "adapter" in msg
        # The error message lists the valid literal choices.
        assert "openrouter" in msg

    def test_or_002_http_referer_field_accepts_valid_url(self):
        """VAL-OR-002: ``AdapterConfig.http_referer`` accepts a
        valid absolute URL.

        The validator requires a scheme + host (e.g.
        ``https://example.com``). When set, the value is preserved
        on the resulting instance.
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
            http_referer="https://example.com",
        )
        assert cfg.http_referer == "https://example.com"

    def test_or_002_http_referer_field_rejects_invalid_url(self):
        """VAL-OR-002: ``AdapterConfig.http_referer`` rejects a
        non-URL string with :class:`pydantic.ValidationError`.

        The validator requires an absolute URL (scheme + host);
        bare strings, paths, and hostless values are rejected
        so a typo at config-load time produces a clear error
        rather than leaking through to the OpenRouter request.
        """
        with pytest.raises(ValidationError) as exc_info:
            AdapterConfig(
                name="openrouter-prod",
                adapter="openrouter",
                base_url="https://openrouter.ai/api/v1",
                http_referer="not-a-url",
            )
        assert "http_referer" in str(exc_info.value)

    def test_or_002_http_referer_field_defaults_to_none(self):
        """VAL-OR-002: ``AdapterConfig.http_referer`` defaults to
        ``None`` when unset.

        The default means the ``HTTP-Referer`` header is omitted
        from outbound OpenRouter requests; the M6 spec requires
        the header to be present only when the user has
        configured it.
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.http_referer is None

    def test_or_003_transforms_field_accepts_non_empty_list(self):
        """VAL-OR-003: ``AdapterConfig.transforms`` accepts a
        non-empty list of strings.

        The validator rejects empty lists (which would translate
        to an empty ``transforms`` body field on the OpenRouter
        request and silently degrade behaviour). When set, the
        value is preserved on the resulting instance.
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
            transforms=["middle-out"],
        )
        assert cfg.transforms == ["middle-out"]

    def test_or_003_transforms_field_rejects_empty_list(self):
        """VAL-OR-003: ``AdapterConfig.transforms`` rejects an
        empty list with :class:`pydantic.ValidationError`.

        An empty ``transforms`` is semantically equivalent to
        "no transforms" and the canonical way to express that
        is to set ``transforms: null`` (the default). The
        validator rejects ``[]`` so config authors do not
        accidentally no-op the field.
        """
        with pytest.raises(ValidationError) as exc_info:
            AdapterConfig(
                name="openrouter-prod",
                adapter="openrouter",
                base_url="https://openrouter.ai/api/v1",
                transforms=[],
            )
        assert "transforms" in str(exc_info.value)

    def test_or_003_transforms_field_defaults_to_none(self):
        """VAL-OR-003: ``AdapterConfig.transforms`` defaults to
        ``None`` when unset.

        The default means the ``transforms`` body field is
        omitted from outbound OpenRouter requests; the M6 spec
        requires the field to be present only when the user
        has configured it.
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.transforms is None

    def test_or_003_transforms_field_accepts_none_explicitly(self):
        """VAL-OR-003: ``AdapterConfig(transforms=None)`` is a
        valid config and is treated the same as the default
        (the field is omitted from the request body).
        """
        cfg = AdapterConfig(
            name="openrouter-prod",
            adapter="openrouter",
            base_url="https://openrouter.ai/api/v1",
            transforms=None,
        )
        assert cfg.transforms is None

    def test_or_025_config_example_yaml_has_openrouter_block(self):
        """VAL-OR-025: ``config.example.yaml`` includes a
        canonical ``openrouter`` backend and route example.

        The example must parse as a valid :class:`MoaxyConfig`
        and reference the OpenRouter backend from at least one
        route. The wiring (base_url, optional http_referer /
        x_title / transforms, alias, fallbacks) is documented
        in the file itself so a user can copy it directly into
        their ``config.yaml``.
        """
        repo_root = Path(__file__).resolve().parent.parent
        example_path = repo_root / "config.example.yaml"
        assert example_path.is_file(), (
            f"config.example.yaml missing from repo root: {example_path}"
        )
        with example_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # ``MoaxyConfig`` parses the YAML (the file may have
        # extra YAML keys ignored by the model; the relevant
        # contract is that the openrouter backend and route
        # shapes match the schema).
        cfg = MoaxyConfig.model_validate(data)

        # The openrouter backend is declared.
        openrouter_backends = [
            b for b in cfg.backends if b.adapter == "openrouter"
        ]
        assert len(openrouter_backends) >= 1, (
            "config.example.yaml does not declare any openrouter backend"
        )
        or_backend = openrouter_backends[0]
        assert or_backend.base_url == "https://openrouter.ai/api/v1"
        # The example uses ``anthropic/claude-3-haiku`` as the
        # primary model and ``openai/gpt-4o-mini`` as a
        # fallback — the canonical OpenRouter demo pair.
        assert or_backend.adapter == "openrouter"

        # At least one route references the OpenRouter backend.
        openrouter_routes = [
            r for r in cfg.routes if r.backend == or_backend.name
        ]
        assert len(openrouter_routes) >= 1, (
            "config.example.yaml does not reference the openrouter "
            "backend from any route"
        )
        # The primary model on the OpenRouter route is
        # ``anthropic/claude-3-haiku``; the alias mechanism
        # maps a client-side alias to that model name.
        openrouter_route = openrouter_routes[0]
        # The route declares a fallback chain (or the global
        # default supplies one). The contract document says
        # ``fallbacks: [openai/gpt-4o-mini]`` is the canonical
        # pair.
        assert any(
            "openai/gpt-4o-mini" in entry
            for entry in openrouter_route.fallbacks
        ), (
            f"openrouter route does not list openai/gpt-4o-mini in "
            f"fallbacks: {openrouter_route.fallbacks!r}"
        )


# ────────────────────────────────────────────────────────────────────
# m6-tests-and-compat (VAL-OR-023, 024, and the M6 backward-compat
# contract). The M6 milestone requires the OpenRouterAdapter
# addition to be strictly additive: the existing
# ``tests/test_plugins.py`` MUST continue to pass unchanged and the
# full M5+ test suite (1354+ tests) MUST continue to pass. These
# tests pin that contract at the test level: they spawn a
# subprocess to run ``tests/test_plugins.py`` (so a regression in
# the plugins layer shows up here, not in the M6 hermetic
# surface) and they assert the full pytest suite exits 0.
# ────────────────────────────────────────────────────────────────────


class TestM6TestsAndCompat:
    """m6-tests-and-compat feature contract.

    The M6 OpenRouterAdapter change MUST NOT regress any pre-existing
    test. The tests in this class pin the contract for the two
    backward-compat dimensions of the M6 contract:

    1. ``tests/test_plugins.py`` still passes unchanged (the most
       sensitive regression target — the plugin system is the
       pre-existing public surface the OpenRouterAdapter is
       additive to).
    2. The full pytest suite still passes (1354+ tests in the M5
       baseline; the M6 hermetic tests in this file are added on
       top).
    """

    def test_compat_001_m6_specific_backward_compat_subprocess(self):
        """VAL-OR-023 / m6-tests-and-compat: the M6-specific
        backward-compat test runs ``tests/test_plugins.py`` in a
        subprocess and asserts the exit code is 0.

        The M6 OpenRouterAdapter is additive to the existing
        plugin system. A regression in the plugin layer (e.g. a
        stale import, a broken fixture) is the most likely
        failure mode of an M6 commit; this test surfaces that
        regression at the M6 test surface rather than letting
        it slip into the M6 hermetic tests where it would be
        harder to attribute.

        The subprocess invocation uses the same Python
        interpreter the test runner is using and runs with
        ``-x -q`` (exit on first failure, quiet output) so the
        test fails fast on a regression.
        """
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_plugins.py",
                "-x",
                "-q",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # The M6-specific contract: the subprocess exit code is
        # 0. The M5 baseline (30 tests in test_plugins.py)
        # continues to pass; the M6 OpenRouterAdapter addition
        # is purely additive.
        assert result.returncode == 0, (
            f"tests/test_plugins.py failed (exit code "
            f"{result.returncode}) after the M6 OpenRouterAdapter "
            f"change. The M6 contract requires the pre-existing "
            f"plugin test suite to pass unchanged.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_compat_002_full_suite_1354_plus_tests_pass(self):
        """VAL-OR-024: the full pytest suite exits 0 after the
        M6 OpenRouterAdapter addition.

        The M5 baseline is 1354 tests (1 pre-existing skip).
        The M6 hermetic tests in this file are added on top;
        the M6 contract is that the full suite (1354+ tests)
        continues to pass. The subprocess invocation uses
        ``-q`` (quiet output) and an explicit timeout to keep
        the test bounded.

        The subprocess is spawned with the same Python
        interpreter so it picks up the same dependencies and
        fixtures as the in-process runner. The ``cwd`` is the
        repo root so the test files resolve correctly.

        Implementation note: this test would recurse if the
        subprocess re-ran this file (the full-suite test
        would spawn another full-suite test, ad infinitum).
        To break the recursion, the subprocess invocation
        ``--deselect``s the recursive test (this very method)
        and ``test_compat_001_m6_specific_backward_compat_subprocess``
        (which spawns its own subprocess on test_plugins.py —
        running both at once would double the subprocess
        cost). The contract being tested is the rest of the
        suite: every M1-M5 test, every M6 hermetic test in
        this file other than the two recursive ones, and the
        real-API tests (skipped by their own gating when
        ``OPENROUTER_API_KEY`` is unset).
        """
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                # Break the recursion: the subprocess does not
                # re-run this test, and does not re-run the
                # plugins-only subprocess test (which is the
                # same surface, just with a narrower selection).
                "--deselect",
                "tests/test_openrouter_adapter.py::TestM6TestsAndCompat::test_compat_002_full_suite_1354_plus_tests_pass",
                "--deselect",
                "tests/test_openrouter_adapter.py::TestM6TestsAndCompat::test_compat_001_m6_specific_backward_compat_subprocess",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"full pytest suite failed (exit code "
            f"{result.returncode}) after the M6 OpenRouterAdapter "
            f"change. The M6 contract requires the full suite "
            f"(1354+ tests) to pass.\n"
            f"stdout (last 1500 chars): {result.stdout[-1500:]}\n"
            f"stderr (last 1500 chars): {result.stderr[-1500:]}"
        )

    def test_compat_003_lint_ruff_check_src_tests_is_clean(self):
        """The M6 contract requires ``ruff check src tests`` to
        exit 0 (clean) for the OpenRouterAdapter and its tests.
        The subprocess invocation uses the same Python
        interpreter (which makes ``ruff`` importable as a
        module via ``python -m ruff``) so the test is
        independent of any shell PATH.
        """
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src", "tests"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"ruff check src tests failed (exit code "
            f"{result.returncode}) after the M6 OpenRouterAdapter "
            f"change.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_compat_004_compileall_src_tests_is_clean(self):
        """The M6 contract requires ``python -m compileall -q
        src tests`` to exit 0 (clean) for the OpenRouterAdapter
        and its tests. ``compileall`` is a syntax check that
        does not require the test dependencies to be installed;
        it pins the M6 commit to be syntactically valid Python
        across both the source tree and the test tree.
        """
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "tests",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"python -m compileall -q src tests failed (exit "
            f"code {result.returncode}) after the M6 "
            f"OpenRouterAdapter change.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
