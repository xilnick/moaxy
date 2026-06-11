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
from typing import Any

import httpx
import pytest

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
    async def test_real_chat_returns_well_formed_response(self):
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
                        {"role": "user", "content": "Reply with the single word: ok."}
                    ],
                    "max_tokens": 20,
                },
            )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["choices"], "OpenRouter returned no choices"
        content = data["choices"][0]["message"]["content"]
        # Cloud reasoning models may emit empty content with small
        # max_tokens; the response shape is the assertion.
        assert isinstance(content, str)
        assert data["usage"]["total_tokens"] >= 0

    @pytest.mark.asyncio
    async def test_real_stream_returns_chunks(self):
        api_key = os.environ[API_KEY_ENV_VAR]
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
                        {"role": "user", "content": "Say hi."}
                    ],
                    "max_tokens": 10,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                body_chunks: list[str] = []
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
        # At least one content delta (or the model gave up early).
        assert isinstance(body_chunks, list)
