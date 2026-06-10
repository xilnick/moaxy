"""Tests for the OllamaAdapter.

Covers:
- Adapter ABC inheritance and the AdapterKind contract.
- Successful chat() POSTs to ``${base_url}/v1/chat/completions`` and parses
  the OpenAI-compatible response into a :class:`ChatResponse`.
- Error mapping: 4xx/5xx → UpstreamError, timeout → UpstreamTimeoutError,
  connection error → UpstreamUnavailableError.
- ChatResponse / Usage dataclass invariants and the UsageAccumulator sum.
- Stream() smoke check using a fake httpx transport.
- Class-level ``name`` attribute is ``"ollama"`` so a future AdapterRegistry
  can key by adapter name.
- Real-Ollama integration (gated on Ollama reachability).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    Usage,
    UsageAccumulator,
)
from moaxy.adapters.ollama import OllamaAdapter


# ────────────────────────────────────────────────────────────────────
# AdapterKind / base types
# ────────────────────────────────────────────────────────────────────


class TestAdapterBaseTypes:
    """The base types the OllamaAdapter depends on must exist and behave."""

    def test_chat_response_defaults(self):
        m = Message(role="assistant", content="hi")
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        r = ChatResponse(id="x", model="m", message=m, usage=u)
        assert r.id == "x"
        assert r.model == "m"
        assert r.message.role == "assistant"
        assert r.message.content == "hi"
        assert r.usage.total_tokens == 3
        assert r.finish_reason is None

    def test_chat_response_finish_reason_preserved(self):
        r = ChatResponse(
            id="x",
            model="m",
            message=Message(role="assistant", content=""),
            usage=Usage(),
            finish_reason="length",
        )
        assert r.finish_reason == "length"

    def test_usage_default_zero(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_usage_accumulator_sums_tokens(self):
        acc = UsageAccumulator()
        acc.add(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        acc.add(Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        acc.add(Usage(prompt_tokens=0, completion_tokens=7, total_tokens=7))
        snap = acc.snapshot()
        assert snap.prompt_tokens == 13
        assert snap.completion_tokens == 14
        assert snap.total_tokens == 27

    def test_usage_accumulator_handles_missing_total(self):
        """If the upstream omits total_tokens, snapshot still sums the parts."""
        acc = UsageAccumulator()
        acc.add(Usage(prompt_tokens=4, completion_tokens=6))  # total_tokens = 0
        snap = acc.snapshot()
        assert snap.prompt_tokens == 4
        assert snap.completion_tokens == 6
        assert snap.total_tokens == 10

    def test_usage_accumulator_reset(self):
        acc = UsageAccumulator()
        acc.add(Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10))
        acc.reset()
        snap = acc.snapshot()
        assert snap.prompt_tokens == 0
        assert snap.completion_tokens == 0
        assert snap.total_tokens == 0

    def test_upstream_error_carries_status(self):
        e = UpstreamError("boom", status_code=500)
        assert e.status_code == 500
        assert "boom" in str(e)
        assert e.body is None

    def test_upstream_error_carries_body(self):
        e = UpstreamError("oops", status_code=400, body='{"err":"bad"}')
        assert e.status_code == 400
        assert e.body == '{"err":"bad"}'

    def test_upstream_timeout_is_upstream_error_subclass(self):
        """Timeouts are a kind of upstream error; callers may catch either."""
        e = UpstreamTimeoutError("slow")
        assert isinstance(e, UpstreamError)
        assert e.status_code is None

    def test_upstream_unavailable_is_upstream_error_subclass(self):
        e = UpstreamUnavailableError("connect failed")
        assert isinstance(e, UpstreamError)
        assert e.status_code is None


# ────────────────────────────────────────────────────────────────────
# OllamaAdapter identity and construction
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterIdentity:
    """The adapter exposes a stable class-level name for the future registry."""

    def test_inherits_from_adapter_abc(self):
        assert issubclass(OllamaAdapter, Adapter)

    def test_class_name_is_ollama(self):
        assert OllamaAdapter.name == "ollama"

    def test_default_base_url(self):
        a = OllamaAdapter()
        assert a.base_url == "http://127.0.0.1:11434"

    def test_custom_base_url_strips_trailing_slash(self):
        a = OllamaAdapter(base_url="http://example.test:9000/")
        assert a.base_url == "http://example.test:9000"

    def test_endpoint_path_is_v1_chat_completions(self):
        a = OllamaAdapter(base_url="http://example.test:9000")
        assert a.endpoint == "http://example.test:9000/v1/chat/completions"

    def test_endpoint_works_with_base_url_already_having_path(self):
        a = OllamaAdapter(base_url="http://example.test:9000/api")
        assert a.endpoint == "http://example.test:9000/api/v1/chat/completions"

    def test_chat_is_coroutine_function(self):
        import inspect

        assert inspect.iscoroutinefunction(OllamaAdapter.chat)

    def test_stream_is_async_gen_function(self):
        """``stream`` is an ``async def`` generator (yields deltas)."""
        import inspect

        assert inspect.isasyncgenfunction(OllamaAdapter.stream)

    def test_close_is_coroutine_function(self):
        import inspect

        assert inspect.iscoroutinefunction(OllamaAdapter.close)


# ────────────────────────────────────────────────────────────────────
# Fake transport helpers
# ────────────────────────────────────────────────────────────────────


class _FakeTransport(httpx.AsyncBaseTransport):
    """In-process httpx transport that returns scripted responses."""

    def __init__(self, handler):
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


def _error_response(status_code: int, message: str = "upstream") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps({"error": {"message": message}}).encode("utf-8"),
    )


def _make_ollama_payload(
    *,
    content: str = "hello",
    model: str = "minimax-m2.7:cloud",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    total_tokens: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-abc123",
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
# Successful chat() — happy path with in-process transport
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterChatHappyPath:
    """A 200 response is parsed into a normalised ChatResponse."""

    @pytest.mark.asyncio
    async def test_chat_returns_chat_response(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return _json_response(_make_ollama_payload(content="hi back"))

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            timeout=5.0,
            _transport=transport,
        )
        try:
            response = await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            await adapter.close()

        assert isinstance(response, ChatResponse)
        assert response.id == "chatcmpl-abc123"
        assert response.model == "minimax-m2.7:cloud"
        assert response.message.role == "assistant"
        assert response.message.content == "hi back"
        assert response.usage.prompt_tokens == 7
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 10
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_posts_to_v1_chat_completions(self):
        seen: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["method"] = request.method
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return _json_response(_make_ollama_payload())

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=64,
                temperature=0.5,
            )
        finally:
            await adapter.close()

        assert seen["method"] == "POST"
        assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
        body = seen["body"]
        assert body["model"] == "minimax-m2.7:cloud"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_forwards_extra_kwargs(self):
        captured: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content.decode("utf-8")))
            return _json_response(_make_ollama_payload())

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "x"}],
                top_p=0.9,
                stop=["END"],
                stream=False,
            )
        finally:
            await adapter.close()

        assert captured["top_p"] == 0.9
        assert captured["stop"] == ["END"]
        assert captured["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_without_usage_returns_zero_usage(self):
        """Ollama's OpenAI endpoint sometimes omits ``usage``; the adapter
        must still return a valid ChatResponse with a zero Usage."""

        async def handler(_request: httpx.Request) -> httpx.Response:
            payload = _make_ollama_payload()
            payload.pop("usage", None)
            return _json_response(payload)

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            response = await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "x"}],
            )
        finally:
            await adapter.close()

        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0
        assert response.message.content == "hello"


# ────────────────────────────────────────────────────────────────────
# Error mapping
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterErrorMapping:
    """4xx/5xx, timeouts, and connection errors are mapped to typed exceptions."""

    @pytest.mark.asyncio
    async def test_4xx_raises_upstream_error_with_status(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(404, "model not found")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="nope",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

        assert exc_info.value.status_code == 404
        assert "model not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_5xx_raises_upstream_error_with_status(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(500, "internal error")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

        assert exc_info.value.status_code == 500
        assert "internal error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_4xx_preserves_body(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(400, "bad request")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

        assert exc_info.value.body is not None
        assert "bad request" in exc_info.value.body

    @pytest.mark.asyncio
    async def test_timeout_raises_upstream_timeout_error(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("the read operation timed out")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            timeout=1.0,
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamTimeoutError):
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_connect_error_raises_upstream_unavailable(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamUnavailableError):
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_remote_protocol_error_raises_upstream_unavailable(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("server closed connection")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamUnavailableError):
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises_upstream_error(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                content=b"<html>oops</html>",
            )

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamError) as exc_info:
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()

        assert exc_info.value.status_code == 200
        assert "decode" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()


# ────────────────────────────────────────────────────────────────────
# Timeout configuration
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterTimeout:
    """Per-call httpx timeout is respected."""

    @pytest.mark.asyncio
    async def test_default_timeout_is_set(self):
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=_FakeTransport(lambda r: _json_response(_make_ollama_payload())),
        )
        try:
            assert adapter.timeout == 30.0
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_custom_timeout_is_stored(self):
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            timeout=12.5,
            _transport=_FakeTransport(lambda r: _json_response(_make_ollama_payload())),
        )
        try:
            assert adapter.timeout == 12.5
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_call_timeout_raises_upstream_timeout_error(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("nope")

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            timeout=0.01,
            _transport=transport,
        )
        try:
            with pytest.raises(UpstreamTimeoutError):
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()


# ────────────────────────────────────────────────────────────────────
# Lifecycle: close() releases the client
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterLifecycle:
    """The adapter can be constructed and closed multiple times."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=_FakeTransport(lambda r: _json_response(_make_ollama_payload())),
        )
        await adapter.close()
        await adapter.close()
        # No exception means success.

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=_FakeTransport(lambda r: _json_response(_make_ollama_payload())),
        )
        await adapter.close()
        assert adapter._client is None


# ────────────────────────────────────────────────────────────────────
# Adapter ABC / stream() smoke test
# ────────────────────────────────────────────────────────────────────


class TestOllamaAdapterABCContract:
    """OllamaAdapter implements the Adapter ABC contract."""

    def test_ollama_adapter_isinstance_of_adapter(self):
        a = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=_FakeTransport(lambda r: _json_response(_make_ollama_payload())),
        )
        assert isinstance(a, Adapter)

    def test_adapter_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            Adapter()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self):
        # Build a minimal fake OpenAI-style streaming payload (server-sent
        # events style is not used by Ollama; instead it returns newline-
        # separated JSON chunks, each a partial chat.completion object).
        chunk_a = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "minimax-m2.7:cloud",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}
            ],
        }
        chunk_b = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "minimax-m2.7:cloud",
            "choices": [
                {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
            ],
        }
        body_text = json.dumps(chunk_a) + "\n" + json.dumps(chunk_b) + "\n"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/x-ndjson"},
                content=body_text.encode("utf-8"),
            )

        transport = _FakeTransport(handler)
        adapter = OllamaAdapter(
            base_url="http://127.0.0.1:11434",
            _transport=transport,
        )
        try:
            chunks: list[str] = []
            async for delta in adapter.stream(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(delta)
        finally:
            await adapter.close()

        assert "".join(chunks) == "Hello"


# ────────────────────────────────────────────────────────────────────
# Real Ollama integration (skipped if Ollama is unreachable)
# ────────────────────────────────────────────────────────────────────


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get("http://127.0.0.1:11434/api/tags")
            return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama is not reachable on 127.0.0.1:11434",
)
class TestOllamaAdapterReal:
    """End-to-end smoke test against the running Ollama instance.

    Validates the assertions the contract expects for VAL-CROSS-014 and
    VAL-CROSS-016 (long-context usage invariant).
    """

    @pytest.mark.asyncio
    async def test_real_chat_returns_a_well_formed_response(self):
        """The first request to a never-called model succeeds (VAL-CROSS-014).

        The cloud reasoning model can return empty ``content`` while burning
        all of ``max_tokens`` on internal reasoning, so we do not assert
        non-empty content; we assert the response is well-formed (id and
        model present, valid finish_reason, usage parsed).
        """
        adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", timeout=120.0)
        try:
            response = await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": "Reply with the single word: ok."}],
                max_tokens=20,
            )
        finally:
            await adapter.close()

        assert response.id, "real Ollama response missing id"
        assert response.model, "real Ollama response missing model"
        assert response.finish_reason in {"stop", "length", None}
        assert response.usage.completion_tokens >= 0

    @pytest.mark.asyncio
    async def test_real_long_context_usage_invariant(self):
        """VAL-CROSS-016: usage.total_tokens == prompt_tokens + completion_tokens.

        Cloud reasoning models may report ``prompt_tokens == 0`` while still
        completing successfully; what matters is the additive invariant on
        usage, not the absolute count.
        """
        word = "alpha "
        big_input = (word * 1000).strip()
        adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", timeout=120.0)
        try:
            response = await adapter.chat(
                model="minimax-m2.7:cloud",
                messages=[{"role": "user", "content": big_input}],
                max_tokens=20,
            )
        finally:
            await adapter.close()

        assert response.usage.completion_tokens >= 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    @pytest.mark.asyncio
    async def test_real_connect_error_when_ollama_down(self):
        # Use a clearly-unreachable port to force a connect error.
        adapter = OllamaAdapter(base_url="http://127.0.0.1:1", timeout=2.0)
        try:
            with pytest.raises(UpstreamUnavailableError):
                await adapter.chat(
                    model="minimax-m2.7:cloud",
                    messages=[{"role": "user", "content": "x"}],
                )
        finally:
            await adapter.close()
