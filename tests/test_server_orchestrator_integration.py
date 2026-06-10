"""End-to-end integration tests for the orchestrator wired into the HTTP route.

These tests prove that the :class:`~moaxy.pipeline.orchestrator.Orchestrator`
is correctly composed with the FastAPI ``/v1/chat/completions`` handler. The
tests are hermetic: a hand-rolled :class:`ScriptedAdapter` records every
backend call and returns scripted responses, mirroring the helpers in
``tests/test_orchestrator.py``. No in-process HTTP, no real Ollama.

The contract pinned here matches the validation contract sections
"Area: HTTP Server" (VAL-HTTP-017, 018, 019, 020, 022) and "Area:
Routing & Aliases" (VAL-RT-007, 008) for the data plane, and the
"Cross-Area Flows" entry VAL-CROSS-001 / VAL-CROSS-013.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
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
from moaxy.server.app import create_app
from moaxy.server.middleware import REQUEST_ID_HEADER

# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter — a programmable backend that records every call
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    Each entry in the script is either a :class:`ChatResponse` (success)
    or a :class:`BaseException` (raised). Calls are recorded in
    :attr:`calls` so tests can assert on the ``model`` and ``messages``
    the orchestrator forwarded.
    """

    name = "scripted"

    def __init__(self, script: list[Any]) -> None:
        self._script: list[Any] = list(script)
        self._index: int = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self._index >= len(self._script):
            raise AssertionError(
                f"ScriptedAdapter: no more scripted responses "
                f"(call #{self._index + 1} for model={model})"
            )
        entry = self._script[self._index]
        self._index += 1
        if isinstance(entry, BaseException):
            raise entry
        if not isinstance(entry, ChatResponse):
            raise AssertionError(
                f"ScriptedAdapter: script entry must be ChatResponse or "
                f"Exception, got {type(entry).__name__}"
            )
        return entry

    async def stream(  # pragma: no cover - streaming is M4
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        if False:
            yield ""

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


def _response(
    content: str,
    *,
    model: str = "minimax-m3:cloud",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    chatcmpl_id: str = "chatcmpl-test",
) -> ChatResponse:
    return ChatResponse(
        id=chatcmpl_id,
        model=model,
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        finish_reason=finish_reason,
    )


def _build_route_config(
    *,
    name: str = "reflective-coder",
    match_model: str = "*",
    backend: str = "olloma-local",
    aliases: dict[str, str] | None = None,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
) -> RouteConfig:
    return RouteConfig(
        name=name,
        match=ConfigRouteMatch(model=match_model, path="/v1/chat/completions"),
        backend=backend,
        aliases=aliases or {},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
        ),
        advisor=AdvisorConfig(model=advisor_model, turns=advisor_turns),
    )


def _build_app(
    adapter: Adapter,
    route: RouteConfig,
    *,
    backend_name: str = "olloma-local",
) -> Any:
    """Build a FastAPI app with the scripted adapter and route mounted."""
    cfg = MoaxyConfig(
        backends=[AdapterConfig(name=backend_name, adapter="ollama", base_url="http://x")],
        routes=[route],
    )
    registry = AdapterRegistry({backend_name: adapter})
    return create_app(config=cfg, adapters=registry)


# ────────────────────────────────────────────────────────────────────
# request_id + alias-resolved headers (VAL-HTTP-017, VAL-HTTP-022)
# ────────────────────────────────────────────────────────────────────


class TestHeaderPresence:
    """The four core moaxy headers are present on every response."""

    @pytest.mark.asyncio
    async def test_request_id_and_alias_resolved_headers_present(self):
        adapter = ScriptedAdapter(
            [_response("hi", model="minimax-m3:cloud")]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        app = _build_app(adapter, route)
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
        assert response.status_code == 200
        assert REQUEST_ID_HEADER in response.headers
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        # x-moaxy-reflect-turns and x-moaxy-reflect-confidence are
        # always present (default 0 / 0.0 on a passthrough route).
        assert "x-moaxy-reflect-turns" in response.headers
        assert "x-moaxy-reflect-confidence" in response.headers
        assert response.headers["x-moaxy-reflect-turns"] == "0"
        assert float(response.headers["x-moaxy-reflect-confidence"]) == 0.0

    @pytest.mark.asyncio
    async def test_request_id_is_distinct_across_requests(self):
        adapter = ScriptedAdapter(
            [_response("a", model="m"), _response("b", model="m")]
        )
        route = _build_route_config()
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "m1",
                    "messages": [{"role": "user", "content": "a"}],
                },
                headers={"Content-Type": "application/json"},
            )
            r2 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "m1",
                    "messages": [{"role": "user", "content": "b"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]


# ────────────────────────────────────────────────────────────────────
# turns=0 passthrough (VAL-PIPE-001, VAL-HTTP-019)
# ────────────────────────────────────────────────────────────────────


class TestReflectionTurnsZero:
    """A route with ``reflection.turns: 0`` is a single-call passthrough."""

    @pytest.mark.asyncio
    async def test_passthrough_returns_initial_response(self):
        adapter = ScriptedAdapter(
            [_response("initial answer", model="minimax-m3:cloud")]
        )
        route = _build_route_config(reflection_turns=0)
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "initial answer"
        # The header reflects the actual turn count (0).
        assert response.headers["x-moaxy-reflect-turns"] == "0"
        # The orchestrator made exactly one LLM call.
        assert len(adapter.calls) == 1

    @pytest.mark.asyncio
    async def test_passthrough_with_alias_echoes_original(self):
        adapter = ScriptedAdapter(
            [_response("ack", model="minimax-m3:cloud")]
        )
        route = _build_route_config(
            reflection_turns=0,
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        app = _build_app(adapter, route)
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
        assert response.status_code == 200
        body = response.json()
        # The response's model field echoes the alias.
        assert body["model"] == "coder-pro"
        # The header reveals the resolved real name.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        # The adapter received the RESOLVED model name.
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_passthrough_response_envelope_is_openai_compatible(self):
        """VAL-HTTP-007: response carries the canonical envelope."""
        adapter = ScriptedAdapter(
            [_response("hi", model="minimax-m3:cloud", chatcmpl_id="chatcmpl-1")]
        )
        route = _build_route_config(reflection_turns=0)
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        body = response.json()
        assert body["id"] == "chatcmpl-1"
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "minimax-m3:cloud"
        assert isinstance(body["choices"], list) and body["choices"]
        first = body["choices"][0]
        assert first["message"]["role"] == "assistant"
        assert first["message"]["content"] == "hi"
        assert first["finish_reason"] == "stop"
        usage = body["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)


# ────────────────────────────────────────────────────────────────────
# turns=1 reflection (VAL-PIPE-002, VAL-HTTP-019, VAL-CROSS-001)
# ────────────────────────────────────────────────────────────────────


class TestReflectionTurnsOne:
    """A route with ``reflection.turns: 1`` runs a single critique+revise pair."""

    @pytest.mark.asyncio
    async def test_one_turn_produces_revised_response(self):
        adapter = ScriptedAdapter(
            [
                _response("initial answer", model="minimax-m3:cloud"),
                _response(
                    "a critique\nREFLECT_CONFIDENCE: 0.5",
                    model="minimax-m3:cloud",
                ),
                _response("revised answer", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route_config(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        # The final response is the revised answer.
        assert body["choices"][0]["message"]["content"] == "revised answer"
        # The header reflects the actual turn count.
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        # The confidence header is the parsed value.
        assert float(response.headers["x-moaxy-reflect-confidence"]) == 0.5
        # The orchestrator made 3 LLM calls: initial + critique + revise.
        assert len(adapter.calls) == 3

    @pytest.mark.asyncio
    async def test_one_turn_alias_echoes_original_in_response_body(self):
        """VAL-CROSS-001: reflective route echoes the alias in the body."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"),
                _response("revised", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route_config(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        app = _build_app(adapter, route)
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
        assert response.status_code == 200
        body = response.json()
        # The body.model echoes the original alias, not the resolved name.
        assert body["model"] == "coder-pro"
        # The x-moaxy-alias-resolved header reports the real name.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        # The reflect turns header reflects the actual count.
        assert response.headers["x-moaxy-reflect-turns"] == "1"

    @pytest.mark.asyncio
    async def test_one_turn_early_exit_short_circuits_revision(self):
        """VAL-PIPE-003: high confidence → 2 calls, no revision."""
        adapter = ScriptedAdapter(
            [
                _response("initial answer", model="minimax-m3:cloud"),
                _response(
                    "looks good\nREFLECT_CONFIDENCE: 0.95",
                    model="minimax-m3:cloud",
                ),
            ]
        )
        route = _build_route_config(
            reflection_turns=1,
            early_exit=True,
            threshold=0.85,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        # The response is the initial answer (no revision ran).
        assert body["choices"][0]["message"]["content"] == "initial answer"
        # The header still reflects the attempted turn.
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        # The confidence is 0.95.
        assert float(response.headers["x-moaxy-reflect-confidence"]) == 0.95
        # Only 2 LLM calls: initial + critique.
        assert len(adapter.calls) == 2


# ────────────────────────────────────────────────────────────────────
# Sampling parameters forwarded (VAL-PIPE-042)
# ────────────────────────────────────────────────────────────────────


class TestSamplingParametersForwarded:
    """Sampling parameters are forwarded verbatim to every LLM call."""

    @pytest.mark.asyncio
    async def test_temperature_top_p_max_tokens_forwarded(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
            ]
        )
        route = _build_route_config(
            reflection_turns=1,
            early_exit=False,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 100,
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        for call in adapter.calls:
            assert call["temperature"] == 0.7
            assert call["top_p"] == 0.9
            assert call["max_tokens"] == 100


# ────────────────────────────────────────────────────────────────────
# Fallback integration
# ────────────────────────────────────────────────────────────────────


class TestFallbackIntegration:
    """The orchestrator's fallback chain is reflected in headers."""

    @pytest.mark.asyncio
    async def test_fallback_used_reported_in_header(self):
        adapter = ScriptedAdapter(
            [
                UpstreamError("primary failed", status_code=500, body="err"),
                _response("ok", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route_config(
            reflection_turns=0,
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        # The header is the JSON-encoded list of fallback models used.
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "ok"
        fallbacks_header = response.headers.get("x-moaxy-fallbacks-used")
        assert fallbacks_header is not None
        assert json.loads(fallbacks_header) == ["minimax-m2.7:cloud"]

    @pytest.mark.asyncio
    async def test_no_fallback_header_zero(self):
        adapter = ScriptedAdapter([_response("ok")])
        route = _build_route_config(reflection_turns=0)
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        assert response.headers["x-moaxy-fallbacks-used"] == "0"


# ────────────────────────────────────────────────────────────────────
# Stream flag is ignored at M1-M3 boundary
# ────────────────────────────────────────────────────────────────────


class TestStreamFlagIgnored:
    """``stream: true`` requests are buffered to a JSON response at M1-M3."""

    @pytest.mark.asyncio
    async def test_stream_true_returns_non_streaming_json(self):
        adapter = ScriptedAdapter([_response("ok")])
        route = _build_route_config(reflection_turns=0)
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "ok"


# ────────────────────────────────────────────────────────────────────
# Error translation: adapter exceptions → moaxy HTTP errors
# ────────────────────────────────────────────────────────────────────


class TestErrorTranslation:
    """Adapter/pipeline exceptions become moaxy HTTP error envelopes."""

    @pytest.mark.asyncio
    async def test_all_backends_failed_returns_502(self):
        adapter = ScriptedAdapter(
            [
                UpstreamError("boom", status_code=500, body="err"),
                UpstreamError("boom2", status_code=500, body="err"),
            ]
        )
        route = _build_route_config(
            reflection_turns=0,
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 502
        body = response.json()
        assert "error" in body
        # The last error was a 5xx UpstreamError, so the proxy surfaces
        # it as ``upstream_error`` (the standard 5xx envelope).
        assert body["error"]["type"] == "upstream_error"

    @pytest.mark.asyncio
    async def test_permanent_4xx_upstream_returns_400(self):
        adapter = ScriptedAdapter(
            [UpstreamError("bad", status_code=400, body="bad")]
        )
        route = _build_route_config(reflection_turns=0)
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "upstream rejected" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_request_id_present_on_error_responses(self):
        adapter = ScriptedAdapter(
            [UpstreamError("err", status_code=500, body="err"),
             UpstreamError("err2", status_code=500, body="err2")]
        )
        route = _build_route_config(
            reflection_turns=0,
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 502
        assert REQUEST_ID_HEADER in response.headers
        assert response.headers[REQUEST_ID_HEADER]
        # The body is JSON (no traceback).
        body = response.json()
        assert "error" in body
        assert "traceback" not in response.text.lower()


# ────────────────────────────────────────────────────────────────────
# No-route is a 404 (not 5xx)
# ────────────────────────────────────────────────────────────────────


class TestNoRouteMatch:
    """A request with no matching route returns 404 with a clear error."""

    @pytest.mark.asyncio
    async def test_unmatched_model_returns_404(self):
        adapter = ScriptedAdapter([_response("ok")])
        route = _build_route_config(
            name="specific",
            match_model="specific-model",
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "no-such-model",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 404
        body = response.json()
        assert "no route matches" in body["error"]["message"]
        # The adapter was never called.
        assert len(adapter.calls) == 0


# ────────────────────────────────────────────────────────────────────
# Advisor integration (sanity check; full coverage lives in
# tests/test_orchestrator.py)
# ────────────────────────────────────────────────────────────────────


class TestAdvisorIntegration:
    """The advisor runs after reflection and approves / revises."""

    @pytest.mark.asyncio
    async def test_advisor_approve_keeps_response_unchanged(self):
        adapter = ScriptedAdapter(
            [
                _response("initial"),
                _response("c\nREFLECT_CONFIDENCE: 0.5"),
                _response("revised"),
                _response("ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route_config(
            reflection_turns=1,
            early_exit=False,
            threshold=0.85,
            advisor_model="deepseek-v4-pro:cloud",
            advisor_turns=1,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "revised"
        # The advisor model header is present.
        assert response.headers.get("x-moaxy-advisor-model") == "deepseek-v4-pro:cloud"

    @pytest.mark.asyncio
    async def test_advisor_disabled_omits_advisor_header(self):
        adapter = ScriptedAdapter([_response("initial")])
        route = _build_route_config(
            reflection_turns=0,
            advisor_turns=0,
        )
        app = _build_app(adapter, route)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        assert "x-moaxy-advisor-model" not in response.headers
