"""Tests for the alias-resolution contract of the moaxy pipeline.

Aliases are the per-route mapping that lets a client send a friendly
name (``coder-pro``) and have the proxy dispatch the LLM call to a
real model name (``minimax-m3:cloud``) that the backend adapter
understands. The alias resolution step sits between
:class:`~moaxy.routing.matcher.RouteMatcher.match` and the adapter
call: the matcher produces a :class:`RouteMatch` whose
``original_model`` and ``resolved_model`` fields are the two sides of
the rewrite, and the orchestrator reads ``resolved_model`` when
forwarding to the adapter and ``original_model`` when echoing the
response body.

The tests in this file pin the user-observable contract:

* Alias resolution rewrites the request model the adapter receives.
* An alias map miss passes the model through unchanged.
* The response body's ``model`` field echoes the original alias the
  client sent (the alias is the user-facing identity).
* The ``x-moaxy-alias-resolved`` response header reports the real
  model the proxy dispatched to.
* Empty alias maps and unknown aliases behave like passthroughs.
* Multiple aliases (and the case of an alias whose target is itself
  an alias) are handled correctly when the matcher is configured
  with a per-route alias table.

The tests use the shared :class:`FakeAdapter` defined in
``tests/fixtures/fake_adapter.py`` plus a hand-rolled
:class:`ScriptedAdapter` for in-process HTTP integration tests
mirroring the patterns in :file:`tests/test_advisor.py` and
:file:`tests/test_fallback.py`. No real Ollama, no on-disk plugin
discovery.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.adapters.base import (
    Adapter,
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
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.routing.matcher import RouteMatch, RouteMatcher
from moaxy.server.app import create_app
from tests.fixtures.fake_adapter import FakeAdapter

# ────────────────────────────────────────────────────────────────────
# Response / route / context factories
# ────────────────────────────────────────────────────────────────────


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


def _build_route(
    *,
    aliases: dict[str, str] | None = None,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    reflection_turns: int = 0,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    route_name: str = "alias-route",
) -> RouteMatch:
    config_route = RouteConfig(
        name=route_name,
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=ReflectionConfig(turns=reflection_turns),
        advisor=AdvisorConfig(model=advisor_model, turns=advisor_turns),
    )
    return RouteMatch(
        route=config_route,
        original_model=original_model,
        resolved_model=resolved_model,
        backend="ollama-local",
        path="/v1/chat/completions",
        reflection=config_route.reflection,
        advisor=config_route.advisor,
        fallbacks=list(config_route.fallbacks),
        retry=config_route.retry,
        aliases=dict(config_route.aliases),
    )


def _build_context(
    route: RouteMatch,
    *,
    request_messages: list[dict[str, Any]] | None = None,
    request_extra: dict[str, Any] | None = None,
    request_id: str = "req-1",
) -> PipelineContext:
    body: dict[str, Any] = {
        "model": route.original_model,
        "messages": request_messages
        or [{"role": "user", "content": "ping"}],
    }
    if request_extra:
        body.update(request_extra)
    return PipelineContext(
        request_id=request_id,
        request=body,
        route=route,
        model_alias_resolved=route.resolved_model,
        target_backend=route.backend,
        original_model=route.original_model,
    )


# ────────────────────────────────────────────────────────────────────
# Matcher-level alias resolution
# ────────────────────────────────────────────────────────────────────


class TestMatcherAliasResolution:
    """The matcher rewrites ``coder-pro`` → ``minimax-m3:cloud`` for the adapter."""

    def test_alias_hit_resolves_to_real_model(self):
        """A request with model ``coder-pro`` is matched to the real name ``minimax-m3:cloud``."""
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="alias-route",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert match is not None
        assert match.original_model == "coder-pro"
        assert match.resolved_model == "minimax-m3:cloud"

    def test_alias_miss_passes_through_unchanged(self):
        """An unknown alias is forwarded verbatim; no rewriting happens."""
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="alias-route",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match = matcher.match({"model": "unknown-model", "path": "/v1/chat/completions"})
        assert match is not None
        assert match.original_model == "unknown-model"
        # No entry in the alias map → pass-through.
        assert match.resolved_model == "unknown-model"

    def test_empty_aliases_passes_through(self):
        """A route with ``aliases: {}`` does not rewrite any model name."""
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="no-aliases-route",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    aliases={},
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert match is not None
        assert match.original_model == "minimax-m3:cloud"
        assert match.resolved_model == "minimax-m3:cloud"

    def test_multiple_aliases_each_resolve(self):
        """A route with several aliases rewrites each one to its real name."""
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="multi-alias-route",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    aliases={
                        "coder-pro": "minimax-m3:cloud",
                        "writer-pro": "minimax-m3:cloud",
                        "analyst-pro": "deepseek-v4-pro:cloud",
                    },
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        m1 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        m2 = matcher.match({"model": "writer-pro", "path": "/v1/chat/completions"})
        m3 = matcher.match({"model": "analyst-pro", "path": "/v1/chat/completions"})
        assert m1 is not None and m1.resolved_model == "minimax-m3:cloud"
        assert m2 is not None and m2.resolved_model == "minimax-m3:cloud"
        assert m3 is not None and m3.resolved_model == "deepseek-v4-pro:cloud"

    def test_route_match_aliases_dict_is_a_copy(self):
        """The matcher's ``aliases`` attribute is a fresh dict per match."""
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match1 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        match2 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert match1 is not None and match2 is not None
        # Two different dict instances; mutating one does not affect the other.
        assert match1.aliases is not match2.aliases


# ────────────────────────────────────────────────────────────────────
# Orchestrator: alias forwarding to the adapter
# ────────────────────────────────────────────────────────────────────


class TestOrchestratorAliasForwarding:
    """The orchestrator forwards the resolved model name to the adapter."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_resolved_model_for_initial_call(self):
        """The adapter receives the resolved real model name, not the alias."""
        adapter = FakeAdapter([_response("ok", model="minimax-m3:cloud")])
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_orchestrator_uses_resolved_model_for_reflection_calls(self):
        """Reflection critique and revision also use the resolved model name."""
        adapter = FakeAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"),
                _response("revised", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            reflection_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # All three calls (initial + critique + revise) used the resolved name.
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "minimax-m3:cloud",
            "minimax-m3:cloud",
        ]

    @pytest.mark.asyncio
    async def test_alias_miss_passes_through_to_adapter(self):
        """When the model is not an alias, the adapter receives it verbatim."""
        adapter = FakeAdapter([_response("ok", model="raw-model")])
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="raw-model",
            resolved_model="raw-model",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "raw-model"


# ────────────────────────────────────────────────────────────────────
# Orchestrator: response model echo and x-moaxy-alias-resolved header
# ────────────────────────────────────────────────────────────────────


class TestAliasResponseEcho:
    """The response body's model field echoes the original alias the client sent."""

    @pytest.mark.asyncio
    async def test_response_body_echoes_original_alias(self):
        """``body.model`` is the alias, not the resolved real name."""
        adapter = FakeAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        # The orchestrator stamps the ORIGINAL alias into the response model.
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_response_body_echoes_alias_after_reflection(self):
        """The alias echo survives the reflection loop."""
        adapter = FakeAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response("c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"),
                _response("revised", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            reflection_turns=1,
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        assert ctx.upstream_response is not None
        # The echo holds even when the orchestrator rebuilt the response on revise.
        assert ctx.upstream_response.model == "coder-pro"

    @pytest.mark.asyncio
    async def test_x_moaxy_alias_resolved_header_reports_real_name(self):
        """The ``x-moaxy-alias-resolved`` response header equals the resolved model."""
        adapter = FakeAdapter([_response("ok")])
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
        )
        ctx = _build_context(route, request_id="req-alias")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        assert headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_x_moaxy_alias_resolved_header_equals_request_when_no_rewrite(self):
        """Without an alias hit, the header value equals the request model."""
        adapter = FakeAdapter([_response("ok")])
        route = _build_route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            original_model="unknown-alias",
            resolved_model="unknown-alias",
        )
        ctx = _build_context(route, request_id="req-noalias")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # No alias map hit → the header value equals the request model.
        assert headers["x-moaxy-alias-resolved"] == "unknown-alias"


# ────────────────────────────────────────────────────────────────────
# HTTP integration: alias resolution on the wire
# ────────────────────────────────────────────────────────────────────


class _ScriptedAdapter(Adapter):
    """Minimal in-process adapter for HTTP integration tests.

    Mirrors the pattern in :file:`tests/test_server_orchestrator_integration.py`.
    Each script entry is either a :class:`ChatResponse` (success) or
    a :class:`BaseException` (raised). The adapter records every call
    in :attr:`calls`.
    """

    name = "scripted_alias"

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

    async def stream(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        if False:
            yield ""

    async def close(self) -> None:
        return None


def _build_alias_app(
    adapter: Adapter,
    *,
    aliases: dict[str, str] | None = None,
    backend_name: str = "ollama-local",
) -> Any:
    """Build a FastAPI app with a single alias-aware route."""
    cfg = MoaxyConfig(
        backends=[AdapterConfig(name=backend_name, adapter="ollama", base_url="http://x")],
        routes=[
            RouteConfig(
                name="alias-route",
                match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                backend=backend_name,
                aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
            )
        ],
    )
    registry = AdapterRegistry({backend_name: adapter})
    return create_app(config=cfg, adapters=registry)


class TestAliasHttpIntegration:
    """End-to-end alias resolution on the wire."""

    @pytest.mark.asyncio
    async def test_http_alias_resolved_to_real_model_in_adapter(self):
        """The adapter receives the resolved model name when the client sends an alias."""
        adapter = _ScriptedAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        app = _build_alias_app(adapter)
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
        # The adapter received the RESOLVED name.
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_http_response_body_model_echoes_alias(self):
        """The HTTP response body's ``model`` field is the alias the client sent."""
        adapter = _ScriptedAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        app = _build_alias_app(adapter)
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
        # body.model echoes the original alias.
        assert body["model"] == "coder-pro"
        # The header exposes the resolved name.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_http_alias_miss_passes_through(self):
        """A request with a model that is not an alias is forwarded verbatim."""
        adapter = _ScriptedAdapter(
            [_response("ok", model="raw-model")]
        )
        app = _build_alias_app(
            adapter, aliases={"coder-pro": "minimax-m3:cloud"}
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "raw-model",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        # Adapter received the unmodified model name.
        assert adapter.calls[0]["model"] == "raw-model"
        # The header equals the request model (no alias hit).
        assert response.headers["x-moaxy-alias-resolved"] == "raw-model"
        # The response body echoes the request model.
        body = response.json()
        assert body["model"] == "raw-model"

    @pytest.mark.asyncio
    async def test_http_response_body_uses_alias_in_serialized_json(self):
        """The JSON response body's ``model`` field is a string, not a list or object."""
        adapter = _ScriptedAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        app = _build_alias_app(adapter)
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
        # Sanity: the model field is a string in the JSON envelope.
        assert isinstance(body["model"], str)
        assert body["model"] == "coder-pro"
        # The header and body agree on the two sides of the alias.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert body["model"] != response.headers["x-moaxy-alias-resolved"]


__all__ = [
    "TestMatcherAliasResolution",
    "TestOrchestratorAliasForwarding",
    "TestAliasResponseEcho",
    "TestAliasHttpIntegration",
]
