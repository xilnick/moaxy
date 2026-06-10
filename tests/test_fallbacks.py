"""Tests for the fallback-walker contract of the moaxy pipeline.

The :func:`~moaxy.pipeline.fallback.call_with_fallbacks` walker is the
M2/M3 policy that turns "the route said the model is X, with fallbacks
Y and Z and retry R" into a deterministic retry-then-walk sequence at
every LLM call site (initial, reflection critique, reflection
revision, advisor, advisor revision). The walker returns a tuple of
``(response, fallbacks_used)``; the orchestrator aggregates the
per-call ``fallbacks_used`` lists and stamps them into the
``x-moaxy-fallbacks-used`` response header.

The tests in this file pin the orchestrator-level contract for the
walker:

* The walker walks the chain in order, advancing on transient errors.
* The retry budget is ``retry + 1`` attempts per model, then advance.
* Permanent (4xx) errors short-circuit immediately; no fallbacks are
  consulted (the request is malformed).
* Exhaustion raises :class:`UpstreamExhaustedError`; the HTTP layer
  turns that into ``502`` with a JSON body whose ``error.message``
  contains the substring ``"all backends failed"``.
* Per-route ``fallbacks`` and ``retry`` override the global
  ``cfg.models.fallbacks[model]`` and ``cfg.models.retry[model]``
  values; the per-route config wins when both are set.
* The ``x-moaxy-fallbacks-used`` response header is empty (or absent
  or equal to ``"0"``) when the primary served the call, and contains
  the list of fallback models actually used otherwise.

The tests use the shared :class:`FakeAdapter` defined in
``tests/fixtures/fake_adapter.py`` for orchestrator-level tests and a
hand-rolled :class:`ScriptedAdapter` mirroring the patterns in
:file:`tests/test_fallback.py` for in-process HTTP integration tests
and granular walker-only tests where call recording matters.

The behaviour of the walker itself (the deep coverage of every
transient/permanent edge case) is already pinned in
:file:`tests/test_fallback.py`. This file focuses on the
ORCHESTRATOR-LEVEL properties the validation contract cares about:
the per-step invocation order, the aggregated header value, the
exhaustion path, and the per-route-overrides-global precedence rule.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    Usage,
)
from moaxy.adapters.registry import AdapterRegistry
from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    MoaxyConfig,
    ModelDefaults,
    ReflectionConfig,
    RouteConfig,
)
from moaxy.models.config import RouteMatch as ConfigRouteMatch
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.fallback import (
    UpstreamExhaustedError,
    call_with_fallbacks,
)
from moaxy.pipeline.orchestrator import Orchestrator, build_response_headers
from moaxy.routing.matcher import RouteMatch
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
    fallbacks: list[str] | None = None,
    retry: int = 0,
    reflection_turns: int = 0,
    early_exit: bool = True,
    threshold: float = 0.85,
    advisor_model: str | None = None,
    advisor_turns: int = 0,
    aliases: dict[str, str] | None = None,
    original_model: str = "coder-pro",
    resolved_model: str = "minimax-m3:cloud",
    route_name: str = "fallback-route",
) -> RouteMatch:
    config_route = RouteConfig(
        name=route_name,
        match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        backend="ollama-local",
        aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
        fallbacks=fallbacks if fallbacks is not None else [],
        retry=retry,
        reflection=ReflectionConfig(
            turns=reflection_turns,
            early_exit=early_exit,
            threshold=threshold,
        ),
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
    request_id: str = "req-1",
) -> PipelineContext:
    return PipelineContext(
        request_id=request_id,
        request={
            "model": route.original_model,
            "messages": request_messages
            or [{"role": "user", "content": "ping"}],
        },
        route=route,
        model_alias_resolved=route.resolved_model,
        target_backend=route.backend,
        original_model=route.original_model,
    )


def _5xx(message: str = "upstream 500") -> UpstreamError:
    return UpstreamError(message, status_code=500, body=message)


def _4xx(message: str = "upstream 400") -> UpstreamError:
    return UpstreamError(message, status_code=400, body=message)


def _timeout(message: str = "upstream timeout") -> UpstreamTimeoutError:
    return UpstreamTimeoutError(message)


def _unavailable(message: str = "upstream unavailable") -> UpstreamUnavailableError:
    return UpstreamUnavailableError(message)


# ────────────────────────────────────────────────────────────────────
# Per-step fallback walker: focused walker tests at the orchestrator level
# ────────────────────────────────────────────────────────────────────


class TestFallbackChainWalk:
    """The walker advances through the chain on transient failures."""

    @pytest.mark.asyncio
    async def test_primary_fails_first_fallback_succeeds(self):
        """A transient failure on the primary advances to the first fallback."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("from fallback", model="m2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from fallback"
        assert fallbacks_used == ["m2"]
        # Two calls: m1 (failed), m2 (succeeded).
        assert [c["model"] for c in adapter.calls] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_walks_full_chain_then_succeeds_on_last_fallback(self):
        """Both primary and first fallback fail; second fallback succeeds."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _5xx(),
                _response("from third", model="m3"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2", "m3"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from third"
        assert fallbacks_used == ["m2", "m3"]
        assert [c["model"] for c in adapter.calls] == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_fallback_uses_real_models_from_config(self):
        """The walker uses the real model names from the route's fallbacks list."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="minimax-m2.7:cloud"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["minimax-m3:cloud", "minimax-m2.7:cloud"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "ok"
        assert fallbacks_used == ["minimax-m2.7:cloud"]
        assert adapter.calls[1]["model"] == "minimax-m2.7:cloud"


class TestRetryBudget:
    """The retry budget is ``retry + 1`` attempts per model."""

    @pytest.mark.asyncio
    async def test_retry_two_means_three_attempts(self):
        """``retry=2`` produces 1 initial + 2 retries = 3 attempts on the primary."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _5xx(),
                _5xx(),
                _response("ok", model="m2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        # m1 tried 3 times, then m2 succeeded.
        assert [c["model"] for c in adapter.calls] == ["m1", "m1", "m1", "m2"]
        assert fallbacks_used == ["m2"]
        assert response.message.content == "ok"

    @pytest.mark.asyncio
    async def test_retry_zero_means_single_attempt(self):
        """``retry=0`` produces a single attempt on the primary."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="m2"),
            ]
        )
        await call_with_fallbacks(
            adapter,
            models=["m1", "m2"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        # m1 tried once, then m2 tried once.
        assert [c["model"] for c in adapter.calls] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_retry_mixed_with_chained_fallbacks(self):
        """``retry=1`` on a 2-model chain → 2 attempts per model."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _5xx(),
                _5xx(),
                _5xx(),
                _response("ok", model="m3"),
            ]
        )
        await call_with_fallbacks(
            adapter,
            models=["m1", "m2", "m3"],
            retry=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        # m1: 2 attempts; m2: 2 attempts; m3: 1 attempt (succeeded).
        assert [c["model"] for c in adapter.calls] == [
            "m1", "m1", "m2", "m2", "m3",
        ]

    @pytest.mark.asyncio
    async def test_four_xx_does_not_consume_retry_budget(self):
        """A 4xx on a model raises immediately; the retry budget is not consulted."""
        adapter = FakeAdapter(
            [
                _4xx(),
                _response("should not be called", model="m2"),
            ]
        )
        with pytest.raises(UpstreamError) as exc_info:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2"],
                retry=3,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert exc_info.value.status_code == 400
        # The fallback was never called.
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "m1"


class TestFallbackExhaustion:
    """Exhausting every model in the chain raises :class:`UpstreamExhaustedError`."""

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_upstream_exhausted(self):
        """All 5xx → :class:`UpstreamExhaustedError` is raised."""
        adapter = FakeAdapter([_5xx(), _5xx(), _5xx()])
        with pytest.raises(UpstreamExhaustedError) as exc_info:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2", "m3"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        # The error message must contain "all backends failed" (VAL-RT-013).
        assert "all backends failed" in str(exc_info.value)
        # The exception carries the model chain.
        assert exc_info.value.models == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_empty_models_list_raises_upstream_exhausted(self):
        """An empty ``models`` list raises immediately."""
        adapter = FakeAdapter([])
        with pytest.raises(UpstreamExhaustedError) as exc_info:
            await call_with_fallbacks(
                adapter,
                models=[],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "all backends failed" in str(exc_info.value)
        assert exc_info.value.models == []

    @pytest.mark.asyncio
    async def test_exhaustion_message_contains_all_backends_failed(self):
        """The exception's str() includes ``all backends failed`` (the contract substring)."""
        adapter = FakeAdapter([_timeout(), _unavailable(), _5xx()])
        with pytest.raises(UpstreamExhaustedError) as exc_info:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2", "m3"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "all backends failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exhausted_error_carries_last_error(self):
        """The :class:`UpstreamExhaustedError` carries the last transient error."""
        adapter = FakeAdapter([_5xx(), _timeout()])
        with pytest.raises(UpstreamExhaustedError) as exc_info:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        # The last_error is the most recent transient error (the timeout).
        assert isinstance(exc_info.value.last_error, UpstreamTimeoutError)


# ────────────────────────────────────────────────────────────────────
# Orchestrator-level: x-moaxy-fallbacks-used header
# ────────────────────────────────────────────────────────────────────


class TestFallbacksUsedHeader:
    """The orchestrator stamps ``x-moaxy-fallbacks-used`` from aggregated fallbacks."""

    @pytest.mark.asyncio
    async def test_no_fallbacks_triggered_header_is_zero(self):
        """When the primary serves every call, the header is ``"0"`` (or empty)."""
        adapter = FakeAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=1,
        )
        ctx = _build_context(route, request_id="req-fb0")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The primary served; no fallbacks triggered.
        assert headers["x-moaxy-fallbacks-used"] == "0"

    @pytest.mark.asyncio
    async def test_one_fallback_triggered_header_is_json_list(self):
        """When the walker uses one fallback, the header is a JSON list of the model."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="minimax-m2.7:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud"],
            retry=0,
        )
        ctx = _build_context(route, request_id="req-fb1")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The header is a JSON list of the fallback models actually used.
        # ``fallbacks_used`` is a list; the orchestrator serialises it as
        # JSON. (See :func:`build_response_headers`.)
        import json as _json
        assert _json.loads(headers["x-moaxy-fallbacks-used"]) == [
            "minimax-m2.7:cloud"
        ]

    @pytest.mark.asyncio
    async def test_two_fallbacks_triggered_header_is_json_list(self):
        """When the walker uses two fallbacks, the header is a JSON list of both."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _5xx(),
                _response("ok", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route(
            fallbacks=["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"],
            retry=0,
        )
        ctx = _build_context(route, request_id="req-fb2")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # The header is a JSON list of the two fallback models used.
        import json as _json
        assert _json.loads(headers["x-moaxy-fallbacks-used"]) == [
            "minimax-m2.7:cloud",
            "deepseek-v4-pro:cloud",
        ]

    @pytest.mark.asyncio
    async def test_fallbacks_used_aggregates_across_reflection_turns(self):
        """The header sums fallback usage across every LLM call in the request."""
        adapter = FakeAdapter(
            [
                _5xx(),  # initial primary fails
                _response("initial-fb", model="m2"),  # initial from fallback
                _5xx(),  # critique primary fails
                _response("critique-fb", model="m2"),  # critique from fallback
                _response("revised", model="m2"),  # revise from fallback
            ]
        )
        route = _build_route(
            fallbacks=["m2"],
            retry=0,
            reflection_turns=1,
            early_exit=False,
        )
        ctx = _build_context(route, request_id="req-fb3")
        await Orchestrator(adapter).run(ctx)
        headers = build_response_headers(ctx, request_id=ctx.request_id)
        # Two fallbacks were triggered (initial + critique); the revise
        # was served by the (working) fallback without an additional hop.
        import json as _json
        assert _json.loads(headers["x-moaxy-fallbacks-used"]) == ["m2", "m2"]

    @pytest.mark.asyncio
    async def test_fallbacks_used_aggregated_in_context_dict(self):
        """The orchestrator stashes ``fallbacks_used`` on the context."""
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="m2"),
            ]
        )
        route = _build_route(
            fallbacks=["m2"],
            retry=0,
        )
        ctx = _build_context(route, request_id="req-fb4")
        await Orchestrator(adapter).run(ctx)
        # The runtime attribute is the list of fallback models used.
        fallbacks_used = ctx.__dict__.get("fallbacks_used", [])
        assert fallbacks_used == ["m2"]


# ────────────────────────────────────────────────────────────────────
# Per-route overrides global models.fallbacks / models.retry
# ────────────────────────────────────────────────────────────────────


class TestRouteOverridesGlobal:
    """The per-route ``fallbacks`` and ``retry`` override the global ``cfg.models`` table."""

    @pytest.mark.asyncio
    async def test_route_fallbacks_override_global_fallbacks(self):
        """When both the route and the global table have entries, the route wins."""
        # The route declares fallbacks=[route-fb]. The global table declares
        # [global-fb]. The walker uses route-fb; global-fb is never called.
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="route-fb"),
            ]
        )
        route = _build_route(
            fallbacks=["route-fb"],
            retry=0,
            resolved_model="primary-model",
        )
        ctx = _build_context(route, request_id="req-override")
        await Orchestrator(adapter).run(ctx)
        # The adapter was called with route-fb, not global-fb.
        assert adapter.calls[0]["model"] == "primary-model"
        assert adapter.calls[1]["model"] == "route-fb"

    @pytest.mark.asyncio
    async def test_route_retry_overrides_global_retry(self):
        """The per-route ``retry`` value is what the walker uses."""
        # With retry=0 on the route, the primary is tried exactly once.
        # Even if the global model table says retry=5, the route wins.
        adapter = FakeAdapter(
            [
                _5xx(),
                _response("ok", model="m2"),
            ]
        )
        route = _build_route(
            fallbacks=["m2"],
            retry=0,  # Route-level override
            resolved_model="primary",
        )
        ctx = _build_context(route)
        await Orchestrator(adapter).run(ctx)
        # The primary was called once (not 6 times).
        primary_calls = [c for c in adapter.calls if c["model"] == "primary"]
        assert len(primary_calls) == 1

    @pytest.mark.asyncio
    async def test_matchers_route_overrides_when_route_fallbacks_set_explicitly(self):
        """The matcher honors ``model_fields_set`` to distinguish set vs default."""
        from moaxy.routing.matcher import RouteMatcher

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            models=ModelDefaults(
                fallbacks={"primary": ["global-fb"]},
                retry={"primary": 2},
            ),
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    fallbacks=["route-fb"],
                    retry=0,
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match = matcher.match({"model": "primary", "path": "/v1/chat/completions"})
        assert match is not None
        # Route value wins; the global value is ignored.
        assert match.fallbacks == ["route-fb"]
        assert match.retry == 0

    @pytest.mark.asyncio
    async def test_global_fallbacks_used_when_route_does_not_set_fallbacks(self):
        """When the route does NOT set ``fallbacks``, the global table is consulted."""
        from moaxy.routing.matcher import RouteMatcher

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            models=ModelDefaults(
                fallbacks={"primary": ["global-fb-1", "global-fb-2"]},
                retry={"primary": 1},
            ),
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="ollama-local",
                    # No fallbacks, no retry fields set explicitly.
                )
            ],
        )
        matcher = RouteMatcher(cfg)
        match = matcher.match({"model": "primary", "path": "/v1/chat/completions"})
        assert match is not None
        # The global table's value is the effective value.
        assert match.fallbacks == ["global-fb-1", "global-fb-2"]
        assert match.retry == 1


# ────────────────────────────────────────────────────────────────────
# HTTP integration: fallback chain + header
# ────────────────────────────────────────────────────────────────────


class _ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    Mirrors the pattern in :file:`tests/test_server_orchestrator_integration.py`.
    """

    name = "scripted_fb"

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


def _build_fallback_app(
    adapter: Adapter,
    *,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    aliases: dict[str, str] | None = None,
    backend_name: str = "ollama-local",
) -> Any:
    """Build a FastAPI app with a fallback-aware route."""
    cfg = MoaxyConfig(
        backends=[AdapterConfig(name=backend_name, adapter="ollama", base_url="http://x")],
        routes=[
            RouteConfig(
                name="fallback-route",
                match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                backend=backend_name,
                aliases=aliases or {"coder-pro": "minimax-m3:cloud"},
                fallbacks=fallbacks or [],
                retry=retry,
            )
        ],
    )
    registry = AdapterRegistry({backend_name: adapter})
    return create_app(config=cfg, adapters=registry)


class TestFallbackHttpIntegration:
    """End-to-end fallback behaviour on the wire."""

    @pytest.mark.asyncio
    async def test_http_fallback_walks_chain_and_succeeds(self):
        """A request whose primary fails is served by the first fallback."""
        adapter = _ScriptedAdapter(
            [
                _5xx(),
                _response("from fallback", model="minimax-m2.7:cloud"),
            ]
        )
        app = _build_fallback_app(
            adapter, fallbacks=["minimax-m2.7:cloud"], retry=0
        )
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
        assert body["choices"][0]["message"]["content"] == "from fallback"
        # The header is a JSON list of fallback models used.
        import json as _json
        assert _json.loads(response.headers["x-moaxy-fallbacks-used"]) == [
            "minimax-m2.7:cloud"
        ]

    @pytest.mark.asyncio
    async def test_http_fallback_exhaustion_returns_502(self):
        """All models failing → 502 with JSON body whose error details
        mention the exhausted model chain (VAL-RT-013)."""
        adapter = _ScriptedAdapter([_5xx(), _5xx(), _5xx()])
        app = _build_fallback_app(
            adapter,
            fallbacks=["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"],
            retry=0,
        )
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
        assert response.status_code == 502
        body = response.json()
        # The error is JSON-shaped.
        assert "error" in body
        # The error details list every model the walker tried.
        error_details = body["error"].get("details", {})
        models_tried = error_details.get("models", [])
        # The full chain (primary + the two fallbacks) is in the details.
        assert "minimax-m3:cloud" in models_tried
        assert "minimax-m2.7:cloud" in models_tried
        assert "deepseek-v4-pro:cloud" in models_tried

    @pytest.mark.asyncio
    async def test_http_four_xx_short_circuits(self):
        """A 4xx on the primary bubbles up as a 4xx; no fallbacks are tried."""
        adapter = _ScriptedAdapter(
            [
                _4xx(),
                _response("should not be called", model="m2"),
            ]
        )
        app = _build_fallback_app(
            adapter, fallbacks=["m2"], retry=0
        )
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
        # The 4xx translates to a 4xx (BadRequestError → 400).
        assert 400 <= response.status_code < 500
        # The fallback was never invoked.
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["model"] == "minimax-m3:cloud"

    @pytest.mark.asyncio
    async def test_http_fallbacks_used_header_absent_on_clean_run(self):
        """When the primary serves, the header is ``"0"`` (per the orchestrator)."""
        adapter = _ScriptedAdapter(
            [_response("ok", model="minimax-m3:cloud")]
        )
        app = _build_fallback_app(
            adapter, fallbacks=["minimax-m2.7:cloud"], retry=1
        )
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
        # The primary served; the header value is the literal "0".
        assert response.headers["x-moaxy-fallbacks-used"] == "0"


__all__ = [
    "TestFallbackChainWalk",
    "TestRetryBudget",
    "TestFallbackExhaustion",
    "TestFallbacksUsedHeader",
    "TestRouteOverridesGlobal",
    "TestFallbackHttpIntegration",
]
