"""Cross-area end-to-end tests for the moaxy pipeline.

These tests pin the user-observable behaviour of the moaxy proxy when
multiple subsystems are involved in a single request:

* Alias resolution + reflection + advisor (VAL-CROSS-001, 002, 005).
* Self-advise + reflection + alias (VAL-CROSS-003).
* Fallback chain walking under a reflective + advisor load
  (VAL-CROSS-004).
* Concurrent request isolation across the full FastAPI app
  (VAL-CROSS-006).
* Pipeline event ordering across reflection and advisor stages
  (VAL-PIPE-035).

The tests are hermetic: every LLM call is answered by a hand-rolled
:class:`ScriptedAdapter` that records the model name and messages
forwarded by the orchestrator. The FastAPI app is built in-process
via :func:`moaxy.server.app.create_app` and exercised with
:mod:`httpx` + :class:`ASGITransport`. No real Ollama, no on-disk
plugin discovery, no listening socket on the loopback port.

The cross-area tests complement the per-area files
:file:`tests/test_advisor.py`, :file:`tests/test_aliases.py`,
:file:`tests/test_fallbacks.py`, and :file:`tests/test_reflection.py`
by exercising combinations of behaviours that no single-area test
can pin. They are the closest the M3 hermetic test suite comes to a
real end-to-end run.
"""

from __future__ import annotations

import asyncio
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
from moaxy.pipeline.context import PipelineContext
from moaxy.pipeline.orchestrator import Orchestrator
from moaxy.routing.matcher import RouteMatch
from moaxy.server.app import create_app

# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter — hermetic, in-process
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    Mirrors the pattern in :file:`tests/test_server_orchestrator_integration.py`.
    Each script entry is either a :class:`ChatResponse` (success) or
    a :class:`BaseException` (raised). Calls are recorded in
    :attr:`calls`.
    """

    name = "scripted_cross"

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


def _5xx(message: str = "upstream 500") -> UpstreamError:
    return UpstreamError(message, status_code=500, body=message)


def _build_route_config(
    *,
    name: str = "reflective-coder",
    match_model: str = "*",
    backend: str = "ollama-local",
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
    backend_name: str = "ollama-local",
) -> Any:
    """Build a FastAPI app with the scripted adapter and route mounted."""
    cfg = MoaxyConfig(
        backends=[AdapterConfig(name=backend_name, adapter="ollama", base_url="http://x")],
        routes=[route],
    )
    registry = AdapterRegistry({backend_name: adapter})
    return create_app(config=cfg, adapters=registry)


# ────────────────────────────────────────────────────────────────────
# Cross-area: alias + reflection + response echo (VAL-CROSS-001)
# ────────────────────────────────────────────────────────────────────


class TestReflectiveCoderEndToEnd:
    """VAL-CROSS-001: alias → reflect → response with header echo."""

    @pytest.mark.asyncio
    async def test_reflective_coder_full_flow(self):
        """``coder-pro`` → alias → reflect → advisor approve → final response."""
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
            aliases={"coder-pro": "minimax-m3:cloud"},
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
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        # body.model echoes the original alias.
        assert body["model"] == "coder-pro"
        # Final content is the revised answer.
        assert body["choices"][0]["message"]["content"] == "revised answer"
        # The headers reflect the actual flow.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        assert float(response.headers["x-moaxy-reflect-confidence"]) == 0.5
        # Three LLM calls (initial + critique + revise) all to the resolved model.
        assert len(adapter.calls) == 3
        assert all(c["model"] == "minimax-m3:cloud" for c in adapter.calls)


# ────────────────────────────────────────────────────────────────────
# Cross-area: cross-advise (primary != advisor) end-to-end (VAL-CROSS-002)
# ────────────────────────────────────────────────────────────────────


class TestCrossAdviseEndToEnd:
    """VAL-CROSS-002: primary and advisor use different model names."""

    @pytest.mark.asyncio
    async def test_cross_advise_uses_two_distinct_models(self):
        """The initial call goes to the primary; the advisor call goes to the advisor."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection_turns=0,
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
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        # The response echoes the alias.
        assert body["model"] == "coder-pro"
        # The advisor header reports the configured advisor model.
        assert response.headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        # The two calls used different model names.
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "deepseek-v4-pro:cloud",
        ]


# ────────────────────────────────────────────────────────────────────
# Cross-area: self-advise end-to-end (VAL-CROSS-003)
# ────────────────────────────────────────────────────────────────────


class TestSelfAdviseEndToEnd:
    """VAL-CROSS-003: advisor.model equals the primary; both calls share the name."""

    @pytest.mark.asyncio
    async def test_self_advise_two_distinct_llm_calls_same_model_name(self):
        """Self-advise: the orchestrator makes two LLM calls to the same model name."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_APPROVE", model="minimax-m3:cloud"
                ),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection_turns=0,
            advisor_model="minimax-m3:cloud",
            advisor_turns=1,
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
        # The advisor header is present and matches the primary.
        assert response.headers["x-moaxy-advisor-model"] == "minimax-m3:cloud"
        # Two LLM calls: both to the same model name.
        assert len(adapter.calls) == 2
        assert [c["model"] for c in adapter.calls] == [
            "minimax-m3:cloud",
            "minimax-m3:cloud",
        ]


# ────────────────────────────────────────────────────────────────────
# Cross-area: fallback end-to-end (VAL-CROSS-004)
# ────────────────────────────────────────────────────────────────────


class TestFallbackEndToEnd:
    """VAL-CROSS-004: primary 5xx → walk chain → succeed; header reports count."""

    @pytest.mark.asyncio
    async def test_fallback_walks_chain_and_succeeds_with_header(self):
        """Two fallbacks are used; the header reports both."""
        adapter = ScriptedAdapter(
            [
                _5xx(),  # primary fails
                _5xx(),  # first fallback fails
                _response("ok from deepseek", model="deepseek-v4-pro:cloud"),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            fallbacks=["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"],
            retry=0,
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
        assert body["choices"][0]["message"]["content"] == "ok from deepseek"
        # The header is a JSON list of fallback models actually used.
        header_value = response.headers["x-moaxy-fallbacks-used"]
        import json as _json
        assert _json.loads(header_value) == [
            "minimax-m2.7:cloud",
            "deepseek-v4-pro:cloud",
        ]

    @pytest.mark.asyncio
    async def test_fallback_with_retry_succeeds_on_retry(self):
        """Primary fails once, retries, succeeds. ``fallbacks_used`` is empty."""
        adapter = ScriptedAdapter(
            [
                _5xx(),  # primary fails
                _response("primary recovered", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            fallbacks=["minimax-m2.7:cloud"],
            retry=1,
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
        # The primary served (after one retry); no fallbacks were triggered.
        assert body["choices"][0]["message"]["content"] == "primary recovered"
        assert response.headers["x-moaxy-fallbacks-used"] == "0"


# ────────────────────────────────────────────────────────────────────
# Cross-area: reflect + advisor in a single request (VAL-CROSS-005)
# ────────────────────────────────────────────────────────────────────


class TestReflectPlusAdvisorEndToEnd:
    """VAL-CROSS-005: one request runs both reflection and advisor stages."""

    @pytest.mark.asyncio
    async def test_reflect_then_advisor_approve(self):
        """Initial → reflect → revise → advisor approve. All four headers set."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"
                ),
                _response("revised", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection_turns=1,
            early_exit=False,
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
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        # All four x-moaxy-* headers are present and correct.
        assert response.headers["x-moaxy-reflect-turns"] == "1"
        assert float(response.headers["x-moaxy-reflect-confidence"]) == 0.5
        assert response.headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        # 4 LLM calls: initial, critique, revise, advisor.
        assert len(adapter.calls) == 4
        # Final response content is the revised answer (advisor approved).
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "revised"

    @pytest.mark.asyncio
    async def test_reflect_then_advisor_revise(self):
        """Reflect → revise → advisor REVISE → primary revise → final."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"
                ),
                _response("revised", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_REVISE: better final", model="deepseek-v4-pro:cloud"
                ),
                # The primary's final revision after ADVISOR_REVISE.
                _response("primary-final", model="minimax-m3:cloud"),
            ]
        )
        route = _build_route_config(
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection_turns=1,
            early_exit=False,
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
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        # The final response is the primary's revision.
        assert body["choices"][0]["message"]["content"] == "primary-final"
        # 5 LLM calls.
        assert len(adapter.calls) == 5
        # The alias and advisor headers are correct.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert response.headers["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"


# ────────────────────────────────────────────────────────────────────
# Cross-area: pipeline event ordering (VAL-PIPE-035)
# ────────────────────────────────────────────────────────────────────


class TestPipelineEventOrdering:
    """All ``reflect_*`` events appear before any ``advisor*`` event."""

    @pytest.mark.asyncio
    async def test_advisor_events_follow_reflection_events(self):
        """The events list ordered: initial → reflect_* → advisor* → done."""
        adapter = ScriptedAdapter(
            [
                _response("initial", model="minimax-m3:cloud"),
                _response(
                    "c1\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"
                ),
                _response("rev1", model="minimax-m3:cloud"),
                _response(
                    "c2\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"
                ),
                _response("rev2", model="minimax-m3:cloud"),
                _response(
                    "ADVISOR_APPROVE", model="deepseek-v4-pro:cloud"
                ),
            ]
        )
        # Build a context and run the orchestrator directly (no HTTP).
        from moaxy.models.config import RouteMatch as ConfigRouteMatch

        config_route = RouteConfig(
            name="ordering-route",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="ollama-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(turns=2, early_exit=False),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        matcher_route = RouteMatch(
            route=config_route,
            original_model="coder-pro",
            resolved_model="minimax-m3:cloud",
            backend="ollama-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases=dict(config_route.aliases),
        )
        ctx = PipelineContext(
            request_id="req-ordering",
            request={
                "model": "coder-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=matcher_route,
            model_alias_resolved=matcher_route.resolved_model,
            target_backend=matcher_route.backend,
            original_model=matcher_route.original_model,
        )
        await Orchestrator(adapter).run(ctx)
        # Validate event ordering: types list.
        types = [e.type for e in ctx.events]
        assert types == [
            "initial",
            "reflect_critique",
            "reflect_revised",
            "reflect_critique",
            "reflect_revised",
            "advisor",
            "advisor_approve",
        ]
        # The first reflect_critique is at index 1 (after initial).
        assert types[0] == "initial"
        # All reflect_* events appear before any advisor event.
        last_reflect_index = max(
            i for i, t in enumerate(types) if t.startswith("reflect_")
        )
        first_advisor_index = next(
            i for i, t in enumerate(types) if t.startswith("advisor")
        )
        assert last_reflect_index < first_advisor_index


# ────────────────────────────────────────────────────────────────────
# Cross-area: concurrent request isolation on the FastAPI app
# ────────────────────────────────────────────────────────────────────


class TestConcurrentRequestIsolation:
    """VAL-CROSS-006: two concurrent requests are isolated end-to-end."""

    @pytest.mark.asyncio
    async def test_two_concurrent_requests_isolated(self):
        """Two parallel ``POST`` requests produce distinct request_ids and
        distinct response contents."""
        # The adapter is shared; it dispatches in arrival order. We give
        # it enough scripted responses for both requests' flows.
        # Use a single reflective route with aliases and a separate
        # "no-reflection" path for the second request, by using two
        # distinct model names that the matcher routes differently.
        adapter = ScriptedAdapter(
            [
                # Request A: model "reflective" — reflective flow
                # (initial + critique + revise).
                _response("answer-A", model="reflective"),
                _response("c-A\nREFLECT_CONFIDENCE: 0.5", model="reflective"),
                _response("revised-A", model="reflective"),
                # Request B: model "plain" — passthrough.
                _response("answer-B", model="plain"),
            ]
        )
        # Two distinct routes with EXACT model matchers (no glob).
        route_a = _build_route_config(
            name="reflective-route",
            match_model="coder-pro",  # only matches "coder-pro"
            aliases={"coder-pro": "reflective"},
            reflection_turns=1,
            early_exit=False,
        )
        route_b = _build_route_config(
            name="plain-route",
            match_model="plain-model",  # only matches "plain-model"
            aliases={"plain-model": "plain"},
            reflection_turns=0,
        )
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[route_a, route_b],
        )
        registry = AdapterRegistry({"ollama-local": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response_a, response_b = await asyncio.gather(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "coder-pro",
                        "messages": [{"role": "user", "content": "A"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "plain-model",
                        "messages": [{"role": "user", "content": "B"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
            )
        # Both responses are 200.
        assert response_a.status_code == 200, response_a.text
        assert response_b.status_code == 200, response_b.text
        # Distinct request_ids.
        from moaxy.server.middleware import REQUEST_ID_HEADER
        request_id_a = response_a.headers[REQUEST_ID_HEADER]
        request_id_b = response_b.headers[REQUEST_ID_HEADER]
        assert request_id_a != request_id_b
        # Each response carries its own content (no cross-talk).
        body_a = response_a.json()
        body_b = response_b.json()
        assert body_a["choices"][0]["message"]["content"] == "revised-A"
        assert body_b["choices"][0]["message"]["content"] == "answer-B"
        # Reflect-turns header differs: A ran reflection, B did not.
        assert response_a.headers["x-moaxy-reflect-turns"] == "1"
        assert response_b.headers["x-moaxy-reflect-turns"] == "0"

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_different_aliases(self):
        """Two requests with different aliases are resolved independently."""
        adapter = ScriptedAdapter(
            [
                # Request A: alias "writer-pro" → "minimax-m3:cloud"
                _response("A", model="minimax-m3:cloud"),
                # Request B: alias "coder-pro" → "minimax-m3:cloud"
                _response("B", model="minimax-m3:cloud"),
            ]
        )
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="multi-alias",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="ollama-local",
                    aliases={
                        "writer-pro": "minimax-m3:cloud",
                        "coder-pro": "minimax-m3:cloud",
                    },
                )
            ],
        )
        registry = AdapterRegistry({"ollama-local": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response_a, response_b = await asyncio.gather(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "writer-pro",
                        "messages": [{"role": "user", "content": "A"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "coder-pro",
                        "messages": [{"role": "user", "content": "B"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
            )
        # Each response echoes its own alias.
        body_a = response_a.json()
        body_b = response_b.json()
        assert body_a["model"] == "writer-pro"
        assert body_b["model"] == "coder-pro"
        # Both headers report the same resolved real name.
        assert response_a.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert response_b.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"


# ────────────────────────────────────────────────────────────────────
# Cross-area: route override of global fallbacks
# ────────────────────────────────────────────────────────────────────


class TestRouteOverridesGlobalInPipeline:
    """The per-route ``fallbacks`` value wins over the global ``models.fallbacks`` table."""

    @pytest.mark.asyncio
    async def test_route_fallbacks_used_when_both_set(self):
        """When both are set, the per-route value drives the walker."""
        from moaxy.models.config import ModelDefaults

        adapter = ScriptedAdapter(
            [
                _5xx(),  # primary fails
                _response(
                    "ok from route-fb", model="route-fb-model"
                ),
            ]
        )
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="ollama-local", adapter="ollama", base_url="http://x")],
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["global-fb-model"]},
            ),
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="ollama-local",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                    fallbacks=["route-fb-model"],  # route override
                )
            ],
        )
        registry = AdapterRegistry({"ollama-local": adapter})
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
        assert response.status_code == 200
        # The walker used the route's fallback list, not the global one.
        import json as _json
        fallbacks_used = _json.loads(response.headers["x-moaxy-fallbacks-used"])
        assert fallbacks_used == ["route-fb-model"]
        # The adapter was called with the route-level fallback model,
        # not the global one.
        assert adapter.calls[1]["model"] == "route-fb-model"


__all__ = [
    "TestReflectiveCoderEndToEnd",
    "TestCrossAdviseEndToEnd",
    "TestSelfAdviseEndToEnd",
    "TestFallbackEndToEnd",
    "TestReflectPlusAdvisorEndToEnd",
    "TestPipelineEventOrdering",
    "TestConcurrentRequestIsolation",
    "TestRouteOverridesGlobalInPipeline",
]
