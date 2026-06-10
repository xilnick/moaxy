"""Tests for the M4 auth gate middleware and admin API endpoints.

Covers:

* Auth gate behaviour: enabled vs disabled, exempt paths, header
  parsing, 401 envelope, principal attachment.
* Env-substitution in ``api_keys[*].key_value`` (VAL-CROSS-012 / the
  auth-side analogue).
* Admin CRUD: GET list, GET single, POST create, DELETE remove,
  404 / 409 envelopes, and role/auth enforcement.
* Backwards compatibility: auth disabled is the default and the
  data plane keeps working without changes.

All tests are in-process via :class:`httpx.ASGITransport` so they
do not require a live uvicorn or a live Ollama. The :class:`FakeAdapter`
and the existing ``make_config`` / ``make_ollama_adapter`` helpers
from ``tests.conftest`` are used to keep the surface consistent
with the rest of the test suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.models.config import (
    AdapterConfig,
    ApiKey,
    AuthConfig,
    MoaxyConfig,
    RouteConfig,
)
from moaxy.models.config import (
    RouteMatch as ConfigRouteMatch,
)
from moaxy.server.app import create_app
from moaxy.server.auth_gate import (
    AuthGateMiddleware,
    Principal,
    build_principal_index,
)
from tests.conftest import (
    make_ollama_adapter,
    make_ollama_payload,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _config_with_auth(
    *,
    enabled: bool = True,
    api_keys: list[ApiKey] | None = None,
    exempt_paths: list[str] | None = None,
    header_names: list[str] | None = None,
    routes: list[RouteConfig] | None = None,
) -> MoaxyConfig:
    """Build a :class:`MoaxyConfig` with the given auth settings.

    Defaults: one backend named ``"b"`` (Ollama at 127.0.0.1:11434)
    and one catch-all route named ``"r"`` (matching model ``*``,
    path ``/v1/chat/completions``). Pass ``routes=[]`` explicitly
    to start with an empty route list.
    """
    if routes is None:
        routes = [
            RouteConfig(
                name="r",
                match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                backend="b",
            )
        ]
    return MoaxyConfig(
        backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://127.0.0.1:11434")],
        routes=routes,
        auth=AuthConfig(
            enabled=enabled,
            exempt_paths=list(exempt_paths) if exempt_paths is not None else ["/health"],
            header_names=list(header_names) if header_names is not None else ["X-API-Key", "Authorization"],
            api_keys=api_keys or [],
        ),
    )


def _write_yaml(path: Path, content: str) -> Path:
    """Write a YAML config file and return the path."""
    path.write_text(content, encoding="utf-8")
    return path


def _chat_payload(model: str = "minimax-m2.7:cloud") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
    }


# ── Auth gate: structural tests ──────────────────────────────────────


class TestPrincipalAndIndex:
    """The :class:`Principal` dataclass and the index builder."""

    def test_principal_is_frozen(self):
        p = Principal(key_id="k1", roles=("admin",), scopes=("*",))
        with pytest.raises((AttributeError, Exception)):
            p.key_id = "k2"  # type: ignore[misc]

    def test_principal_is_admin_for_admin_role(self):
        p = Principal(key_id="k1", roles=("admin",))
        assert p.is_admin is True

    def test_principal_is_admin_for_wildcard_role(self):
        p = Principal(key_id="k1", roles=("*",))
        assert p.is_admin is True

    def test_principal_is_not_admin_for_data_role(self):
        p = Principal(key_id="k1", roles=("user",))
        assert p.is_admin is False

    def test_build_principal_index_keys_by_value(self):
        keys = [
            ApiKey(key_id="k1", key_value="v1"),
            ApiKey(key_id="k2", key_value="v2", roles=["admin"]),
        ]
        idx = build_principal_index(keys)
        assert set(idx.keys()) == {"v1", "v2"}
        assert idx["v1"].key_id == "k1"
        assert idx["v2"].roles == ("admin",)

    def test_build_principal_index_handles_empty(self):
        assert build_principal_index([]) == {}


# ── Auth gate: HTTP behaviour ───────────────────────────────────────


class TestAuthGateHttp:
    """End-to-end auth gate behaviour through the FastAPI app."""

    @pytest.mark.asyncio
    async def test_auth_disabled_passes_without_key(self):
        cfg = _config_with_auth(enabled=False)
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_passes_with_wrong_key(self):
        cfg = _config_with_auth(enabled=False)
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models", headers={"X-API-Key": "wrong-key"}
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_enabled_rejects_no_key(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["type"] == "unauthorized"
        assert "API key" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_auth_enabled_rejects_wrong_key(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models", headers={"X-API-Key": "wrong"}
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_enabled_accepts_correct_key(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models", headers={"X-API-Key": "test-secret-key"}
            )
        assert response.status_code == 200
        assert response.json()["object"] == "list"

    @pytest.mark.asyncio
    async def test_auth_enabled_accepts_bearer_token(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models", headers={"Authorization": "Bearer test-secret-key"}
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_enabled_rejects_non_bearer_authorization(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models",
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_health_exempt_returns_200_without_key(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_exempt_works_with_wrong_key(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health", headers={"X-API-Key": "wrong"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_completions_requires_key_when_auth_enabled(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            from tests.conftest import make_json_response
            return make_json_response(make_ollama_payload(content="ok"))

        adapter = make_ollama_adapter(handler)
        from moaxy.adapters.registry import AdapterRegistry
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        registry = AdapterRegistry({"b": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_chat_completions_succeeds_with_correct_key(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            from tests.conftest import make_json_response
            return make_json_response(make_ollama_payload(content="ok"))

        adapter = make_ollama_adapter(handler)
        from moaxy.adapters.registry import AdapterRegistry
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        registry = AdapterRegistry({"b": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_payload(),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "test-secret-key",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_401_response_is_json_with_request_id(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="test", key_value="test-secret-key")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        assert response.status_code == 401
        assert response.headers["content-type"].startswith("application/json")
        assert "x-moaxy-request-id" in response.headers
        assert len(response.headers["x-moaxy-request-id"]) > 0

    @pytest.mark.asyncio
    async def test_custom_exempt_paths(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            exempt_paths=["/custom-public"],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # /health is no longer exempt, returns 401
            r1 = await client.get("/health")
            assert r1.status_code == 401
            # /custom-public is exempt
            r2 = await client.get("/custom-public")
            # It does not exist as a route, so should be 404
            assert r2.status_code == 404

    @pytest.mark.asyncio
    async def test_custom_header_names(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            header_names=["X-Custom-Key"],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_ok = await client.get(
                "/v1/models", headers={"X-Custom-Key": "v"}
            )
            assert r_ok.status_code == 200
            # X-API-Key is no longer recognised
            r_wrong = await client.get(
                "/v1/models", headers={"X-API-Key": "v"}
            )
            assert r_wrong.status_code == 401

    @pytest.mark.asyncio
    async def test_multiple_keys_picks_match(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[
                ApiKey(key_id="k1", key_value="v1"),
                ApiKey(key_id="k2", key_value="v2"),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/v1/models", headers={"X-API-Key": "v1"})
            r2 = await client.get("/v1/models", headers={"X-API-Key": "v2"})
            r3 = await client.get("/v1/models", headers={"X-API-Key": "v3"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 401

    @pytest.mark.asyncio
    async def test_x_api_key_takes_precedence_over_authorization(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[
                ApiKey(key_id="k1", key_value="v1"),
                ApiKey(key_id="k2", key_value="v2"),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models",
                headers={
                    "X-API-Key": "v1",
                    "Authorization": "Bearer v2",
                },
            )
        assert response.status_code == 200


# ── Auth gate: env substitution ─────────────────────────────────────


class TestAuthEnvSubstitution:
    """``${ENV}`` substitution in ``api_keys[*].key_value``."""

    def test_env_substitution_in_api_key_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MOAXY_ADMIN_API_KEY", "secret-from-env")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: olloma-local
    adapter: ollama
    base_url: http://127.0.0.1:11434
auth:
  enabled: true
  api_keys:
    - key_id: admin
      key_value: "${MOAXY_ADMIN_API_KEY}"
      roles: ["admin"]
""",
        )
        from moaxy.config import load_config
        cfg = load_config(path=path)
        assert cfg.auth is not None
        assert cfg.auth.enabled is True
        assert len(cfg.auth.api_keys) == 1
        assert cfg.auth.api_keys[0].key_value == "secret-from-env"
        assert cfg.auth.api_keys[0].key_id == "admin"

    def test_missing_env_var_fails_loud_in_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOAXY_MISSING_KEY", raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: olloma-local
    adapter: ollama
    base_url: http://127.0.0.1:11434
auth:
  enabled: true
  api_keys:
    - key_id: admin
      key_value: "${MOAXY_MISSING_KEY}"
""",
        )
        from moaxy.config import SubstitutionError, load_config
        with pytest.raises(SubstitutionError) as exc:
            load_config(path=path)
        assert "MOAXY_MISSING_KEY" in str(exc.value)

    def test_env_substitution_works_end_to_end(self, tmp_path, monkeypatch):
        """End-to-end: env subst -> config -> auth gate accepts the key."""
        monkeypatch.setenv("MOAXY_ADMIN_API_KEY", "live-secret-123")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    backend: b
auth:
  enabled: true
  api_keys:
    - key_id: admin
      key_value: "${MOAXY_ADMIN_API_KEY}"
""",
        )
        from moaxy.config import load_config
        cfg = load_config(path=path)
        app = create_app(config=cfg)

        async def run() -> None:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # No key
                r_no = await client.get("/v1/models")
                # Right key (from env)
                r_ok = await client.get(
                    "/v1/models", headers={"X-API-Key": "live-secret-123"}
                )
                # /health
                r_health = await client.get("/health")
            assert r_no.status_code == 401
            assert r_ok.status_code == 200
            assert r_health.status_code == 200

        asyncio.run(run())


# ── Admin API: GET endpoints ────────────────────────────────────────


class TestAdminListRoutes:
    """``GET /admin/routes`` lists every registered route."""

    @pytest.mark.asyncio
    async def test_list_routes_returns_all_routes(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="r1",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                ),
                RouteConfig(
                    name="r2",
                    match=ConfigRouteMatch(model="foo", path="/v1/chat/completions"),
                    backend="b",
                ),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/admin/routes", headers={"X-API-Key": "v"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert {route["name"] for route in body["routes"]} == {"r1", "r2"}

    @pytest.mark.asyncio
    async def test_list_routes_requires_auth_when_enabled(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/admin/routes")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_list_routes_returns_empty_when_no_routes(self):
        from moaxy.models.config import MoaxyConfig
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[],
            auth=AuthConfig(enabled=True, api_keys=[ApiKey(key_id="k", key_value="v")]),
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/admin/routes", headers={"X-API-Key": "v"})
        assert r.status_code == 200
        assert r.json() == {"routes": [], "count": 0}


class TestAdminGetRoute:
    """``GET /admin/routes/{name}`` returns one route by name."""

    @pytest.mark.asyncio
    async def test_get_route_returns_matching_route(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="reflective",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                ),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/admin/routes/reflective", headers={"X-API-Key": "v"})
        assert r.status_code == 200
        body = r.json()
        assert body["route"]["name"] == "reflective"
        assert body["route"]["aliases"] == {"coder-pro": "minimax-m3:cloud"}

    @pytest.mark.asyncio
    async def test_get_route_returns_404_for_unknown_name(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/admin/routes/nope", headers={"X-API-Key": "v"})
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"


# ── Admin API: POST create ──────────────────────────────────────────


class TestAdminCreateRoute:
    """``POST /admin/routes`` appends a new route."""

    @pytest.mark.asyncio
    async def test_create_route_appends_and_returns_201(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        body = {
            "name": "new-route",
            "match": {"model": "foo", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={"X-API-Key": "v", "Content-Type": "application/json"},
                json=body,
            )
        assert r.status_code == 201
        assert r.json()["route"]["name"] == "new-route"
        # Location header is set
        assert r.headers["Location"] == "/admin/routes/new-route"

    @pytest.mark.asyncio
    async def test_created_route_is_listed_and_retrievable(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[],
        )
        app = create_app(config=cfg)
        body = {
            "name": "new-route",
            "match": {"model": "foo", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/admin/routes",
                headers={"X-API-Key": "v", "Content-Type": "application/json"},
                json=body,
            )
            r2 = await client.get("/admin/routes", headers={"X-API-Key": "v"})
            r3 = await client.get(
                "/admin/routes/new-route", headers={"X-API-Key": "v"}
            )
        assert r1.status_code == 201
        assert r2.status_code == 200
        assert r2.json()["count"] == 1
        assert r3.status_code == 200
        assert r3.json()["route"]["name"] == "new-route"

    @pytest.mark.asyncio
    async def test_create_route_409_on_duplicate_name(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="r1",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                ),
            ],
        )
        app = create_app(config=cfg)
        body = {
            "name": "r1",
            "match": {"model": "*", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={"X-API-Key": "v", "Content-Type": "application/json"},
                json=body,
            )
        assert r.status_code == 409
        assert r.json()["error"]["type"] == "conflict"

    @pytest.mark.asyncio
    async def test_create_route_400_on_invalid_body(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={"X-API-Key": "v", "Content-Type": "application/json"},
                json={"name": "bad"},  # missing match
            )
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "bad_request"

    @pytest.mark.asyncio
    async def test_create_route_400_on_malformed_json(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={"X-API-Key": "v", "Content-Type": "application/json"},
                content=b"{not json",
            )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_create_route_requires_auth(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        body = {
            "name": "new-route",
            "match": {"model": "foo", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={"Content-Type": "application/json"},
                json=body,
            )
        assert r.status_code == 401


# ── Admin API: DELETE ───────────────────────────────────────────────


class TestAdminDeleteRoute:
    """``DELETE /admin/routes/{name}`` removes a route."""

    @pytest.mark.asyncio
    async def test_delete_route_returns_204(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="victim",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                ),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.delete(
                "/admin/routes/victim", headers={"X-API-Key": "v"}
            )
        assert r.status_code == 204

    @pytest.mark.asyncio
    async def test_deleted_route_is_gone(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="victim",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                ),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_del = await client.delete(
                "/admin/routes/victim", headers={"X-API-Key": "v"}
            )
            r_get = await client.get(
                "/admin/routes/victim", headers={"X-API-Key": "v"}
            )
            r_list = await client.get("/admin/routes", headers={"X-API-Key": "v"})
        assert r_del.status_code == 204
        assert r_get.status_code == 404
        assert r_list.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_delete_route_404_for_unknown(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.delete(
                "/admin/routes/nope", headers={"X-API-Key": "v"}
            )
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    @pytest.mark.asyncio
    async def test_delete_route_requires_auth(self):
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            routes=[
                RouteConfig(
                    name="victim",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                ),
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.delete("/admin/routes/victim")
        assert r.status_code == 401


# ── Admin API: integration with routing ─────────────────────────────


class TestAdminRouteIntegration:
    """The admin CRUD changes the matcher's view of the world."""

    @pytest.mark.asyncio
    async def test_created_route_actually_routes_requests(self):
        """VAL-CROSS-008 (partial): new route is routable after creation."""
        async def handler(_request: httpx.Request) -> httpx.Response:
            from tests.conftest import make_json_response
            return make_json_response(make_ollama_payload(content="ok"))

        adapter = make_ollama_adapter(handler)
        from moaxy.adapters.registry import AdapterRegistry
        cfg = _config_with_auth(
            enabled=True,
            api_keys=[ApiKey(key_id="k", key_value="v")],
            # Start with no routes
            routes=[],
        )
        registry = AdapterRegistry({"b": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Pre-create: 404
            r_pre = await client.post(
                "/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "v",
                },
                json=_chat_payload(model="my-model"),
            )
            # Create a route for my-model
            r_create = await client.post(
                "/admin/routes",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "v",
                },
                json={
                    "name": "new",
                    "match": {
                        "model": "my-model",
                        "path": "/v1/chat/completions",
                    },
                    "backend": "b",
                },
            )
            # Post-create: 200
            r_post = await client.post(
                "/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "v",
                },
                json=_chat_payload(model="my-model"),
            )
        assert r_pre.status_code == 404
        assert r_create.status_code == 201
        assert r_post.status_code == 200
        assert r_post.json()["choices"][0]["message"]["content"] == "ok"


# ── Auth gate: when auth is disabled (backwards compatibility) ──────


class TestAuthDisabledBehaviour:
    """When ``auth.enabled`` is false, the data plane works as before."""

    @pytest.mark.asyncio
    async def test_admin_returns_401_when_auth_disabled(self):
        """Admin endpoints require auth even when auth.enabled is false.

        The contract: admin endpoints require auth (X-Admin-Key or
        similar). When auth is disabled, there is no key to present,
        so admin is unreachable. This is the secure default.
        """
        cfg = _config_with_auth(enabled=False)
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_list = await client.get("/admin/routes")
            r_get = await client.get("/admin/routes/anything")
            r_post = await client.post(
                "/admin/routes",
                headers={"Content-Type": "application/json"},
                json={"name": "x"},
            )
            r_del = await client.delete("/admin/routes/anything")
        assert r_list.status_code == 401
        assert r_get.status_code == 401
        assert r_post.status_code == 401
        assert r_del.status_code == 401

    @pytest.mark.asyncio
    async def test_data_plane_unaffected_when_auth_disabled(self):
        """When auth is disabled, /health and /v1/models return 200."""
        cfg = _config_with_auth(enabled=False)
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_health = await client.get("/health")
            r_models = await client.get("/v1/models")
        assert r_health.status_code == 200
        assert r_models.status_code == 200


# ── Auth gate: middleware-level unit tests ──────────────────────────


class TestAuthGateMiddlewareUnit:
    """Direct unit tests of the middleware without the FastAPI app."""

    def test_is_exempt_exact_match(self):
        mw = AuthGateMiddleware(
            app=None,  # type: ignore[arg-type]
            principal_index={},
            exempt_paths=("/health", "/public"),
        )
        assert mw.is_exempt("/health") is True
        assert mw.is_exempt("/public") is True
        assert mw.is_exempt("/v1/models") is False
        assert mw.is_exempt("/health/sub") is False
