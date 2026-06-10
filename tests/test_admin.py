"""Tests for the M4 admin API: route CRUD endpoints and auth gating.

The admin API exposes runtime route CRUD over HTTP:

* ``GET /admin/routes`` — list every registered route.
* ``GET /admin/routes/{name}`` — fetch a single route by name.
* ``POST /admin/routes`` — append a new route.
* ``DELETE /admin/routes/{name}`` — remove a route by name.

All admin endpoints require an API key (``X-API-Key`` header or
``Authorization: Bearer <key>``). When ``auth.enabled`` is true,
the auth gate rejects unauthenticated requests with 401 before
they reach the admin handler. When ``auth.enabled`` is false, the
in-handler :func:`_require_authenticated` check still rejects
unauthenticated requests with 401 — the admin surface is
secure-by-default and is never reachable without an explicit API
key, even when the data plane is unlocked.

This file complements the broader :mod:`tests.test_auth_admin`
suite (which covers the full auth/admin matrix including bearer
token parsing, env substitution, and the
``test_auth_enabled_rejects_*`` variants). The tests here focus
on the M4 admin CRUD contract: GET list, GET one, POST create,
DELETE remove, and the auth requirement on every admin endpoint.
"""

from __future__ import annotations

from typing import Any

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

# ────────────────────────────────────────────────────────────────────
# Config builder
# ────────────────────────────────────────────────────────────────────


def _config(
    *,
    auth_enabled: bool = True,
    api_key_value: str = "admin-secret",
    api_key_id: str = "admin",
    routes: list[RouteConfig] | None = None,
) -> MoaxyConfig:
    """Build a :class:`MoaxyConfig` with the given auth settings.

    Defaults: a single backend named ``"b"`` and a single
    catch-all route named ``"r"``. Pass ``routes=[]`` for an
    empty route list.
    """
    if routes is None:
        routes = [
            RouteConfig(
                name="r",
                match=ConfigRouteMatch(
                    model="*", path="/v1/chat/completions"
                ),
                backend="b",
            )
        ]
    return MoaxyConfig(
        backends=[
            AdapterConfig(
                name="b", adapter="ollama", base_url="http://127.0.0.1:11434"
            )
        ],
        routes=routes,
        auth=AuthConfig(
            enabled=auth_enabled,
            api_keys=[
                ApiKey(
                    key_id=api_key_id,
                    key_value=api_key_value,
                    roles=["admin"],
                )
            ],
        ),
    )


def _auth_headers(key: str = "admin-secret") -> dict[str, str]:
    return {"X-API-Key": key}


# ────────────────────────────────────────────────────────────────────
# GET /admin/routes
# ────────────────────────────────────────────────────────────────────


class TestAdminListRoutes:
    """``GET /admin/routes`` lists every registered route."""

    @pytest.mark.asyncio
    async def test_list_routes_returns_count_and_array(self):
        """The list endpoint returns a JSON envelope with
        ``routes`` (array) and ``count`` (integer). Both must
        reflect the current route table.
        """
        cfg = _config(
            routes=[
                RouteConfig(
                    name="r1",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="b",
                ),
                RouteConfig(
                    name="r2",
                    match=ConfigRouteMatch(
                        model="minimax-*",
                        path="/v1/chat/completions",
                    ),
                    backend="b",
                ),
                RouteConfig(
                    name="r3",
                    match=ConfigRouteMatch(
                        model="deepseek-*",
                        path="/v1/chat/completions",
                    ),
                    backend="b",
                ),
            ]
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        names = {route["name"] for route in body["routes"]}
        assert names == {"r1", "r2", "r3"}
        # Every entry carries the canonical RouteConfig shape.
        for route in body["routes"]:
            assert "match" in route
            assert route["match"]["path"] == "/v1/chat/completions"
            assert route["backend"] == "b"

    @pytest.mark.asyncio
    async def test_list_routes_requires_auth(self):
        """The list endpoint rejects unauthenticated requests with 401."""
        cfg = _config()
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_no = await client.get("/admin/routes")
            r_wrong = await client.get(
                "/admin/routes", headers=_auth_headers("wrong")
            )
        assert r_no.status_code == 401
        assert r_wrong.status_code == 401
        # Both error envelopes are JSON with the canonical
        # ``error`` shape.
        for r in (r_no, r_wrong):
            body = r.json()
            assert body["error"]["type"] == "unauthorized"
            assert "API key" in body["error"]["message"]


# ────────────────────────────────────────────────────────────────────
# GET /admin/routes/{name}
# ────────────────────────────────────────────────────────────────────


class TestAdminGetRoute:
    """``GET /admin/routes/{name}`` returns one route by name."""

    @pytest.mark.asyncio
    async def test_get_route_returns_named_route(self):
        """The single-route endpoint returns the matching
        :class:`RouteConfig` (including aliases and fallbacks) in
        the ``route`` field of the response.
        """
        cfg = _config(
            routes=[
                RouteConfig(
                    name="reflective",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                    fallbacks=["minimax-m2.7:cloud"],
                    retry=2,
                )
            ]
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/admin/routes/reflective", headers=_auth_headers()
            )
        assert response.status_code == 200
        body = response.json()
        assert body["route"]["name"] == "reflective"
        assert body["route"]["aliases"] == {"coder-pro": "minimax-m3:cloud"}
        assert body["route"]["fallbacks"] == ["minimax-m2.7:cloud"]
        assert body["route"]["retry"] == 2

    @pytest.mark.asyncio
    async def test_get_unknown_route_returns_404(self):
        """An unknown name returns 404 with a structured error body."""
        cfg = _config()
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/admin/routes/does-not-exist", headers=_auth_headers()
            )
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["type"] == "not_found"
        assert "does-not-exist" in body["error"]["message"]


# ────────────────────────────────────────────────────────────────────
# POST /admin/routes
# ────────────────────────────────────────────────────────────────────


class TestAdminCreateRoute:
    """``POST /admin/routes`` appends a new :class:`RouteConfig`."""

    @pytest.mark.asyncio
    async def test_create_route_returns_201_and_is_retrievable(self):
        """A successful create returns 201 with the new route in
        the body and a ``Location`` header pointing at the
        per-name endpoint. The route is then listable and
        fetchable through the GET endpoints.
        """
        cfg = _config(routes=[])
        app = create_app(config=cfg)
        new_route: dict[str, Any] = {
            "name": "new-route",
            "match": {"model": "foo", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_create = await client.post(
                "/admin/routes",
                headers={
                    **_auth_headers(),
                    "Content-Type": "application/json",
                },
                json=new_route,
            )
            r_list = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
            r_get = await client.get(
                "/admin/routes/new-route", headers=_auth_headers()
            )
        assert r_create.status_code == 201
        assert r_create.json()["route"]["name"] == "new-route"
        assert r_create.headers["Location"] == "/admin/routes/new-route"
        # The new route is now part of the matcher.
        assert r_list.status_code == 200
        assert r_list.json()["count"] == 1
        assert r_get.status_code == 200
        assert r_get.json()["route"]["name"] == "new-route"

    @pytest.mark.asyncio
    async def test_create_route_409_on_duplicate_name(self):
        """A POST with a name that already exists returns 409
        (Conflict) and does not modify the route table.
        """
        cfg = _config(
            routes=[
                RouteConfig(
                    name="existing",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="b",
                )
            ]
        )
        app = create_app(config=cfg)
        body = {
            "name": "existing",
            "match": {"model": "*", "path": "/v1/chat/completions"},
            "backend": "b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_create = await client.post(
                "/admin/routes",
                headers={
                    **_auth_headers(),
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r_list = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
        assert r_create.status_code == 409
        assert r_create.json()["error"]["type"] == "conflict"
        # The original route is still the only one.
        assert r_list.json()["count"] == 1
        assert r_list.json()["routes"][0]["name"] == "existing"

    @pytest.mark.asyncio
    async def test_create_route_400_on_invalid_body(self):
        """A POST with a body that fails Pydantic validation (e.g.
        missing ``match``) returns 400 with a structured error
        body that names the validation problems.
        """
        cfg = _config()
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/admin/routes",
                headers={
                    **_auth_headers(),
                    "Content-Type": "application/json",
                },
                json={"name": "bad"},  # missing match
            )
        assert r.status_code == 400
        body = r.json()
        assert body["error"]["type"] == "bad_request"
        # The details include the raw Pydantic error list.
        assert "errors" in body["error"]["details"]

    @pytest.mark.asyncio
    async def test_create_route_requires_auth(self):
        """The create endpoint rejects unauthenticated requests with 401."""
        cfg = _config()
        app = create_app(config=cfg)
        body = {
            "name": "x",
            "match": {"model": "*", "path": "/v1/chat/completions"},
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


# ────────────────────────────────────────────────────────────────────
# DELETE /admin/routes/{name}
# ────────────────────────────────────────────────────────────────────


class TestAdminDeleteRoute:
    """``DELETE /admin/routes/{name}`` removes a route."""

    @pytest.mark.asyncio
    async def test_delete_route_returns_204_and_route_is_gone(self):
        """A successful delete returns 204 with an empty body.
        The deleted route is no longer listable or fetchable.
        """
        cfg = _config(
            routes=[
                RouteConfig(
                    name="victim",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="b",
                )
            ]
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_del = await client.delete(
                "/admin/routes/victim", headers=_auth_headers()
            )
            r_get = await client.get(
                "/admin/routes/victim", headers=_auth_headers()
            )
            r_list = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
        assert r_del.status_code == 204
        # The DELETE handler returns ``JSONResponse(content=None,
        # status_code=204)``; FastAPI serialises ``None`` to the
        # JSON literal ``null``. The HTTP spec allows a 204 with
        # an empty body, so the wire body is treated as empty
        # for downstream consumers — but the parsed body
        # decodes to ``None``. The test asserts on the parsed
        # value (which is what clients see).
        assert r_del.json() is None
        # The route is gone.
        assert r_get.status_code == 404
        assert r_list.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_delete_route_404_for_unknown(self):
        """A DELETE on a non-existent name returns 404 with a
        structured error body.
        """
        cfg = _config()
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.delete(
                "/admin/routes/nope", headers=_auth_headers()
            )
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["type"] == "not_found"
        assert "nope" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_delete_route_requires_auth(self):
        """The delete endpoint rejects unauthenticated requests with 401."""
        cfg = _config(
            routes=[
                RouteConfig(
                    name="victim",
                    match=ConfigRouteMatch(
                        model="*", path="/v1/chat/completions"
                    ),
                    backend="b",
                )
            ]
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.delete("/admin/routes/victim")
        assert r.status_code == 401


# ────────────────────────────────────────────────────────────────────
# Admin CRUD combined: a full create-list-get-delete cycle
# ────────────────────────────────────────────────────────────────────


class TestAdminFullCycle:
    """A complete CRUD cycle: create, list, get, delete, then
    confirm the matcher state matches the actions taken.
    """

    @pytest.mark.asyncio
    async def test_full_crud_cycle(self):
        """Run the canonical admin workflow end-to-end:

        1. POST /admin/routes to append a new route.
        2. GET /admin/routes and confirm the new route is listed.
        3. GET /admin/routes/{name} and confirm the body matches.
        4. DELETE /admin/routes/{name} and confirm 204.
        5. GET /admin/routes and confirm the route is gone.
        """
        cfg = _config(routes=[])
        app = create_app(config=cfg)
        new_route = {
            "name": "crud-victim",
            "match": {"model": "foo", "path": "/v1/chat/completions"},
            "backend": "b",
            "aliases": {"foo": "minimax-m3:cloud"},
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/admin/routes",
                headers={
                    **_auth_headers(),
                    "Content-Type": "application/json",
                },
                json=new_route,
            )
            assert r1.status_code == 201

            r2 = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
            assert r2.status_code == 200
            assert r2.json()["count"] == 1
            assert r2.json()["routes"][0]["name"] == "crud-victim"

            r3 = await client.get(
                "/admin/routes/crud-victim", headers=_auth_headers()
            )
            assert r3.status_code == 200
            assert r3.json()["route"]["aliases"] == {
                "foo": "minimax-m3:cloud"
            }

            r4 = await client.delete(
                "/admin/routes/crud-victim", headers=_auth_headers()
            )
            assert r4.status_code == 204

            r5 = await client.get(
                "/admin/routes", headers=_auth_headers()
            )
            assert r5.status_code == 200
            assert r5.json()["count"] == 0
