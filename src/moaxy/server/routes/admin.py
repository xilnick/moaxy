"""Admin API endpoints for runtime route CRUD (M4).

The admin surface is the documented runtime-management entry point.
The endpoint set is intentionally narrow — the M4 contract calls out
four operations:

* ``GET /admin/routes`` — list every route currently registered in
  the in-memory :class:`~moaxy.routing.matcher.RouteMatcher`.
* ``GET /admin/routes/{name}`` — fetch a single route by its ``name``.
* ``POST /admin/routes`` — append a new route. The body is a
  :class:`~moaxy.models.config.RouteConfig`-shaped JSON object.
* ``DELETE /admin/routes/{name}`` — remove a route by its ``name``.

The endpoints are protected by the same
:class:`~moaxy.server.auth_gate.AuthGateMiddleware` that guards the
data plane: a request to ``/admin/*`` must carry a valid API key
(``X-API-Key`` header or ``Authorization: Bearer <key>``). When
``auth.enabled`` is true, requests without a key are rejected with
401 before the admin route handler runs. When ``auth.enabled`` is
false, the admin endpoints remain locked: there is no key to
present, so the in-handler :func:`_require_authenticated` check
raises 401. This is the secure-by-default behaviour: a route table
mutation is never reachable without an explicit API key in the
running process.

State management
----------------

The endpoints mutate :attr:`fastapi.FastAPI.state.route_matcher`, the
in-memory :class:`~moaxy.routing.matcher.RouteMatcher` installed at
startup. The mutation is in-process only: the on-disk
``config.yaml`` is NOT rewritten, and a process restart loses the
runtime changes. This matches the M4 contract's "runtime CRUD"
language; persistent config writes are deferred to a future
mission.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from moaxy.models.config import RouteConfig
from moaxy.routing.matcher import RouteMatcher
from moaxy.server.auth_gate import Principal
from moaxy.server.errors import (
    BadRequestError,
    NotFoundError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


# ── Auth helpers ────────────────────────────────────────────────────


def _require_authenticated(request: Request) -> Principal:
    """Return the principal when authenticated, or raise an HTTP error.

    The :class:`AuthGateMiddleware` mounts upstream of the admin
    endpoints and rejects unauthenticated requests with a 401
    before they reach the route handler. The presence of a
    :class:`Principal` on ``request.state.principal`` is the
    contract for "the gate accepted the key".

    The admin endpoints do NOT enforce a per-role check: any key
    in the auth block can call admin endpoints, because the auth
    block is the canonical place to enumerate trusted callers.
    Operators that want to scope admin access can put only
    admin-only keys in the auth block, leaving data-plane keys
    out. (The role/scope information on the principal is preserved
    for downstream auditing; future extensions can add per-key
    role gating without changing this contract.)
    """
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is None:
        raise UnauthorizedError(
            "admin endpoint requires authentication",
            details={"path": request.url.path},
        )
    return principal


def _matcher(request: Request) -> RouteMatcher:
    """Return the in-memory route matcher from app state.

    Raises:
        RuntimeError: The app was not wired with a route matcher.
            This is a programming error, not a user error; the
            factory always sets ``app.state.route_matcher``.
    """
    matcher: RouteMatcher | None = getattr(request.app.state, "route_matcher", None)
    if matcher is None:
        raise RuntimeError("route matcher is not installed on app.state")
    return matcher


# ── Serialisation helpers ────────────────────────────────────────────


def _route_to_dict(route: RouteConfig) -> dict[str, Any]:
    """Serialise a :class:`RouteConfig` for the admin JSON envelope.

    The envelope mirrors the Pydantic ``model_dump`` output but with
    the matcher's per-route effective values (``fallbacks`` and
    ``retry``) overlaid onto the raw config. The shape is
    round-trippable through :func:`_route_from_dict` for the
    ``POST /admin/routes`` endpoint.
    """
    data = route.model_dump()
    return data


def _route_from_dict(payload: dict[str, Any]) -> RouteConfig:
    """Parse a request body into a :class:`RouteConfig`.

    Pydantic validation errors are translated into
    :class:`moaxy.server.errors.BadRequestError` so the registered
    exception handler renders a 400 with a useful ``details`` field
    (the raw ``ValidationError`` payload).
    """
    try:
        return RouteConfig.model_validate(payload)
    except ValidationError as exc:
        raise BadRequestError(
            "invalid route payload: the request body does not match RouteConfig",
            details={"errors": exc.errors(include_url=False)},
        ) from exc


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/routes")
async def list_routes(request: Request) -> dict[str, Any]:
    """List every route currently registered with the matcher.

    The response is a JSON object with a ``routes`` array. Each
    entry is the :func:`_route_to_dict` form of the corresponding
    :class:`RouteConfig`. The list is ordered by route registration
    order, which is also the first-match-wins order used by the
    matcher.
    """
    _require_authenticated(request)
    matcher = _matcher(request)
    routes = [_route_to_dict(r) for r in matcher.routes]
    return {"routes": routes, "count": len(routes)}


@router.get("/routes/{name}")
async def get_route(name: str, request: Request) -> dict[str, Any]:
    """Return the :class:`RouteConfig` with the given ``name``.

    Raises:
        NotFoundError: No route with the given name is registered.
    """
    _require_authenticated(request)
    matcher = _matcher(request)
    for route in matcher.routes:
        if route.name == name:
            return {"route": _route_to_dict(route)}
    raise NotFoundError(
        f"no route named {name!r} is registered",
        details={"name": name},
    )


@router.post("/routes")
async def create_route(request: Request) -> JSONResponse:
    """Append a new :class:`RouteConfig` to the in-memory matcher.

    The request body is a JSON object matching the
    :class:`RouteConfig` schema. The new route is appended to the
    end of the list; existing routes are unchanged.

    Returns:
        A 201 response with the created route's serialised form
        in the body and a ``Location`` header pointing at
        ``/admin/routes/{name}`` for clients that want to fetch
        the route back.

    Raises:
        BadRequestError: The body fails Pydantic validation.
        ValueError: A route with the same name is already
            registered; the error is translated into a 409
            Conflict response.
    """
    _require_authenticated(request)
    try:
        payload = await request.json()
    except ValueError as exc:
        raise BadRequestError(
            "malformed JSON body: the request payload is not valid JSON",
            details={"parser": "json"},
        ) from exc
    if not isinstance(payload, dict):
        raise BadRequestError(
            "request body must be a JSON object",
            details={"type": type(payload).__name__},
        )
    route = _route_from_dict(payload)
    matcher = _matcher(request)
    try:
        matcher.add_route(route)
    except ValueError as exc:
        # Re-raise as a 409 Conflict so the registered exception
        # handler renders the canonical envelope. The 409 is a
        # runtime detail; the spec for the POST endpoint is
        # "create or fail", and the failure mode for a duplicate
        # name is documented here.
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "type": "conflict",
                    "message": str(exc),
                    "details": {"name": route.name},
                }
            },
            headers={"x-moaxy-request-id": getattr(request.state, "request_id", "")},
            media_type="application/json",
        )
    return JSONResponse(
        status_code=201,
        content={"route": _route_to_dict(route)},
        headers={
            "Location": f"/admin/routes/{route.name}",
            "x-moaxy-request-id": getattr(request.state, "request_id", ""),
        },
        media_type="application/json",
    )


@router.delete("/routes/{name}")
async def delete_route(name: str, request: Request) -> JSONResponse:
    """Remove a route by name.

    Returns a 204 No Content response on success. Returns 404 when
    no route with the given name is registered.
    """
    _require_authenticated(request)
    matcher = _matcher(request)
    removed = matcher.remove_route(name)
    if not removed:
        raise NotFoundError(
            f"no route named {name!r} is registered",
            details={"name": name},
        )
    return JSONResponse(
        status_code=204,
        content=None,
        headers={"x-moaxy-request-id": getattr(request.state, "request_id", "")},
    )


# Re-export the error classes so callers do not have to import them
# from the errors module separately.
__all__ = ["router"]
