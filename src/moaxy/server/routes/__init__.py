"""HTTP route handlers for the moaxy proxy.

The package exposes a single ``router`` aggregator that :func:`moaxy.server.app.create_app`
mounts under the root prefix. Each module owns a logical surface:

* :mod:`moaxy.server.routes.health` — liveness probe at ``GET /health``.
* :mod:`moaxy.server.routes.models` — ``GET /v1/models`` model catalogue.
* :mod:`moaxy.server.routes.proxy` — ``POST /v1/chat/completions`` data
  plane and the catch-all OpenAI-compatible entry point.
* :mod:`moaxy.server.routes.admin` — ``/admin/*`` runtime CRUD endpoints
  (M4). Protected by :class:`moaxy.server.auth_gate.AuthGateMiddleware`
  plus a per-endpoint ``admin``-role check.
"""

from fastapi import APIRouter

from moaxy.server.routes import admin, health, models, proxy

router = APIRouter()
router.include_router(health.router)
router.include_router(models.router)
router.include_router(proxy.router)
router.include_router(admin.router)

__all__ = ["router"]
