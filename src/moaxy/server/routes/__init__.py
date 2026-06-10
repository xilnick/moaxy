"""HTTP route handlers for the moaxy proxy.

The package exposes a single ``router`` aggregator that :func:`moaxy.server.app.create_app`
mounts under the root prefix. Each module owns a logical surface:

* :mod:`moaxy.server.routes.health` — liveness probe at ``GET /health``.
* :mod:`moaxy.server.routes.models` — ``GET /v1/models`` model catalogue.
* :mod:`moaxy.server.routes.proxy` — ``POST /v1/chat/completions`` data
  plane and the catch-all OpenAI-compatible entry point.
"""

from fastapi import APIRouter

from moaxy.server.routes import health, models, proxy

router = APIRouter()
router.include_router(health.router)
router.include_router(models.router)
router.include_router(proxy.router)

__all__ = ["router"]
