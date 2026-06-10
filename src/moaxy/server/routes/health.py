"""Liveness probe at ``GET /health``.

The handler returns a stable JSON body (``{"status": "ok"}``) so external
load balancers, container health checks, and the validation suite can
confirm the proxy is up. The endpoint is exempt from auth in the M1
configuration (it is added to ``auth.exempt_paths`` by default).

The handler is a plain function rather than a class to keep the M1
surface minimal; a class-based version can replace it once the
``/admin/healthz`` style richer probes are added in M4.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for liveness probes."""
    return {"status": "ok"}


__all__ = ["router"]
