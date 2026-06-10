"""``GET /v1/models`` — OpenAI-compatible model list.

The handler returns ``{"object": "list", "data": [<model>, ...]}`` where
each entry is shaped like OpenAI's ``model`` object (``{"id": ..., "object": "model"}``).

The model list is built from the parsed :class:`moaxy.models.config.MoaxyConfig`:

* Every ``route.aliases`` key is added as a first-class model id (the
  client sees aliases as the primary identity).
* Every ``route.aliases`` value (i.e. the resolved real model name) is
  also included.
* The set is deduped and sorted for deterministic output.

The route is wired in :func:`moaxy.server.app.create_app`. The handler
itself is intentionally simple — it does not consult the upstream
provider for its model catalogue, so it works even when Ollama is down.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from moaxy.models.config import MoaxyConfig

router = APIRouter()


def _build_model_ids(config: MoaxyConfig) -> list[str]:
    """Collect every model id the proxy knows about, deduped and sorted."""
    ids: set[str] = set()
    for route in config.routes:
        ids.update(route.aliases.keys())
        ids.update(route.aliases.values())
        if route.backend:
            ids.add(route.backend)
    return sorted(ids)


def _model_entry(model_id: str) -> dict[str, str]:
    """Build an OpenAI-shaped model entry for a single id."""
    return {"id": model_id, "object": "model"}


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    """Return an OpenAI-shaped model catalogue built from the route table."""
    config: MoaxyConfig = request.app.state.config
    ids = _build_model_ids(config)
    return {"object": "list", "data": [_model_entry(mid) for mid in ids]}


__all__ = ["router"]
