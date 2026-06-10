"""FastAPI app factory for the moaxy OpenAI-compatible proxy.

:func:`create_app` is the single entry point. It wires the routes,
middleware, error handlers, and per-app state (config, adapter
registry, route matcher) on a fresh :class:`fastapi.FastAPI` instance.

The factory accepts an optional pre-built :class:`moaxy.models.config.MoaxyConfig`
and :class:`moaxy.adapters.registry.AdapterRegistry`. Tests use this to
inject deterministic configurations and an in-process transport-backed
adapter; production callers typically pass nothing and rely on
:func:`moaxy.config.loader.load_config` and
:func:`moaxy.adapters.registry.build_registry` to construct the
defaults from the on-disk config file.

The application is intentionally self-contained: importing this module
performs no I/O and starts no background tasks. uvicorn (or any other
ASGI server) takes the result of ``create_app()`` and binds it to
``127.0.0.1:8765``; the address is determined by the server launch
command, not by the app factory.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from moaxy.adapters.registry import AdapterRegistry, build_registry
from moaxy.config.loader import load_config
from moaxy.models.config import MoaxyConfig
from moaxy.routing.matcher import RouteMatcher
from moaxy.server.auth_gate import (
    AuthGateMiddleware,
    build_principal_index,
)
from moaxy.server.errors import (
    MethodNotAllowedError,
    register_error_handlers,
)
from moaxy.server.middleware import (
    RequestIdMiddleware,
    StructuredLoggingMiddleware,
    TimingMiddleware,
)
from moaxy.server.routes import router as routes_router

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set up the moaxy loggers so structured access log lines are visible.

    uvicorn configures only its own ``uvicorn.*`` loggers via
    ``LOGGING_CONFIG``. The root logger stays at ``WARNING`` with no
    handlers, so the ``moaxy.server.access`` log lines emitted from
    :class:`StructuredLoggingMiddleware` are dropped on the floor.

    The fix is to attach a stderr stream handler to the ``moaxy`` parent
    logger and set its level to ``INFO``. The handler is attached only
    if the parent logger has no handlers of its own, so callers that
    pre-configure logging (e.g. tests with ``caplog``) are not
    overridden.
    """
    parent = logging.getLogger("moaxy")
    if not parent.handlers and not parent.parent.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    logging.getLogger("moaxy.server.access").setLevel(logging.INFO)
    logging.getLogger("moaxy.server").setLevel(logging.INFO)


def create_app(
    config: MoaxyConfig | None = None,
    adapters: AdapterRegistry | None = None,
    *,
    plugins_dir: str | None = None,
) -> FastAPI:
    """Build and return a configured :class:`fastapi.FastAPI` instance.

    Args:
        config: Pre-parsed :class:`MoaxyConfig`. When ``None`` the factory
            calls :func:`moaxy.config.loader.load_config` to discover the
            on-disk config file (``MOAXY_CONFIG_PATH`` → ``config.yaml``
            → ``config.yml`` → ``config.json`` → defaults).
        adapters: Pre-built :class:`AdapterRegistry`. When ``None`` the
            factory calls :func:`moaxy.adapters.registry.build_registry`
            with the parsed config's ``backends`` list.
        plugins_dir: Override for the plugin discovery directory. When
            ``None`` the factory uses ``config.plugins.plugins_dir``. M1
            does not yet invoke the plugin manager from the request
            pipeline; the value is preserved on ``app.state`` for
            downstream milestones.

    Returns:
        A fully configured :class:`fastapi.FastAPI` app. The same app
        can be passed to ``uvicorn`` (``uvicorn moaxy.server.app:create_app
        --factory``) or used in-process via ``httpx.AsyncClient`` with
        ``ASGITransport``.
    """
    if config is None:
        config = load_config()
    if adapters is None:
        adapters = build_registry(config.backends)

    _configure_logging()

    app = FastAPI(
        title="moaxy",
        version="0.1.0",
        default_response_class=JSONResponse,
    )

    app.state.config = config
    app.state.adapters = adapters
    app.state.route_matcher = RouteMatcher(config)
    app.state.plugins_dir = (
        plugins_dir
        if plugins_dir is not None
        else config.plugins.plugins_dir
    )

    _install_middleware(app, config)
    _install_routes(app)
    _install_method_not_allowed_handlers(app)
    register_error_handlers(app)

    return app


def _install_middleware(app: FastAPI, config: MoaxyConfig) -> None:
    """Mount the server-level middlewares in the correct order.

    Starlette's ``add_middleware`` installs middlewares in REVERSE order
    of the calls, so the first call below is the OUTERMOST middleware
    in the request lifecycle. We want the request id to be set FIRST
    (outermost) so the timing and logging middlewares can read it
    from ``request.state``.

    The order of mounting (top → bottom of the call list) is:

    * StructuredLoggingMiddleware (outermost)
    * TimingMiddleware
    * RequestIdMiddleware
    * AuthGateMiddleware (innermost of the user-installed middlewares)

    On the request path, this means: id is attached, then start_time,
    then the auth gate validates the API key (when ``auth.enabled``
    is true), then the route handler runs. On the response path, the
    route handler returns, the auth gate (if it ran) returns
    normally, the timing middleware adds ``x-moaxy-time-ms``, the
    request id middleware adds ``x-moaxy-request-id``, and finally
    the logging middleware emits a log line.

    The :class:`AuthGateMiddleware` is only installed when
    ``config.auth is not None and config.auth.enabled``; otherwise
    the no-op overhead is avoided and the data plane runs as before.
    """
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    if config.auth is not None and config.auth.enabled:
        header_names = tuple(config.auth.header_names) or (
            "X-API-Key",
            "Authorization",
        )
        exempt_paths = tuple(config.auth.exempt_paths) or ("/health",)
        principal_index = build_principal_index(config.auth.api_keys)
        # Install as a USER MIDDLEWARE (not a route middleware) so
        # it wraps the BaseHTTPMiddleware chain. When the auth
        # gate short-circuits with a 401, the BaseHTTPMiddleware
        # middlewares (request id, timing, logging) never run,
        # but the gate itself attaches the request id header
        # directly. See auth_gate.py for the rationale.
        app.add_middleware(
            AuthGateMiddleware,
            principal_index=principal_index,
            header_names=header_names,
            exempt_paths=exempt_paths,
        )


def _install_routes(app: FastAPI) -> None:
    """Mount the top-level routes package on the app."""
    app.include_router(routes_router)


def _install_method_not_allowed_handlers(app: FastAPI) -> None:
    """Add 405 Method Not Allowed responses for the data-plane endpoints.

    FastAPI returns 405 only when an exact path is matched by a route
    but the method is wrong. To guarantee a 405 (and never a 404) for
    well-known paths, we add explicit POST/GET handlers on the
    opposite-method case for ``/health`` and ``/v1/chat/completions``.
    """

    @app.post("/health", include_in_schema=False)
    async def _health_post(request: Request) -> JSONResponse:
        raise MethodNotAllowedError(
            "POST is not allowed on /health; use GET",
            details={"method": "POST", "path": "/health", "allowed": ["GET"]},
        )

    @app.get("/v1/chat/completions", include_in_schema=False)
    async def _chat_completions_get(request: Request) -> JSONResponse:
        raise MethodNotAllowedError(
            "GET is not allowed on /v1/chat/completions; use POST",
            details={
                "method": "GET",
                "path": "/v1/chat/completions",
                "allowed": ["POST"],
            },
        )


__all__ = ["create_app"]
