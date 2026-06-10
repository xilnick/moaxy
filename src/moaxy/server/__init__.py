"""HTTP server for the moaxy OpenAI-compatible proxy.

The :func:`moaxy.server.app.create_app` factory builds a configured
:class:`fastapi.FastAPI` app with all routes, middleware, and exception
handlers wired up. The server is a thin shim over the routing and adapter
layers; the heavy lifting (route matching, alias resolution, request
shaping) lives in :mod:`moaxy.routing` and :mod:`moaxy.adapters`.
"""

from moaxy.server.app import create_app

__all__ = ["create_app"]
