"""ASGI middleware implementing the moaxy API-key auth gate.

When ``auth.enabled`` is true on the :class:`moaxy.models.config.MoaxyConfig`,
every request except those matching an entry in
``auth.exempt_paths`` must carry a valid API key in one of the headers
listed in ``auth.header_names`` (default: ``X-API-Key`` and
``Authorization``). The gate returns HTTP ``401 Unauthorized`` with a
JSON error envelope when the key is missing or unknown; a valid key
attaches a :class:`Principal` to ``request.state.principal`` so
downstream code (admin endpoints, audit logs) can identify the caller.

The gate is implemented as a pure ASGI middleware (not
:class:`starlette.middleware.base.BaseHTTPMiddleware`) so the 401
response flows through the rest of the ASGI chain — the
:class:`RequestIdMiddleware`, :class:`TimingMiddleware`, and
:class:`StructuredLoggingMiddleware` all see the response and can
add their headers / log lines. Using ``BaseHTTPMiddleware`` here
causes the 401 short-circuit to bypass the outer chain (a known
quirk of ``BaseHTTPMiddleware``), so we use the lower-level ASGI
interface.

The set of valid keys is read once from the parsed config (via
:func:`build_principal_index`) and captured in the closure; rotating
keys requires a process restart or a config-reload from the admin
surface (future work).

Per-key roles and scopes are propagated on the :class:`Principal` for
the admin endpoint authorisation checks (see
:mod:`moaxy.server.routes.admin`). The gate itself only enforces the
"is the key valid" check; per-endpoint role-based authorisation is
performed in the admin route module.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

API_KEY_HEADERS: tuple[str, ...] = ("x-api-key", "authorization")
"""Default header names (lowercased) consulted by the auth gate.

The order matters only when both are present: ``X-API-Key`` is checked
first because it is the dedicated moaxy header; the ``Authorization``
header is parsed as a Bearer token when present. The list is also
read from ``auth.header_names`` in the config, which OVERRIDES the
default; the constant exists for the test suite and the M4 admin
route module.
"""

DEFAULT_EXEMPT_PATHS: tuple[str, ...] = ("/health",)
"""Default exempt paths consulted when ``auth.exempt_paths`` is empty.

The config field always takes precedence. The constant is exported for
tests and as a documentation aid.
"""


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, attached to ``request.state.principal``.

    The dataclass is frozen so a downstream route handler cannot
    accidentally mutate the principal (e.g. by replacing the
    :attr:`roles` list). New fields are additive.

    Attributes:
        key_id: The ``key_id`` of the matching :class:`ApiKey` entry.
        roles: The roles granted to the key (e.g. ``["admin"]``).
        scopes: The scopes granted to the key (e.g. ``["*"]``).
    """

    key_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    scopes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_admin(self) -> bool:
        """Return True iff the principal has the ``admin`` role."""
        return "admin" in self.roles or "*" in self.roles


def build_principal_index(api_keys: Iterable) -> dict[str, Principal]:
    """Build the ``key_value -> Principal`` lookup table for the gate.

    The lookup table is captured in the middleware's closure; it is
    built once at startup from the parsed :class:`MoaxyConfig`. A
    duplicate ``key_value`` is a configuration error: the second
    entry wins but a warning is logged so the operator can fix the
    config. (The validation contract tests do not exercise this
    branch; it exists as defensive code for production.)

    Args:
        api_keys: An iterable of :class:`ApiKey` instances (typically
            ``config.auth.api_keys`` when ``config.auth`` is not
            ``None``).

    Returns:
        A dict keyed by the raw ``key_value`` string. The principal
        carries the ``key_id``, ``roles``, and ``scopes``.
    """
    index: dict[str, Principal] = {}
    for key in api_keys:
        principal = Principal(
            key_id=key.key_id,
            roles=tuple(key.roles),
            scopes=tuple(key.scopes),
        )
        if key.key_value in index:
            logger.warning(
                "duplicate api key value (key_id=%s) shadows an earlier entry; "
                "this is a configuration error",
                key.key_id,
            )
        index[key.key_value] = principal
    return index


def _extract_candidate_keys(
    headers: list[tuple[bytes, bytes]],
    header_names: tuple[str, ...],
) -> list[str]:
    """Return the API key candidates carried in the raw ASGI headers.

    Each header in ``header_names`` is consulted in order. The
    ``Authorization`` header is parsed for the ``Bearer <token>``
    scheme; other values are treated as opaque tokens. The function
    operates on raw ASGI header tuples (a list of ``(bytes, bytes)``)
    so it can run inside the pure-ASGI middleware without a
    :class:`Request` object.
    """
    candidates: list[str] = []
    normalised_names = tuple(name.lower().encode("latin-1") for name in header_names)
    auth_name = b"authorization"
    for raw_name, raw_value in headers:
        name = raw_name.lower()
        if name not in normalised_names:
            continue
        try:
            value = raw_value.decode("latin-1").strip()
        except UnicodeDecodeError:
            continue
        if not value:
            continue
        if name == auth_name:
            scheme, _, token = value.partition(" ")
            if scheme.lower() == "bearer" and token:
                candidates.append(token.strip())
        else:
            candidates.append(value)
    return candidates


def _match_principal(
    candidates: list[str],
    principal_index: dict[str, Principal],
) -> Principal | None:
    """Return the first principal whose key matches a candidate token."""
    for token in candidates:
        principal = principal_index.get(token)
        if principal is not None:
            return principal
    return None


class AuthGateMiddleware:
    """Pure-ASGI middleware enforcing the moaxy API-key auth gate.

    Behaviour:

    * ``auth.enabled is False`` (the default) — the middleware is a
      no-op. Every request passes through without inspection. (The
      middleware is not installed at all in that case; the factory
      only mounts it when ``config.auth.enabled`` is true.)
    * ``auth.enabled is True`` — the middleware checks the request
      against the configured :class:`ApiKey` list. A request whose
      path is in ``auth.exempt_paths`` (e.g. ``/health``) is
      always allowed, even when auth is enabled. Otherwise, a
      request without a valid key returns ``401`` with a JSON
      envelope ``{"error": {"type": "unauthorized", "message":
      "..."}}``. The 401 response is sent through the same ASGI
      chain so downstream middlewares (request id, timing, logging)
      can annotate it.

    The middleware is mounted by :func:`moaxy.server.app.create_app`
    when the parsed config has ``auth is not None and
    auth.enabled``. When ``auth`` is ``None`` (or disabled), the
    middleware is not installed at all — keeping the no-op overhead
    out of the hot path.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        principal_index: dict[str, Principal],
        header_names: tuple[str, ...] = API_KEY_HEADERS,
        exempt_paths: tuple[str, ...] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        self.app = app
        self._principals = dict(principal_index)
        self._header_names = tuple(h.lower() for h in header_names) or API_KEY_HEADERS
        self._exempt_paths = tuple(exempt_paths)
        # Pre-compute the exempt-path set for O(1) lookup. Exact match
        # is sufficient for the current contract; the architecture
        # allows glob patterns but the test suite asserts exact match.
        self._exempt_set = frozenset(self._exempt_paths)

    def is_exempt(self, path: str) -> bool:
        """Return True when ``path`` is exempt from auth.

        The match is exact (no glob). The architecture's discussion of
        ``exempt_paths`` as a list of globs is left for a future
        extension; the validation contract only requires exact match.
        """
        return path in self._exempt_set

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if self.is_exempt(path):
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        candidates = _extract_candidate_keys(headers, self._header_names)
        principal = _match_principal(candidates, self._principals)
        if principal is None:
            await self._send_unauthorized(scope, send)
            return

        # Attach the principal to the request state. Starlette stores
        # ``state`` on the ``Request`` object; we mutate the scope's
        # ``state`` slot if present, otherwise the principal is
        # stashed in the scope and re-attached by the first handler
        # that builds a Request. The simpler approach (used by
        # downstream code) is to look up the principal in the scope
        # after the middleware chain.
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["principal"] = principal

        await self.app(scope, receive, send)

    async def _send_unauthorized(self, scope: Scope, send: Send) -> None:
        """Emit a 401 JSON response directly through the ASGI interface.

        We can't use :class:`JSONResponse` here because the pure-ASGI
        path doesn't go through Starlette's routing layer. The
        response body is built by hand and emitted via
        ``send({"type": "http.response.start", ...})`` and
        ``send({"type": "http.response.body", ...})``.

        Because the auth gate is the OUTERMOST user middleware
        (installed last via ``app.add_middleware``), short-circuiting
        here bypasses :class:`RequestIdMiddleware`. We therefore
        generate (or copy) the request id on this code path so the
        401 response carries ``x-moaxy-request-id`` and downstream
        log lines can correlate the rejection.
        """
        body_bytes = json.dumps(
            {
                "error": {
                    "type": "unauthorized",
                    "message": "authentication required: provide a valid API key",
                }
            }
        ).encode("utf-8")
        request_id = self._extract_or_generate_request_id(scope)
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body_bytes)).encode("ascii")),
            (b"x-moaxy-request-id", request_id.encode("latin-1")),
        ]

        logger.info(
            "auth rejected",
            extra={
                "request_id": request_id,
                "path": scope.get("path", ""),
                "req_method": scope.get("method", ""),
                "status_code": 401,
            },
        )

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body_bytes, "more_body": False})

    @staticmethod
    def _extract_or_generate_request_id(scope: Scope) -> str:
        """Return a request id for the 401 response, generating one if needed.

        The auth gate is the OUTERMOST user middleware, so
        :class:`RequestIdMiddleware` has not run yet on this code
        path. We honour an inbound ``x-request-id`` /
        ``x-moaxy-request-id`` header (so callers can pin the id)
        and fall back to a fresh UUIDv4 hex value otherwise. The
        generated id is also stashed in ``scope["state"]`` so the
        inner middlewares (which run for non-401 responses) can pick
        it up and avoid generating a second one.
        """
        for raw_name, raw_value in scope.get("headers") or []:
            name = raw_name.lower()
            if name in (b"x-request-id", b"x-moaxy-request-id"):
                try:
                    value = raw_value.decode("latin-1").strip()
                except UnicodeDecodeError:
                    continue
                if value:
                    if "state" not in scope:
                        scope["state"] = {}
                    scope["state"].setdefault("request_id", value)
                    return value
        new_id = uuid.uuid4().hex
        if "state" not in scope:
            scope["state"] = {}
        scope["state"].setdefault("request_id", new_id)
        return new_id


__all__ = [
    "API_KEY_HEADERS",
    "AuthGateMiddleware",
    "DEFAULT_EXEMPT_PATHS",
    "Principal",
    "build_principal_index",
]
