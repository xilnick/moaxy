"""Route matching and alias resolution for moaxy.

The :class:`RouteMatcher` walks a route table in declaration (YAML file)
order and returns the first route whose :class:`~moaxy.models.config.RouteMatch`
glob patterns match the incoming request's ``model`` and ``path``. When no
route matches, :meth:`RouteMatcher.match` returns ``None`` so the server
can answer with a structured 404/502.

A successful match yields a :class:`RouteMatch` value object that carries
every piece of state the orchestrator needs to dispatch a request:

* ``original_model`` — the model name the client sent (kept verbatim for
  response echo and ``x-moaxy-*`` headers).
* ``resolved_model`` — the alias-resolved real model name. If the request
  model is in the route's ``aliases`` map, the value is rewritten; if not,
  the original model is passed through unchanged. A miss never raises.
* ``backend`` — the route's configured backend name; looked up in
  :class:`~moaxy.adapters.registry.AdapterRegistry`.
* ``reflection``, ``advisor``, ``fallbacks``, ``retry``, ``aliases`` —
  copied verbatim from the matched route's :class:`RouteConfig`.

Glob matching uses :mod:`fnmatch` (POSIX shell-style). The literal pattern
``"*"`` matches any non-empty string; the pattern ``"minimax-*"`` matches
any string with that prefix; ``"/v1/chat/completions"`` is treated as an
exact match. Empty request fields never match.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Mapping

from moaxy.models.config import (
    AdvisorConfig,
    MoaxyConfig,
    ReflectionConfig,
    RouteConfig,
)


@dataclass(frozen=True)
class RouteMatch:
    """The routing decision for a single request.

    The result is a plain value object: it is built once per request and
    passed through the orchestrator pipeline. It is frozen so accidental
    mutation by downstream code is caught immediately.
    """

    route: RouteConfig
    """The :class:`RouteConfig` that matched the request."""

    original_model: str
    """The model name the client sent (e.g. ``"coder-pro"``)."""

    resolved_model: str
    """The alias-resolved real model name (e.g. ``"minimax-m3:cloud"``)."""

    backend: str | None
    """The route's configured backend name, or ``None`` for multi-backend routes."""

    path: str
    """The request path the matcher evaluated."""

    reflection: ReflectionConfig
    """Reflection loop configuration copied from the matched route."""

    advisor: AdvisorConfig
    """Advisor stage configuration copied from the matched route."""

    fallbacks: list[str]
    """Per-route fallback model list (a copy, free to mutate)."""

    retry: int
    """Per-route retry budget."""

    aliases: dict[str, str]
    """Alias-to-real-name map from the matched route (a copy)."""


class RouteMatcher:
    """First-match-wins route matcher with fnmatch glob support.

    The matcher is constructed from a :class:`MoaxyConfig` (or any object
    that exposes a ``routes`` list of :class:`RouteConfig`). The matcher
    is immutable from the caller's perspective: it caches the route list
    snapshot at construction time so reloading the config produces a new
    matcher instance, not a mutation of an existing one.

    The :meth:`match` method is synchronous and pure: it does not perform
    I/O and does not mutate the route table. A request is the minimal
    shape ``{"model": <str>, "path": <str>}``; any extra fields on the
    request are ignored.
    """

    def __init__(self, config: MoaxyConfig | Mapping[str, Any] | list[RouteConfig]) -> None:
        """Build a matcher from a config object, a config-like mapping, or a route list.

        Args:
            config: One of:
                * a :class:`MoaxyConfig` instance — the production path.
                * a mapping with a ``"routes"`` key — useful for tests that
                  want to skip parsing a full config.
                * a list of :class:`RouteConfig` instances — the lowest-level
                  form, used by tests that already have a route list.
        """
        if isinstance(config, list):
            self._routes: list[RouteConfig] = list(config)
        elif isinstance(config, MoaxyConfig):
            self._routes = list(config.routes)
        elif isinstance(config, Mapping) and "routes" in config:
            self._routes = list(config["routes"])
        else:
            raise TypeError(
                "RouteMatcher expects a MoaxyConfig, a mapping with a 'routes' key, "
                f"or a list of RouteConfig; got {type(config).__name__}"
            )

    @property
    def routes(self) -> list[RouteConfig]:
        """Return the snapshot of routes this matcher was built with."""
        return list(self._routes)

    def match(self, request: Mapping[str, Any]) -> RouteMatch | None:
        """Return the first matching :class:`RouteMatch`, or ``None``.

        A request matches a route when BOTH ``request["model"]`` matches
        the route's ``match.model`` glob AND ``request["path"]`` matches
        the route's ``match.path`` glob. ``fnmatch`` is used; the literal
        ``"*"`` matches any non-empty string. An empty or missing model
        or path on either side prevents the match.

        Alias resolution happens at match time, not at construction time,
        so the matcher always reflects the routes' current alias table.

        Args:
            request: A mapping with at least ``"model"`` and ``"path"``
                string fields. Extra keys are ignored.

        Returns:
            A :class:`RouteMatch` if a route matches, otherwise ``None``.
        """
        model = request.get("model")
        path = request.get("path")
        if not isinstance(model, str) or not model:
            return None
        if not isinstance(path, str) or not path:
            return None

        for route in self._routes:
            if not self._route_matches(route, model=model, path=path):
                continue
            return self._build_route_match(route, original_model=model, path=path)
        return None

    @staticmethod
    def _route_matches(route: RouteConfig, *, model: str, path: str) -> bool:
        pattern_model = route.match.model
        pattern_path = route.match.path
        if not pattern_model or not pattern_path:
            return False
        if not fnmatch(model, pattern_model):
            return False
        if not fnmatch(path, pattern_path):
            return False
        return True

    @staticmethod
    def _build_route_match(
        route: RouteConfig,
        *,
        original_model: str,
        path: str,
    ) -> RouteMatch:
        resolved_model = route.aliases.get(original_model, original_model)
        return RouteMatch(
            route=route,
            original_model=original_model,
            resolved_model=resolved_model,
            backend=route.backend,
            path=path,
            reflection=route.reflection,
            advisor=route.advisor,
            fallbacks=list(route.fallbacks),
            retry=route.retry,
            aliases=dict(route.aliases),
        )


__all__ = ["RouteMatch", "RouteMatcher"]
