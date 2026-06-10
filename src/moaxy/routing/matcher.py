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
* ``backend`` — the route's selected backend name, looked up in
  :class:`~moaxy.adapters.registry.AdapterRegistry`. For a single-backend
  route (``strategy: single``) this is the route's ``backend`` field.
  For a multi-backend route (``strategy: weighted`` or
  ``strategy: round_robin``) this is the name selected from
  ``route.backends`` by the per-route backend selector.
* ``reflection``, ``advisor``, ``aliases`` — copied verbatim from the
  matched route's :class:`RouteConfig`.
* ``fallbacks`` and ``retry`` — the **effective** values after applying
  the per-route override of the global ``cfg.models`` defaults (see
  :func:`_resolve_fallbacks` and :func:`_resolve_retry`). The raw route
  values remain available on :attr:`RouteMatch.route`.

Glob matching uses :mod:`fnmatch` (POSIX shell-style). The literal pattern
``"*"`` matches any non-empty string; the pattern ``"minimax-*"`` matches
any string with that prefix; ``"/v1/chat/completions"`` is treated as an
exact match. Empty request fields never match.

Per-route override of the global models defaults
------------------------------------------------

The matcher's contract is that per-route configuration always wins over
the global ``cfg.models`` defaults:

* If a route declares a non-empty ``fallbacks`` list, the walker uses
  that list verbatim. An empty ``fallbacks: []`` (explicitly set) is
  also treated as an override and means "no fallbacks".
* If a route does NOT declare ``fallbacks`` at all, the walker falls
  back to ``cfg.models.fallbacks[resolved_model]``.
* If a route declares a ``retry`` value (including ``retry: 0``), the
  walker uses that value. An absent ``retry`` field falls back to
  ``cfg.models.retry[resolved_model]`` (or ``0`` if the model is not
  in the table).

The "absent" detection uses Pydantic v2's :attr:`BaseModel.model_fields_set`
attribute, which records the field names the user explicitly provided at
construction time. This means the matcher preserves the distinction
between "field was set to its default value" and "field was left
uninitialised". The orchestrator reads only the effective values from
:attr:`RouteMatch.fallbacks` and :attr:`RouteMatch.retry`; the raw route
fields are still available via :attr:`RouteMatch.route` for callers that
need to distinguish between the two.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from moaxy.models.config import (
    AdvisorConfig,
    MoaxyConfig,
    ModelDefaults,
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
    """The route's selected backend name, or ``None`` when no backend is configured.

    For ``strategy: single`` routes, this is the route's ``backend``
    field verbatim. For ``strategy: weighted`` and
    ``strategy: round_robin`` routes, this is the name selected from
    ``route.backends`` by the matcher's per-route selector. The
    selector's state lives on the matcher; the value is captured here
    so the orchestrator and HTTP handler see a stable string for the
    lifetime of the request.
    """

    path: str
    """The request path the matcher evaluated."""

    reflection: ReflectionConfig
    """Reflection loop configuration copied from the matched route."""

    advisor: AdvisorConfig
    """Advisor stage configuration copied from the matched route."""

    fallbacks: list[str]
    """Effective fallback model list (a copy, free to mutate).

    The list is the result of applying the per-route override: if the
    route declared a ``fallbacks`` field (even ``fallbacks: []``), that
    value is used verbatim. Otherwise, the matcher falls back to
    ``cfg.models.fallbacks[resolved_model]``. When neither source has
    an entry, the list is empty.

    The raw route value (before the override) is available at
    :attr:`RouteMatch.route.fallbacks` for callers that need to
    distinguish "route set an empty list" from "route did not set
    fallbacks at all".
    """

    retry: int
    """Effective retry budget.

    The value is the result of applying the per-route override: if the
    route declared a ``retry`` field (including ``retry: 0``), that
    value is used. Otherwise, the matcher falls back to
    ``cfg.models.retry[resolved_model]``. When neither source has an
    entry, the budget is ``0``.

    The raw route value (before the override) is available at
    :attr:`RouteMatch.route.retry`.
    """

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

    def __init__(
        self,
        config: MoaxyConfig | Mapping[str, Any] | list[RouteConfig],
        *,
        random_seed: int | None = None,
    ) -> None:
        """Build a matcher from a config object, a config-like mapping, or a route list.

        Args:
            config: One of:
                * a :class:`MoaxyConfig` instance — the production path.
                  The matcher snapshots the route list and the
                  ``models`` defaults (used for the per-route override
                  of ``fallbacks`` and ``retry``).
                * a mapping with a ``"routes"`` key — useful for tests
                  that want to skip parsing a full config. The
                  optional ``"models"`` key, if present, must be a
                  :class:`ModelDefaults`-shaped mapping (with
                  ``"fallbacks"`` and/or ``"retry"`` dicts) and is used
                  for the override. Without it, only the route's own
                  ``fallbacks``/``retry`` are honoured.
                * a list of :class:`RouteConfig` instances — the
                  lowest-level form, used by tests that already have a
                  route list. No ``models`` defaults are available;
                  the matcher uses each route's own fields as-is.
            random_seed: Optional seed for the internal
                :class:`random.Random` instance used by the
                ``weighted`` strategy. When set, weighted selection is
                deterministic across processes (useful for tests and
                for reproducible load distribution). When omitted, the
                matcher uses the platform-default random source.
        """
        if isinstance(config, list):
            self._routes: list[RouteConfig] = list(config)
            self._models_defaults = ModelDefaults()
        elif isinstance(config, MoaxyConfig):
            self._routes = list(config.routes)
            self._models_defaults = config.models
        elif isinstance(config, Mapping) and "routes" in config:
            self._routes = list(config["routes"])
            models_payload = config.get("models")
            if isinstance(models_payload, ModelDefaults):
                self._models_defaults = models_payload
            elif isinstance(models_payload, Mapping):
                self._models_defaults = ModelDefaults.model_validate(models_payload)
            else:
                self._models_defaults = ModelDefaults()
        else:
            raise TypeError(
                "RouteMatcher expects a MoaxyConfig, a mapping with a 'routes' key, "
                f"or a list of RouteConfig; got {type(config).__name__}"
            )
        self._selectors: dict[str, _BackendSelector] = {}
        self._random_seed = random_seed

    @property
    def routes(self) -> list[RouteConfig]:
        """Return the snapshot of routes this matcher was built with."""
        return list(self._routes)

    def add_route(self, route: RouteConfig) -> None:
        """Append a new :class:`RouteConfig` to the in-memory route list.

        The route is appended to the end of the list, so it is consulted
        AFTER every pre-existing route. The matcher is otherwise
        immutable: there is no "insert at position" or "replace by
        name" operation, only append and remove. The first-match-wins
        rule from :meth:`match` means a new route never preempts an
        existing one — to "redefine" a route, the caller must
        :meth:`remove_route` first.

        Validation: the new route's name must be unique. A duplicate
        name raises :class:`ValueError` because the data model already
        requires uniqueness at the config level (the same constraint
        is enforced here for runtime CRUD). The caller is expected to
        catch the exception and translate it into the appropriate
        HTTP error envelope.

        A fresh :class:`_BackendSelector` is created for the new route
        with its own round-robin counter and weighted-selection state.
        Existing routes' selectors are untouched, so the round-robin
        cycle and weighted distribution continue uninterrupted.
        """
        if any(existing.name == route.name for existing in self._routes):
            raise ValueError(
                f"route {route.name!r} already exists; use a different name "
                "or remove the existing route first"
            )
        self._routes.append(route)
        self._selectors[route.name] = _BackendSelector(
            route, random_seed=self._random_seed
        )

    def remove_route(self, name: str) -> bool:
        """Remove a route by name.

        Returns ``True`` when a route with the given name was found
        and removed; ``False`` when no such route exists. The matcher
        performs a linear scan; routes are small (typically < 100
        entries), so the cost is negligible.

        The route's per-route :class:`_BackendSelector` is dropped
        along with the route. Adding a new route with the same name
        later starts a fresh selection cycle.
        """
        for index, existing in enumerate(self._routes):
            if existing.name == name:
                del self._routes[index]
                self._selectors.pop(name, None)
                return True
        return False

    def match(self, request: Mapping[str, Any]) -> RouteMatch | None:
        """Return the first matching :class:`RouteMatch`, or ``None``.

        A request matches a route when BOTH ``request["model"]`` matches
        the route's ``match.model`` glob AND ``request["path"]`` matches
        the route's ``match.path`` glob. ``fnmatch`` is used; the literal
        ``"*"`` matches any non-empty string. An empty or missing model
        or path on either side prevents the match.

        Alias resolution happens at match time, not at construction time,
        so the matcher always reflects the routes' current alias table.

        For a matched route, the returned :class:`RouteMatch`'s
        ``backend`` field is the name selected by the route's per-route
        :class:`_BackendSelector`:

        * ``strategy: single`` → ``route.backend`` (a single string).
        * ``strategy: weighted`` → a backend picked at random from
          ``route.backends``, proportional to each entry's ``weight``.
        * ``strategy: round_robin`` → the next entry in ``route.backends``
          in order, advancing the route's round-robin counter.

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
            selector = self._selectors.get(route.name)
            if selector is None:
                selector = _BackendSelector(route, random_seed=self._random_seed)
                self._selectors[route.name] = selector
            selected_backend = selector.select()
            return self._build_route_match(
                route,
                original_model=model,
                path=path,
                models_defaults=self._models_defaults,
                selected_backend=selected_backend,
            )
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
        models_defaults: ModelDefaults,
        selected_backend: str | None,
    ) -> RouteMatch:
        resolved_model = route.aliases.get(original_model, original_model)
        effective_fallbacks = _resolve_fallbacks(
            route, resolved_model=resolved_model, models_defaults=models_defaults
        )
        effective_retry = _resolve_retry(
            route, resolved_model=resolved_model, models_defaults=models_defaults
        )
        return RouteMatch(
            route=route,
            original_model=original_model,
            resolved_model=resolved_model,
            backend=selected_backend,
            path=path,
            reflection=route.reflection,
            advisor=route.advisor,
            fallbacks=effective_fallbacks,
            retry=effective_retry,
            aliases=dict(route.aliases),
        )


def _resolve_fallbacks(
    route: RouteConfig,
    *,
    resolved_model: str,
    models_defaults: ModelDefaults,
) -> list[str]:
    """Compute the effective fallback list for a route match.

    The per-route ``fallbacks`` list always wins when it has been
    explicitly set (including an explicit empty list, which means
    "no fallbacks"). The Pydantic v2 ``model_fields_set`` attribute
    is the source of truth for "explicitly set".

    When the route did not declare a ``fallbacks`` field, the matcher
    consults ``models_defaults.fallbacks[resolved_model]``. If neither
    source provides a list, the effective list is empty.

    Args:
        route: The matched :class:`RouteConfig`.
        resolved_model: The alias-resolved real model name. Used as
            the lookup key into ``models_defaults.fallbacks``.
        models_defaults: The :class:`ModelDefaults` snapshot captured
            by the matcher at construction time.

    Returns:
        A fresh list (always a copy) of fallback model names. The
        caller is free to mutate it without affecting the underlying
        route or the matcher's internal state.
    """
    if "fallbacks" in route.model_fields_set:
        return list(route.fallbacks)
    return list(models_defaults.fallbacks.get(resolved_model, []))


def _resolve_retry(
    route: RouteConfig,
    *,
    resolved_model: str,
    models_defaults: ModelDefaults,
) -> int:
    """Compute the effective retry budget for a route match.

    The per-route ``retry`` value always wins when it has been
    explicitly set (including ``retry: 0``, which means "no
    retries"). The Pydantic v2 ``model_fields_set`` attribute is the
    source of truth for "explicitly set".

    When the route did not declare a ``retry`` field, the matcher
    consults ``models_defaults.retry[resolved_model]``. If neither
    source provides a budget, the effective budget is ``0``.

    Args:
        route: The matched :class:`RouteConfig`.
        resolved_model: The alias-resolved real model name. Used as
            the lookup key into ``models_defaults.retry``.
        models_defaults: The :class:`ModelDefaults` snapshot captured
            by the matcher at construction time.

    Returns:
        The effective retry budget (a non-negative integer).
    """
    if "retry" in route.model_fields_set:
        return int(route.retry)
    return int(models_defaults.retry.get(resolved_model, 0))


class _BackendSelector:
    """Per-route backend selection state for the multi-backend strategies.

    A :class:`RouteMatcher` owns one :class:`_BackendSelector` per
    :class:`RouteConfig`. The selector encapsulates the per-route
    state required by the ``weighted`` and ``round_robin`` strategies
    (a private :class:`random.Random` instance and a round-robin
    counter, respectively) and exposes a single :meth:`select` method
    that returns the backend name to use for the next match.

    The selector is intentionally NOT re-entrant safe across
    concurrent coroutines: Python's GIL makes each individual
    statement atomic, but a single round-robin ``select()`` is a
    short read-modify-write sequence. The moaxy data plane is
    structured as one :class:`RouteMatcher` per process and the
    per-route counter is shared across all requests that match the
    route. A worst-case race only causes occasional repeated
    selection of the same backend, which is benign for load
    distribution. Callers that need strict per-request
    determinism can construct a fresh matcher per request.
    """

    def __init__(
        self,
        route: RouteConfig,
        *,
        random_seed: int | None = None,
    ) -> None:
        self._route = route
        self._random = random.Random(random_seed)
        self._round_robin_index = 0

    def select(self) -> str | None:
        """Return the backend name to use for the next match.

        The selection algorithm depends on the route's ``strategy``:

        * ``single`` — returns ``route.backend`` (which may be
          ``None`` if the route has no single backend configured).
        * ``weighted`` — picks one entry of ``route.backends`` at
          random, with probability proportional to each entry's
          ``weight``. Degenerate inputs (empty ``backends`` or all
          zero weights) fall back to ``route.backend`` or the first
          entry's name, respectively, so the selector never crashes
          on malformed routes.
        * ``round_robin`` — returns the next entry of ``route.backends``
          in order, wrapping around at the end. The internal counter
          advances by one on every call. An empty ``backends`` list
          falls back to ``route.backend``.

        The round-robin counter is process-local; it is not shared
        across processes. Multi-process deployments will distribute
        independently from each other, which is the desired
        behaviour for a stateless proxy.
        """
        strategy = self._route.strategy
        if strategy == "weighted":
            return self._select_weighted()
        if strategy == "round_robin":
            return self._select_round_robin()
        # ``single`` and any future strategy: defer to the route's
        # configured single backend. This is the historical default.
        return self._route.backend

    def _select_weighted(self) -> str | None:
        backends = self._route.backends
        if not backends:
            return self._route.backend
        total_weight = sum(ref.weight for ref in backends)
        if total_weight <= 0:
            # All weights are zero (or the only entry has weight 0).
            # The selection is undefined; degrade to the first
            # backend so the call site still gets a usable name
            # rather than ``None``. The route's ``backends`` list
            # was already validated by Pydantic to have at least
            # one entry by the time we get here, so this fallback
            # is always safe.
            return backends[0].name
        target = self._random.uniform(0.0, total_weight)
        cumulative = 0.0
        for ref in backends:
            cumulative += ref.weight
            if target < cumulative:
                return ref.name
        # Floating-point edge: target == total_weight (the loop's
        # last strict-less-than check failed). Return the last
        # entry's name to keep the selection well-defined.
        return backends[-1].name

    def _select_round_robin(self) -> str | None:
        backends = self._route.backends
        if not backends:
            return self._route.backend
        index = self._round_robin_index % len(backends)
        self._round_robin_index += 1
        return backends[index].name


__all__ = ["RouteMatch", "RouteMatcher"]
