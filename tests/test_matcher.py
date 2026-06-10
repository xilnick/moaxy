"""Tests for the routing matcher (first-match-wins + alias resolution).

Covers the routing surface from the validation contract:
VAL-RT-001 through VAL-RT-006, VAL-RT-009, VAL-RT-010, VAL-RT-019
through VAL-RT-025, and VAL-CROSS-020.

The matcher is a pure function over the route table — no I/O, no adapter
calls — so the tests are deterministic and do not depend on Ollama.
"""

from __future__ import annotations

import pytest

from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    BackendRef,
    MoaxyConfig,
    ModelDefaults,
    ReflectionConfig,
    RouteConfig,
)
from moaxy.models.config import (
    RouteMatch as ConfigRouteMatch,
)
from moaxy.routing.matcher import RouteMatch, RouteMatcher

# ────────────────────────────────────────────────────────────────────
# Test helpers
# ────────────────────────────────────────────────────────────────────


def _backend(name: str = "olloma-local") -> AdapterConfig:
    """Build a minimal backend config for the tests below."""
    return AdapterConfig(name=name, adapter="ollama", base_url="http://127.0.0.1:11434")


def _route(
    *,
    name: str = "r",
    model_pattern: str = "*",
    path_pattern: str = "/v1/chat/completions",
    backend: str | None = "olloma-local",
    strategy: str = "single",
    backends: list[BackendRef] | None = None,
    aliases: dict[str, str] | None = None,
    fallbacks: list[str] | None = None,
    retry: int = 0,
    reflection: ReflectionConfig | None = None,
    advisor: AdvisorConfig | None = None,
) -> RouteConfig:
    """Build a single :class:`RouteConfig` with sensible defaults."""
    return RouteConfig(
        name=name,
        match=ConfigRouteMatch(model=model_pattern, path=path_pattern),
        strategy=strategy,  # type: ignore[arg-type]
        backend=backend,
        backends=backends or [],
        aliases=aliases or {},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=reflection or ReflectionConfig(),
        advisor=advisor or AdvisorConfig(),
    )


def _config(*routes: RouteConfig, backends: list[AdapterConfig] | None = None) -> MoaxyConfig:
    """Wrap a list of routes in a :class:`MoaxyConfig` with a single backend by default.

    Args:
        routes: The route list.
        backends: Optional list of :class:`AdapterConfig`. When
            omitted, a single backend named ``"olloma-local"`` is
            used. The default is fine for single-backend routes; the
            multi-backend strategy tests pass an explicit list.
    """
    if backends is None:
        backends = [_backend()]
    return MoaxyConfig(backends=backends, routes=list(routes))


# ────────────────────────────────────────────────────────────────────
# VAL-RT-002: Glob match.model "*" matches any model
# ────────────────────────────────────────────────────────────────────


class TestGlobModelStar:
    """``"*"`` matches any non-empty model string."""

    @pytest.mark.parametrize(
        "model",
        [
            "minimax-m3:cloud",
            "minimax-m2.7:cloud",
            "deepseek-v4-pro:cloud",
            "kimi-k2.6:cloud",
            "any-model-name",
            "weird:name:format",
        ],
    )
    def test_star_matches_any_nonempty_model(self, model: str):
        cfg = _config(_route(model_pattern="*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": model, "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "r"
        assert result.original_model == model
        assert result.resolved_model == model

    def test_empty_model_does_not_match(self):
        cfg = _config(_route(model_pattern="*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "", "path": "/v1/chat/completions"})
        assert result is None

    def test_missing_model_does_not_match(self):
        cfg = _config(_route(model_pattern="*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"path": "/v1/chat/completions"})
        assert result is None


# ────────────────────────────────────────────────────────────────────
# VAL-RT-003: Glob match.model "minimax-*" matches prefix
# ────────────────────────────────────────────────────────────────────


class TestGlobModelPrefix:
    """``"minimax-*"`` matches the prefix only."""

    def test_prefix_matches_minimax_m3(self):
        cfg = _config(_route(model_pattern="minimax-*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "r"

    def test_prefix_matches_minimax_m2_7(self):
        cfg = _config(_route(model_pattern="minimax-*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m2.7:cloud", "path": "/v1/chat/completions"})
        assert result is not None

    def test_prefix_does_not_match_deepseek(self):
        cfg = _config(_route(model_pattern="minimax-*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "deepseek-v4-pro:cloud", "path": "/v1/chat/completions"}
        )
        assert result is None

    def test_prefix_does_not_match_kimi(self):
        cfg = _config(_route(model_pattern="minimax-*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "kimi-k2.6:cloud", "path": "/v1/chat/completions"})
        assert result is None

    def test_question_mark_glob_works(self):
        """``"a?c"`` matches ``"abc"`` (one char), not ``"ac"`` or ``"abbc"``."""
        cfg = _config(_route(name="q", model_pattern="a?c"))
        matcher = RouteMatcher(cfg)
        assert matcher.match({"model": "abc", "path": "/v1/chat/completions"}) is not None
        assert matcher.match({"model": "ac", "path": "/v1/chat/completions"}) is None
        assert matcher.match({"model": "abbc", "path": "/v1/chat/completions"}) is None

    def test_bracket_glob_works(self):
        """``"ab[cd]ef"`` matches ``"abcef"`` and ``"abdef"`` (one of c/d)."""
        cfg = _config(_route(name="b", model_pattern="ab[cd]ef"))
        matcher = RouteMatcher(cfg)
        assert matcher.match({"model": "abcef", "path": "/v1/chat/completions"}) is not None
        assert matcher.match({"model": "abdef", "path": "/v1/chat/completions"}) is not None
        assert matcher.match({"model": "abxef", "path": "/v1/chat/completions"}) is None


# ────────────────────────────────────────────────────────────────────
# VAL-RT-004: Glob match.path "/v1/chat/completions" matches the request path
# ────────────────────────────────────────────────────────────────────


class TestGlobPath:
    """Path matching is glob-style; ``/v1/chat/completions`` is exact."""

    def test_exact_path_matches(self):
        cfg = _config(_route(path_pattern="/v1/chat/completions"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.path == "/v1/chat/completions"

    def test_different_path_does_not_match(self):
        cfg = _config(_route(path_pattern="/v1/chat/completions"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/embeddings"})
        assert result is None

    def test_wildcard_path_matches_subpaths(self):
        """``"/v1/*"`` should match any path under ``/v1/``."""
        cfg = _config(_route(name="wild", path_pattern="/v1/*"))
        matcher = RouteMatcher(cfg)
        assert (
            matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
            is not None
        )
        assert (
            matcher.match({"model": "minimax-m3:cloud", "path": "/v1/embeddings"})
            is not None
        )

    def test_empty_path_does_not_match(self):
        cfg = _config(_route(path_pattern="/v1/chat/completions"))
        matcher = RouteMatcher(cfg)
        assert matcher.match({"model": "minimax-m3:cloud", "path": ""}) is None
        assert matcher.match({"model": "minimax-m3:cloud"}) is None


# ────────────────────────────────────────────────────────────────────
# VAL-RT-001: First-match-wins ordering of routes
# VAL-RT-023: First-match-wins respects YAML file order, not alphabetic
# ────────────────────────────────────────────────────────────────────


class TestFirstMatchWins:
    """First matching route in declaration order wins."""

    def test_first_matching_route_wins(self):
        star = _route(name="star", model_pattern="*")
        prefix = _route(name="prefix", model_pattern="minimax-*")
        cfg = _config(star, prefix)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "star"

    def test_yaml_order_not_alphabetic(self):
        """``Z-route`` declared first beats ``A-route`` declared second."""
        z_route = _route(name="Z-route", model_pattern="*")
        a_route = _route(name="A-route", model_pattern="*")
        cfg = _config(z_route, a_route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "Z-route"

    def test_prefix_beats_star_when_prefix_declared_first(self):
        prefix = _route(name="prefix", model_pattern="minimax-*")
        star = _route(name="star", model_pattern="*")
        cfg = _config(prefix, star)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "prefix"

    def test_more_specific_path_wins_when_first(self):
        completions = _route(name="completions", path_pattern="/v1/chat/completions")
        embeddings = _route(name="embeddings", path_pattern="/v1/embeddings")
        star = _route(name="star", path_pattern="/v1/*")
        cfg = _config(completions, embeddings, star)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "completions"
        result2 = matcher.match({"model": "m", "path": "/v1/embeddings"})
        assert result2 is not None
        assert result2.route.name == "embeddings"


# ────────────────────────────────────────────────────────────────────
# VAL-RT-005: No matching route returns None (server emits 404/502)
# ────────────────────────────────────────────────────────────────────


class TestNoMatch:
    """When no route matches, the matcher returns ``None``."""

    def test_no_routes_returns_none(self):
        cfg = _config()  # no routes
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is None

    def test_no_matching_model_returns_none(self):
        cfg = _config(_route(model_pattern="minimax-*"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "deepseek-v4-pro:cloud", "path": "/v1/chat/completions"})
        assert result is None

    def test_no_matching_path_returns_none(self):
        cfg = _config(_route(path_pattern="/v1/chat/completions"))
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/audio/speech"})
        assert result is None


# ────────────────────────────────────────────────────────────────────
# VAL-RT-006, VAL-RT-009, VAL-RT-010: Alias resolution semantics
# ────────────────────────────────────────────────────────────────────


class TestAliasResolution:
    """Alias map behaviour: hit, miss, empty."""

    def test_alias_hit_resolves_to_target(self):
        route = _route(
            aliases={"coder-pro": "minimax-m3:cloud", "reviewer": "deepseek-v4-pro:cloud"},
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.original_model == "coder-pro"
        assert result.resolved_model == "minimax-m3:cloud"
        assert result.aliases == {"coder-pro": "minimax-m3:cloud", "reviewer": "deepseek-v4-pro:cloud"}

    def test_alias_miss_passes_through_unchanged(self):
        """An unknown alias must NOT raise; the model is passed through."""
        route = _route(aliases={"coder-pro": "minimax-m3:cloud"})
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "unknown-alias-xyz", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.original_model == "unknown-alias-xyz"
        assert result.resolved_model == "unknown-alias-xyz"

    def test_empty_aliases_passes_through(self):
        route = _route(aliases={})
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.original_model == "minimax-m3:cloud"
        assert result.resolved_model == "minimax-m3:cloud"

    def test_default_aliases_is_empty(self):
        """A route with no aliases field defaults to ``{}``."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "minimax-m3:cloud", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.aliases == {}
        assert result.resolved_model == "minimax-m3:cloud"

    def test_alias_table_is_a_copy(self):
        """Mutating the returned ``aliases`` dict must not affect the route."""
        route = _route(aliases={"coder-pro": "minimax-m3:cloud"})
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert result is not None
        result.aliases["new"] = "deepseek"  # type: ignore[index]
        # Re-match and confirm the original route is untouched.
        result2 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert result2 is not None
        assert "new" not in result2.aliases
        assert result2.aliases == {"coder-pro": "minimax-m3:cloud"}

    def test_fallbacks_list_is_a_copy(self):
        """Mutating the returned ``fallbacks`` list must not affect the route."""
        route = _route(fallbacks=["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"])
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        result.fallbacks.append("kimi-k2.6:cloud")
        result2 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result2 is not None
        assert result2.fallbacks == ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"]


# ────────────────────────────────────────────────────────────────────
# VAL-RT-020: RouteMatch exposes backend
# VAL-RT-021: RouteMatch carries reflection, advisor, fallbacks, aliases, retry
# VAL-RT-022: RouteMatch.original_model and resolved_model
# ────────────────────────────────────────────────────────────────────


class TestRouteMatchFields:
    """The result exposes every piece of state from the matched route."""

    def test_result_exposes_backend_name(self):
        route = _route(backend="olloma-local")
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "olloma-local"

    def test_result_carries_reflection(self):
        reflection = ReflectionConfig(turns=2, early_exit=False, threshold=0.9, parallel=True)
        route = _route(reflection=reflection)
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.reflection.turns == 2
        assert result.reflection.early_exit is False
        assert result.reflection.threshold == 0.9
        assert result.reflection.parallel is True

    def test_result_carries_advisor(self):
        advisor = AdvisorConfig(model="deepseek-v4-pro:cloud", turns=1, parallel=False)
        route = _route(advisor=advisor)
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.advisor.model == "deepseek-v4-pro:cloud"
        assert result.advisor.turns == 1
        assert result.advisor.parallel is False

    def test_result_carries_fallbacks_and_retry(self):
        route = _route(
            fallbacks=["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"],
            retry=2,
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.fallbacks == ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"]
        assert result.retry == 2

    def test_result_carries_aliases(self):
        route = _route(aliases={"coder-pro": "minimax-m3:cloud"})
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.aliases == {"coder-pro": "minimax-m3:cloud"}

    def test_original_and_resolved_model(self):
        route = _route(aliases={"coder-pro": "minimax-m3:cloud"})
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.original_model == "coder-pro"
        assert result.resolved_model == "minimax-m3:cloud"

    def test_result_carries_path(self):
        route = _route()
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.path == "/v1/chat/completions"

    def test_result_is_frozen(self):
        """A :class:`RouteMatch` instance is immutable."""
        route = _route()
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        with pytest.raises((AttributeError, Exception)):
            result.original_model = "tampered"  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────
# VAL-RT-024: Empty routes list makes every request fail to match
# ────────────────────────────────────────────────────────────────────


class TestEmptyRouteList:
    def test_empty_routes_means_no_match(self):
        cfg = _config()
        matcher = RouteMatcher(cfg)
        assert matcher.match({"model": "m", "path": "/v1/chat/completions"}) is None
        assert matcher.match({"model": "anything", "path": "/anything"}) is None

    def test_matcher_from_empty_routes_list(self):
        matcher = RouteMatcher([])
        assert matcher.match({"model": "m", "path": "/v1/chat/completions"}) is None


# ────────────────────────────────────────────────────────────────────
# VAL-RT-019: Alias rewriting does not happen in fallback models
# (The matcher returns the raw fallback list from the route; alias
# resolution only applies to the request's ``model`` field, not to
# the entries in ``route.fallbacks``.)
# ────────────────────────────────────────────────────────────────────


class TestFallbacksNotAliased:
    def test_fallbacks_passed_through_verbatim(self):
        """The matcher's ``fallbacks`` list is a copy of the route's list.

        Alias resolution only rewrites ``request["model"]``; it never
        rewrites the strings inside ``route.fallbacks``.
        """
        route = _route(
            aliases={"coder-pro": "minimax-m3:cloud"},
            fallbacks=["minimax-m3:cloud", "deepseek-v4-pro:cloud"],
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.fallbacks == ["minimax-m3:cloud", "deepseek-v4-pro:cloud"]


# ────────────────────────────────────────────────────────────────────
# VAL-RT-025: Routing does not require auth to function
# (The matcher is a pure function with no auth dependency; auth lives
# in a separate middleware layer.)
# ────────────────────────────────────────────────────────────────────


class TestRoutingHasNoAuthDependency:
    def test_matcher_does_not_read_auth(self):
        """Building a matcher never inspects ``cfg.auth``."""
        cfg = MoaxyConfig(
            backends=[_backend()],
            routes=[_route()],
            auth=None,
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.route.name == "r"


# ────────────────────────────────────────────────────────────────────
# Construction variants
# ────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_build_from_moaxy_config(self):
        cfg = _config(_route(name="r1"), _route(name="r2"))
        matcher = RouteMatcher(cfg)
        assert len(matcher.routes) == 2
        assert [r.name for r in matcher.routes] == ["r1", "r2"]

    def test_build_from_route_list(self):
        routes = [_route(name="a"), _route(name="b")]
        matcher = RouteMatcher(routes)
        assert len(matcher.routes) == 2

    def test_build_from_mapping_with_routes_key(self):
        routes = [_route(name="a"), _route(name="b")]
        matcher = RouteMatcher({"routes": routes})
        assert len(matcher.routes) == 2

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError):
            RouteMatcher(42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            RouteMatcher("not a config")  # type: ignore[arg-type]

    def test_routes_property_returns_a_copy(self):
        """The ``routes`` property returns a defensive copy."""
        cfg = _config(_route(name="a"))
        matcher = RouteMatcher(cfg)
        snap = matcher.routes
        snap.append(_route(name="b"))
        assert len(matcher.routes) == 1

    def test_routes_list_immutability_across_matches(self):
        """Repeated matches return independent result objects."""
        cfg = _config(_route(fallbacks=["a", "b"]))
        matcher = RouteMatcher(cfg)
        r1 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        r2 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert r1 is not r2
        assert r1 == r2


# ────────────────────────────────────────────────────────────────────
# VAL-RT-015..018: models.fallbacks / models.retry defaults
# ────────────────────────────────────────────────────────────────────


def _config_with_models(
    *routes: RouteConfig,
    models: ModelDefaults | None = None,
) -> MoaxyConfig:
    """Wrap routes in a MoaxyConfig with optional models defaults."""
    return MoaxyConfig(
        backends=[_backend()],
        routes=list(routes),
        models=models or ModelDefaults(),
    )


class TestModelsFallbacksDefault:
    """``models.fallbacks[model]`` is used when the route omits ``fallbacks``."""

    def test_models_fallbacks_used_when_route_omits_field(self):
        """VAL-RT-015: global default kicks in when the route has no list.

        The route is built without ever passing ``fallbacks=`` to
        Pydantic, so the field is in the route's default state (i.e.
        it is NOT in ``model_fields_set``). The matcher falls back to
        ``models.fallbacks[resolved_model]``.
        """
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["deepseek-v4-pro:cloud"]},
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.fallbacks == ["deepseek-v4-pro:cloud"]
        # The raw route.fallbacks is still empty (no mutation).
        assert result.route.fallbacks == []
        # Confirm the field was NOT in the route's model_fields_set —
        # the test was specifically about the "absent" case.
        assert "fallbacks" not in route.model_fields_set

    def test_models_fallbacks_keyed_by_resolved_model(self):
        """The lookup uses the alias-resolved model name, not the client's."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["deepseek-v4-pro:cloud"]},
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "coder-pro", "path": "/v1/chat/completions"}
        )
        assert result is not None
        # The fallback list comes from models.fallbacks[minimax-m3:cloud].
        assert result.fallbacks == ["deepseek-v4-pro:cloud"]
        assert result.resolved_model == "minimax-m3:cloud"

    def test_models_fallbacks_unrelated_model_yields_empty(self):
        """A model not in the table resolves to an empty fallback list."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={"some-other-model": ["x"]},
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.fallbacks == []

    def test_models_fallbacks_default_is_empty_dict(self):
        """With no models defaults at all, fallbacks is empty."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config_with_models(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.fallbacks == []


class TestRouteFallbacksOverride:
    """``route.fallbacks`` wins over ``models.fallbacks[model]``."""

    def test_route_fallbacks_wins_when_both_set(self):
        """VAL-RT-016: route.fallbacks wins over models.fallbacks[model]."""
        route = _route(fallbacks=["kimi-k2.6:cloud"])
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["deepseek-v4-pro:cloud"]},
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        # The route's list is the one used, not the models default.
        assert result.fallbacks == ["kimi-k2.6:cloud"]

    def test_route_fallbacks_empty_list_wins_over_models_default(self):
        """An explicit empty ``fallbacks: []`` on the route means no fallbacks.

        This is the "explicitly set to empty" path — the route author
        declared ``fallbacks: []`` deliberately, so the models default
        is NOT consulted.
        """
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            fallbacks=[],
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["deepseek-v4-pro:cloud"]},
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.fallbacks == []
        # The raw value is the empty list (explicitly set).
        assert result.route.fallbacks == []
        # Confirm the field WAS in model_fields_set.
        assert "fallbacks" in route.model_fields_set

    def test_route_fallbacks_multi_entry_wins(self):
        """A multi-entry route.fallbacks is used verbatim."""
        route = _route(
            fallbacks=["minimax-m2.7:cloud", "kimi-k2.6:cloud"],
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={
                    "minimax-m3:cloud": ["deepseek-v4-pro:cloud"],
                },
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.fallbacks == ["minimax-m2.7:cloud", "kimi-k2.6:cloud"]


class TestModelsRetryDefault:
    """``models.retry[model]`` is used when the route omits ``retry``."""

    def test_models_retry_used_when_route_omits_field(self):
        """VAL-RT-017: global retry budget kicks in when the route has none.

        The route is built without ever passing ``retry=`` to Pydantic,
        so the field is in the route's default state (i.e. it is NOT
        in ``model_fields_set``). The matcher falls back to
        ``models.retry[resolved_model]``.
        """
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(retry={"minimax-m3:cloud": 2}),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.retry == 2
        # The raw route.retry is still 0 (no mutation).
        assert result.route.retry == 0
        # Confirm the field was NOT in the route's model_fields_set.
        assert "retry" not in route.model_fields_set

    def test_models_retry_keyed_by_resolved_model(self):
        """The lookup uses the alias-resolved model name."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(retry={"minimax-m3:cloud": 3}),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "coder-pro", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.retry == 3

    def test_models_retry_unrelated_model_yields_zero(self):
        """A model not in the table resolves to a zero budget."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(retry={"some-other-model": 2}),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.retry == 0


class TestRouteRetryOverride:
    """``route.retry`` wins over ``models.retry[model]``."""

    def test_route_retry_zero_overrides_models_retry(self):
        """VAL-RT-018: route.retry=0 wins over models.retry=2."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            retry=0,
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(retry={"minimax-m3:cloud": 2}),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.retry == 0
        # The raw route value is also 0 (explicitly set).
        assert result.route.retry == 0
        # Confirm the field WAS in model_fields_set.
        assert "retry" in route.model_fields_set

    def test_route_retry_nonzero_overrides_models_retry(self):
        """A non-zero route.retry wins over a different models.retry."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            retry=5,
        )
        cfg = _config_with_models(
            route,
            models=ModelDefaults(retry={"minimax-m3:cloud": 2}),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"}
        )
        assert result is not None
        assert result.retry == 5


# ────────────────────────────────────────────────────────────────────
# Aliases are not re-applied to fallback model names
# (VAL-RT-019: alias map applies to the request's model field only,
# not to the entries in route.fallbacks or models.fallbacks.)
# ────────────────────────────────────────────────────────────────────


class TestModelsFallbacksNotReAliased:
    """The alias map is NOT re-applied to entries in models.fallbacks."""

    def test_models_fallbacks_passed_through_verbatim(self):
        """An alias target that happens to be a fallback key is unchanged.

        The alias map rewrites only ``request["model"]``; the entries
        in ``models.fallbacks[primary]`` are walked verbatim, even if
        their names happen to also be aliases on the same route.
        """
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            aliases={"coder-pro": "minimax-m3:cloud"},
        )
        # The models.fallbacks key "minimax-m3:cloud" is also an alias
        # target on this route, but that should be irrelevant.
        cfg = _config_with_models(
            route,
            models=ModelDefaults(
                fallbacks={
                    "minimax-m3:cloud": [
                        "minimax-m2.7:cloud",
                        "deepseek-v4-pro:cloud",
                    ],
                },
            ),
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match(
            {"model": "coder-pro", "path": "/v1/chat/completions"}
        )
        assert result is not None
        # The list is used verbatim — the alias map is not re-applied.
        assert result.fallbacks == [
            "minimax-m2.7:cloud",
            "deepseek-v4-pro:cloud",
        ]


# ────────────────────────────────────────────────────────────────────
# Backend selection strategies (single / weighted / round_robin)
# M4: implement weighted and round_robin strategies.
# ────────────────────────────────────────────────────────────────────


def _multi_backend_config(
    *routes: RouteConfig,
    backend_names: list[str] | None = None,
) -> MoaxyConfig:
    """Build a config with several named backends for the multi-backend tests.

    The default backend set is ``["b1", "b2", "b3"]``; the tests
    override this when they need a different number of backends.
    The single helper exists so the test code can be terse: it
    always builds a valid :class:`MoaxyConfig` that the routes can
    reference.
    """
    names = backend_names or ["b1", "b2", "b3"]
    backends = [
        AdapterConfig(
            name=name, adapter="ollama", base_url=f"http://127.0.0.1:9{idx}"
        )
        for idx, name in enumerate(names)
    ]
    return MoaxyConfig(backends=backends, routes=list(routes))


# ────────────────────────────────────────────────────────────────────
# VAL-M4 (single): single strategy uses route.backend
# ────────────────────────────────────────────────────────────────────


class TestSingleStrategy:
    """``strategy: single`` returns the route's ``backend`` field."""

    def test_single_strategy_returns_route_backend(self):
        route = _route(
            name="r",
            strategy="single",
            backend="olloma-local",
            backends=[],  # no multi-backend entries
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "olloma-local"

    def test_single_strategy_does_not_consult_backends_list(self):
        """Even if ``backends`` is populated, ``single`` ignores it.

        This codifies the contract that ``strategy: single`` is the
        historical default and is orthogonal to the multi-backend
        strategies: a route may have a populated ``backends`` list
        for documentation or future use, and ``single`` still picks
        the ``backend`` field.
        """
        route = _route(
            name="r",
            strategy="single",
            backend="primary-backend",
            backends=[
                BackendRef(name="extra-1", weight=1),
                BackendRef(name="extra-2", weight=1),
            ],
        )
        cfg = _multi_backend_config(
            route, backend_names=["primary-backend", "extra-1", "extra-2"]
        )
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "primary-backend"

    def test_single_strategy_is_default(self):
        """A route that does not declare a strategy defaults to ``single``."""
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
            backend="olloma-local",
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "olloma-local"

    def test_single_strategy_with_no_backend(self):
        """A ``single`` route with no ``backend`` set yields ``None``."""
        route = _route(name="r", strategy="single", backend=None)
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend is None

    def test_single_strategy_is_stable_across_calls(self):
        """Repeated matches return the same backend name."""
        route = _route(name="r", strategy="single", backend="olloma-local")
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        for _ in range(10):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            assert result.backend == "olloma-local"


# ────────────────────────────────────────────────────────────────────
# VAL-M4 (weighted): random selection proportional to weight
# ────────────────────────────────────────────────────────────────────


class TestWeightedStrategy:
    """``strategy: weighted`` returns a random backend from ``route.backends``."""

    def test_weighted_strategy_returns_one_of_route_backends(self):
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
                BackendRef(name="b3", weight=1),
            ],
        )
        cfg = _multi_backend_config(route)
        matcher = RouteMatcher(cfg, random_seed=42)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend in {"b1", "b2", "b3"}

    def test_weighted_strategy_with_seeded_random_is_deterministic(self):
        """A fixed seed makes weighted selection reproducible."""
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["b1", "b2"])
        matcher1 = RouteMatcher(cfg, random_seed=123)
        matcher2 = RouteMatcher(cfg, random_seed=123)
        # Run several calls; the sequences must match across the
        # two matchers because they share the same seed.
        seq1 = [
            matcher1.match({"model": "m", "path": "/v1/chat/completions"}).backend
            for _ in range(20)
        ]
        seq2 = [
            matcher2.match({"model": "m", "path": "/v1/chat/completions"}).backend
            for _ in range(20)
        ]
        assert seq1 == seq2

    def test_weighted_distribution_matches_weights_statistically(self):
        """Over many trials, the observed distribution approximates the weights.

        We use 3 backends with weights 1, 2, 3 (so the expected
        fractions are 1/6, 2/6, 3/6 = ~0.167, ~0.333, ~0.500). With
        6,000 trials, the standard error on each proportion is
        below 0.01; we assert the observed fractions are within
        0.02 of the expected values, leaving a wide margin to
        keep the test robust to normal RNG behaviour.
        """
        weights = [1, 2, 3]
        names = ["b1", "b2", "b3"]
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name=n, weight=w) for n, w in zip(names, weights, strict=True)
            ],
        )
        cfg = _multi_backend_config(route)
        matcher = RouteMatcher(cfg, random_seed=20240610)
        trials = 6000
        counts: dict[str, int] = {n: 0 for n in names}
        for _ in range(trials):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            counts[result.backend] += 1
        total_weight = sum(weights)
        for n, w in zip(names, weights, strict=True):
            expected = w / total_weight
            observed = counts[n] / trials
            assert abs(observed - expected) < 0.02, (
                f"weight for {n!r} (={w}) expected ~{expected:.3f}, "
                f"observed {observed:.3f} over {trials} trials"
            )

    def test_weighted_strategy_does_not_select_backends_with_zero_weight(self):
        """A backend with weight 0 should be effectively never selected.

        With weights [0, 1, 1] over 2000 trials, the zero-weight
        backend should be selected 0 times (the uniform draw falls
        into the non-zero cumulative range every time).
        """
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="zero", weight=0),
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["zero", "a", "b"])
        matcher = RouteMatcher(cfg, random_seed=7)
        for _ in range(2000):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            assert result.backend != "zero"

    def test_weighted_strategy_does_not_starve_low_weight(self):
        """A low-weight backend must still be selectable.

        With weights [1, 100] over 5000 trials, the low-weight
        backend must be selected at least once. The expected
        fraction is ~0.0099, so a tight lower bound like 0.005
        (i.e. 25/5000) catches starvation but tolerates normal
        statistical noise.
        """
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="rare", weight=1),
                BackendRef(name="common", weight=100),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["rare", "common"])
        matcher = RouteMatcher(cfg, random_seed=99)
        counts = {"rare": 0, "common": 0}
        for _ in range(5000):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            counts[result.backend] += 1
        assert counts["rare"] >= 25
        assert counts["common"] >= 4500

    def test_weighted_strategy_with_all_zero_weights_falls_back(self):
        """When every weight is 0, the selector returns the first entry."""
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=0),
                BackendRef(name="b2", weight=0),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["b1", "b2"])
        matcher = RouteMatcher(cfg, random_seed=1)
        # Degenerate weights: the selector falls back to the first entry.
        for _ in range(10):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            assert result.backend == "b1"

    def test_weighted_strategy_with_empty_backends_uses_route_backend(self):
        """Empty ``backends`` list falls back to the route's ``backend``."""
        route = _route(
            name="r",
            strategy="weighted",
            backend="olloma-local",
            backends=[],
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg, random_seed=1)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "olloma-local"

    def test_weighted_strategy_uses_seeded_random_across_matchers(self):
        """Two matchers with the same seed produce the same first pick."""
        route = _route(
            name="r",
            strategy="weighted",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
                BackendRef(name="b3", weight=1),
            ],
        )
        cfg = _multi_backend_config(route)
        a = RouteMatcher(cfg, random_seed=0)
        b = RouteMatcher(cfg, random_seed=0)
        a_first = a.match({"model": "m", "path": "/v1/chat/completions"}).backend
        b_first = b.match({"model": "m", "path": "/v1/chat/completions"}).backend
        assert a_first == b_first


# ────────────────────────────────────────────────────────────────────
# VAL-M4 (round_robin): cycle through route.backends in order
# ────────────────────────────────────────────────────────────────────


class TestRoundRobinStrategy:
    """``strategy: round_robin`` cycles through ``route.backends`` in order."""

    def test_round_robin_cycles_through_backends_in_order(self):
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
                BackendRef(name="b3", weight=1),
            ],
        )
        cfg = _multi_backend_config(route)
        matcher = RouteMatcher(cfg)
        sequence = [
            matcher.match({"model": "m", "path": "/v1/chat/completions"}).backend
            for _ in range(6)
        ]
        assert sequence == ["b1", "b2", "b3", "b1", "b2", "b3"]

    def test_round_robin_two_backends(self):
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["a", "b"])
        matcher = RouteMatcher(cfg)
        sequence = [
            matcher.match({"model": "m", "path": "/v1/chat/completions"}).backend
            for _ in range(4)
        ]
        assert sequence == ["a", "b", "a", "b"]

    def test_round_robin_ignores_weight_values(self):
        """The order in ``backends`` is what matters, not the weights."""
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=10),
                BackendRef(name="b2", weight=1),
                BackendRef(name="b3", weight=100),
            ],
        )
        cfg = _multi_backend_config(route)
        matcher = RouteMatcher(cfg)
        sequence = [
            matcher.match({"model": "m", "path": "/v1/chat/completions"}).backend
            for _ in range(6)
        ]
        assert sequence == ["b1", "b2", "b3", "b1", "b2", "b3"]

    def test_round_robin_single_backend_always_returns_same(self):
        """With one entry, the cycle never advances and is constant."""
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            backends=[BackendRef(name="solo", weight=1)],
        )
        cfg = _multi_backend_config(route, backend_names=["solo"])
        matcher = RouteMatcher(cfg)
        for _ in range(5):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            assert result.backend == "solo"

    def test_round_robin_with_empty_backends_uses_route_backend(self):
        """Empty ``backends`` falls back to ``route.backend``."""
        route = _route(
            name="r",
            strategy="round_robin",
            backend="olloma-local",
            backends=[],
        )
        cfg = _config(route)
        matcher = RouteMatcher(cfg)
        for _ in range(3):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            assert result.backend == "olloma-local"

    def test_round_robin_state_persists_across_routes(self):
        """A match against a different route must not reset this route's cycle."""
        rr = _route(
            name="rr",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
            ],
        )
        plain = _route(
            name="plain",
            model_pattern="other",
            strategy="single",
            backend="olloma-local",
        )
        cfg = _multi_backend_config(
            rr, plain, backend_names=["b1", "b2", "olloma-local"]
        )
        matcher = RouteMatcher(cfg)
        # First two calls hit the rr route and cycle.
        r1 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        r2 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert r1.backend == "b1"
        assert r2.backend == "b2"
        # The next rr call should continue from index 2 (mod 2 == 0 -> b1).
        r3 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert r3.backend == "b1"

    def test_round_robin_per_route_independent_cycles(self):
        """Two different round-robin routes have independent counters."""
        rr1 = _route(
            name="rr1",
            model_pattern="groupA-*",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        rr2 = _route(
            name="rr2",
            model_pattern="groupB-*",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="x", weight=1),
                BackendRef(name="y", weight=1),
            ],
        )
        cfg = _multi_backend_config(rr1, rr2, backend_names=["a", "b", "x", "y"])
        matcher = RouteMatcher(cfg)
        # First call against rr1 -> "a"; first call against rr2 -> "x".
        first_rr1 = matcher.match({"model": "groupA-1", "path": "/v1/chat/completions"})
        first_rr2 = matcher.match({"model": "groupB-1", "path": "/v1/chat/completions"})
        assert first_rr1.backend == "a"
        assert first_rr2.backend == "x"
        # Each route cycles independently.
        second_rr1 = matcher.match({"model": "groupA-2", "path": "/v1/chat/completions"})
        second_rr2 = matcher.match({"model": "groupB-2", "path": "/v1/chat/completions"})
        assert second_rr1.backend == "b"
        assert second_rr2.backend == "y"

    def test_round_robin_count_advances_per_call(self):
        """The internal counter is exposed indirectly: repeated cycles match."""
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
                BackendRef(name="c", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["a", "b", "c"])
        matcher = RouteMatcher(cfg)
        # 9 calls = 3 full cycles. The cycle index modulo 3 must
        # match the call number modulo 3.
        for call_index in range(9):
            result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
            assert result is not None
            expected_idx = call_index % 3
            expected = ["a", "b", "c"][expected_idx]
            assert result.backend == expected


# ────────────────────────────────────────────────────────────────────
# Selector lifecycle (add_route / remove_route interactions)
# ────────────────────────────────────────────────────────────────────


class TestSelectorLifecycle:
    """Selector state is created on first match and managed by add/remove."""

    def test_add_route_creates_fresh_selector(self):
        """Adding a new round-robin route starts its counter at 0."""
        cfg = _config()
        matcher = RouteMatcher(cfg)
        route = _route(
            name="rr",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        matcher.add_route(route)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        # First call of a fresh cycle -> index 0.
        assert result.backend == "a"

    def test_remove_route_drops_selector(self):
        """Removing a round-robin route clears its cycle state."""
        route = _route(
            name="rr",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["a", "b"])
        matcher = RouteMatcher(cfg)
        # Advance the cycle.
        matcher.match({"model": "m", "path": "/v1/chat/completions"})
        matcher.match({"model": "m", "path": "/v1/chat/completions"})
        # Remove and re-add: the new selector should start at "a".
        assert matcher.remove_route("rr") is True
        matcher.add_route(route)
        result = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert result is not None
        assert result.backend == "a"

    def test_add_route_does_not_disturb_existing_selectors(self):
        """Adding a route after matches does not reset earlier selectors."""
        first = _route(
            name="first",
            strategy="round_robin",
            backend=None,
            backends=[
                BackendRef(name="a", weight=1),
                BackendRef(name="b", weight=1),
            ],
        )
        cfg = _multi_backend_config(first, backend_names=["a", "b"])
        matcher = RouteMatcher(cfg)
        # Cycle "first" once.
        r1 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert r1.backend == "a"
        # Add a new route.
        second = _route(
            name="second",
            model_pattern="other",
            strategy="round_robin",
            backend=None,
            backends=[BackendRef(name="c", weight=1)],
        )
        matcher.add_route(second)
        # The first route's cycle should advance normally.
        r2 = matcher.match({"model": "m", "path": "/v1/chat/completions"})
        assert r2.backend == "b"

    def test_add_route_rejects_duplicate_name(self):
        cfg = _config(_route(name="dup"))
        matcher = RouteMatcher(cfg)
        with pytest.raises(ValueError, match="already exists"):
            matcher.add_route(_route(name="dup"))


# ────────────────────────────────────────────────────────────────────
# Strategy interacts correctly with alias resolution
# ────────────────────────────────────────────────────────────────────


class TestStrategyWithAliases:
    """Backend selection is independent of alias resolution."""

    def test_alias_resolution_unaffected_by_strategy(self):
        route = _route(
            name="r",
            strategy="round_robin",
            backend=None,
            aliases={"coder-pro": "minimax-m3:cloud"},
            backends=[
                BackendRef(name="b1", weight=1),
                BackendRef(name="b2", weight=1),
            ],
        )
        cfg = _multi_backend_config(route, backend_names=["b1", "b2"])
        matcher = RouteMatcher(cfg)
        r1 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        r2 = matcher.match({"model": "coder-pro", "path": "/v1/chat/completions"})
        assert r1 is not None
        assert r2 is not None
        # Alias resolution is independent of the backend selector.
        assert r1.original_model == "coder-pro"
        assert r1.resolved_model == "minimax-m3:cloud"
        assert r2.original_model == "coder-pro"
        assert r2.resolved_model == "minimax-m3:cloud"
        # The round-robin cycle is unaffected.
        assert r1.backend == "b1"
        assert r2.backend == "b2"


# ────────────────────────────────────────────────────────────────────
# Re-exports from the routing package
# ────────────────────────────────────────────────────────────────────


class TestPackageReexports:
    def test_routing_package_exposes_classes(self):
        from moaxy.routing import RouteMatch as RT  # noqa: F401
        from moaxy.routing import RouteMatcher as RM  # noqa: F401

        assert RT is RouteMatch
        assert RM is RouteMatcher
