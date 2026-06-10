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
    MoaxyConfig,
    ReflectionConfig,
    RouteConfig,
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
        backend=backend,
        aliases=aliases or {},
        fallbacks=fallbacks or [],
        retry=retry,
        reflection=reflection or ReflectionConfig(),
        advisor=advisor or AdvisorConfig(),
    )


def _config(*routes: RouteConfig) -> MoaxyConfig:
    """Wrap a list of routes in a :class:`MoaxyConfig` with a single backend."""
    return MoaxyConfig(backends=[_backend()], routes=list(routes))


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
# Re-exports from the routing package
# ────────────────────────────────────────────────────────────────────


class TestPackageReexports:
    def test_routing_package_exposes_classes(self):
        from moaxy.routing import RouteMatch as RT  # noqa: F401
        from moaxy.routing import RouteMatcher as RM  # noqa: F401

        assert RT is RouteMatch
        assert RM is RouteMatcher
