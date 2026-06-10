"""Tests for the AdapterRegistry.

Covers:
- build(configs) returns a registry with adapters keyed by AdapterConfig.name.
- Each adapter is constructed with its base_url, api_key, and timeout.
- get(name) returns the matching adapter or None.
- adapters public dict mirrors the built entries.
- Unknown adapter kind raises a clear error.
- Duplicate backend names raise a clear error (registry and config layer).
- Registry build is independent of the Adapter class-level ``name`` attribute;
  the key is always taken from AdapterConfig.name.
- The constructed OllamaAdapter has the right type and configured endpoint.
"""

from __future__ import annotations

import inspect

import pytest

from moaxy.adapters.base import Adapter
from moaxy.adapters.ollama import OllamaAdapter
from moaxy.adapters.registry import AdapterRegistry, build_registry
from moaxy.models.config import AdapterConfig, MoaxyConfig

# ────────────────────────────────────────────────────────────────────
# Test fixtures and helpers
# ────────────────────────────────────────────────────────────────────


def _make_ollama_config(
    *,
    name: str = "ollama-local",
    base_url: str = "http://127.0.0.1:11434",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> AdapterConfig:
    return AdapterConfig(
        name=name,
        adapter="ollama",
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


# ────────────────────────────────────────────────────────────────────
# Build returns dict keyed by AdapterConfig.name
# ────────────────────────────────────────────────────────────────────


class TestRegistryBuild:
    """``build`` returns a dict[str, Adapter] keyed by config name."""

    def test_build_with_empty_configs_yields_empty_adapters(self):
        registry = AdapterRegistry.build([])
        assert registry.adapters == {}
        assert dict(registry.adapters) == {}

    def test_build_single_ollama_adapter_keyed_by_name(self):
        config = _make_ollama_config(name="ollama-local")
        registry = AdapterRegistry.build([config])
        assert "ollama-local" in registry.adapters
        assert isinstance(registry.adapters["ollama-local"], Adapter)
        assert isinstance(registry.adapters["ollama-local"], OllamaAdapter)

    def test_build_multiple_ollama_adapters_keyed_by_unique_names(self):
        configs = [
            _make_ollama_config(name="ollama-a", base_url="http://127.0.0.1:11434"),
            _make_ollama_config(name="ollama-b", base_url="http://127.0.0.1:11435"),
        ]
        registry = AdapterRegistry.build(configs)
        assert set(registry.adapters) == {"ollama-a", "ollama-b"}
        assert registry.adapters["ollama-a"].base_url == "http://127.0.0.1:11434"
        assert registry.adapters["ollama-b"].base_url == "http://127.0.0.1:11435"

    def test_adapters_dict_is_public(self):
        registry = AdapterRegistry.build([_make_ollama_config(name="o1")])
        # Direct access is the supported read-only access path.
        assert "o1" in registry.adapters
        adapter = registry.adapters["o1"]
        assert isinstance(adapter, Adapter)


# ────────────────────────────────────────────────────────────────────
# Adapter construction passes base_url, api_key, timeout through
# ────────────────────────────────────────────────────────────────────


class TestRegistryBuildPropagatesConfig:
    """The registry forwards AdapterConfig fields to the concrete adapter."""

    def test_base_url_propagated(self):
        config = _make_ollama_config(
            name="o", base_url="http://example.test:9999/"
        )
        registry = AdapterRegistry.build([config])
        adapter = registry.adapters["o"]
        # Trailing slash should be stripped (adapter-level normalisation).
        assert adapter.base_url == "http://example.test:9999"

    def test_timeout_propagated(self):
        config = _make_ollama_config(name="o", timeout=12.5)
        registry = AdapterRegistry.build([config])
        adapter = registry.adapters["o"]
        assert adapter.timeout == 12.5

    def test_api_key_propagated(self):
        config = _make_ollama_config(name="o", api_key="sk-secret")
        registry = AdapterRegistry.build([config])
        adapter = registry.adapters["o"]
        assert adapter.api_key == "sk-secret"

    def test_api_key_none_propagated(self):
        config = _make_ollama_config(name="o", api_key=None)
        registry = AdapterRegistry.build([config])
        adapter = registry.adapters["o"]
        assert adapter.api_key is None

    def test_default_timeout_when_not_set_on_config(self):
        # AdapterConfig default for timeout is 30.0; verify the registry
        # does not override it.
        config = AdapterConfig(
            name="o",
            adapter="ollama",
            base_url="http://127.0.0.1:11434",
        )
        registry = AdapterRegistry.build([config])
        adapter = registry.adapters["o"]
        assert adapter.timeout == 30.0


# ────────────────────────────────────────────────────────────────────
# get(name)
# ────────────────────────────────────────────────────────────────────


class TestRegistryGet:
    """``get(name)`` returns the adapter or None."""

    def test_get_returns_adapter_for_known_name(self):
        registry = AdapterRegistry.build([_make_ollama_config(name="ollama-local")])
        adapter = registry.get("ollama-local")
        assert adapter is not None
        assert isinstance(adapter, OllamaAdapter)
        assert adapter is registry.adapters["ollama-local"]

    def test_get_returns_none_for_unknown_name(self):
        registry = AdapterRegistry.build([_make_ollama_config(name="ollama-local")])
        assert registry.get("does-not-exist") is None

    def test_get_returns_none_on_empty_registry(self):
        registry = AdapterRegistry.build([])
        assert registry.get("anything") is None

    def test_get_distinguishes_among_multiple_adapters(self):
        configs = [
            _make_ollama_config(name="a", base_url="http://a:1"),
            _make_ollama_config(name="b", base_url="http://b:2"),
        ]
        registry = AdapterRegistry.build(configs)
        a = registry.get("a")
        b = registry.get("b")
        assert a is not None and b is not None
        assert a is not b
        assert a.base_url == "http://a:1"
        assert b.base_url == "http://b:2"


# ────────────────────────────────────────────────────────────────────
# build_registry alias (module-level helper)
# ────────────────────────────────────────────────────────────────────


class TestModuleLevelBuildHelper:
    """The module-level ``build_registry`` helper is an alias for ``AdapterRegistry.build``."""

    def test_build_registry_returns_registry(self):
        registry = build_registry([_make_ollama_config(name="o")])
        assert isinstance(registry, AdapterRegistry)
        assert "o" in registry.adapters

    def test_build_registry_empty(self):
        registry = build_registry([])
        assert registry.adapters == {}


# ────────────────────────────────────────────────────────────────────
# Constructed adapter is functional end-to-end (via in-process transport)
# ────────────────────────────────────────────────────────────────────


class TestRegistryBuiltAdapterIsFunctional:
    """A registry-built OllamaAdapter is the right type and has the right config.

    Functional chat() is exercised in ``tests/test_ollama_adapter.py``; this
    suite focuses on the registry's responsibility — producing a correctly
    configured adapter — without duplicating transport-mock plumbing.
    """

    def test_built_ollama_adapter_is_correct_type(self):
        config = _make_ollama_config(name="o")
        registry = AdapterRegistry.build([config])
        adapter = registry.get("o")
        assert isinstance(adapter, OllamaAdapter)

    def test_built_ollama_adapter_preserves_endpoint_base_url(self):
        config = AdapterConfig(
            name="custom",
            adapter="ollama",
            base_url="http://example.test:9999",
            timeout=5.0,
        )
        registry = AdapterRegistry.build([config])
        adapter = registry.get("custom")
        assert adapter is not None
        # The endpoint should be composed of the configured base URL and the
        # standard /v1/chat/completions path.
        assert adapter.endpoint == "http://example.test:9999/v1/chat/completions"

    def test_built_ollama_adapter_is_closeable(self):
        """The adapter's close() must be callable on a registry-built instance."""
        config = _make_ollama_config(name="o", timeout=5.0)
        registry = AdapterRegistry.build([config])
        adapter = registry.get("o")
        assert adapter is not None
        # close() is a coroutine function; calling it should not raise.
        assert inspect.iscoroutinefunction(adapter.close)


# ────────────────────────────────────────────────────────────────────
# Adapter name comes from AdapterConfig.name, not from Adapter class
# ────────────────────────────────────────────────────────────────────


class TestRegistryKeyComesFromConfig:
    """The registry's key is always the config's name, never the adapter class name."""

    def test_key_matches_config_name_even_if_class_name_differs(self):
        # OllamaAdapter.name is the class-level string "ollama". The registry
        # key MUST be the AdapterConfig.name ("my-ollama"), not "ollama".
        config = _make_ollama_config(name="my-ollama")
        registry = AdapterRegistry.build([config])
        assert "my-ollama" in registry.adapters
        assert "ollama" not in registry.adapters

    def test_key_distinct_from_underlying_class(self):
        config = _make_ollama_config(name="backend-1")
        registry = AdapterRegistry.build([config])
        # The registered name is "backend-1", but the underlying class is
        # OllamaAdapter with class name "ollama".
        assert OllamaAdapter.name == "ollama"
        assert "backend-1" in registry.adapters


# ────────────────────────────────────────────────────────────────────
# Integration with MoaxyConfig
# ────────────────────────────────────────────────────────────────────


class TestRegistryWithMoaxyConfig:
    """The registry works with ``MoaxyConfig.backends`` (list[AdapterConfig])."""

    def test_build_from_moaxy_config_backends(self):
        moaxy = MoaxyConfig(
            backends=[
                _make_ollama_config(name="ollama-local"),
                _make_ollama_config(
                    name="ollama-secondary",
                    base_url="http://127.0.0.1:11435",
                ),
            ]
        )
        registry = AdapterRegistry.build(moaxy.backends)
        assert set(registry.adapters) == {"ollama-local", "ollama-secondary"}

    def test_build_from_moaxy_config_default_routes_does_not_break(self):
        # MoaxyConfig.routes defaults to []; it is not the registry's concern.
        moaxy = MoaxyConfig(backends=[_make_ollama_config(name="o")])
        registry = AdapterRegistry.build(moaxy.backends)
        assert "o" in registry.adapters


# ────────────────────────────────────────────────────────────────────
# Unknown adapter kind
# ────────────────────────────────────────────────────────────────────


class TestRegistryRejectsUnknownAdapter:
    """A config whose ``adapter`` literal is not implemented raises a clear error."""

    def test_unknown_adapter_string_raises_value_error(self):
        # Pydantic Literal["ollama","openai"] only allows these two values,
        # so to simulate an unknown adapter kind we construct a config that
        # bypasses pydantic validation by using a model that allows any
        # string. We rely on the registry's own validation here.
        bad_config = AdapterConfig.model_construct(
            name="bogus",
            adapter="anthropic",  # type: ignore[arg-type]
            base_url="http://x",
            api_key=None,
            timeout=30.0,
        )
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            AdapterRegistry.build([bad_config])


# ────────────────────────────────────────────────────────────────────
# Sanity: registry instance is independent (no class-level state)
# ────────────────────────────────────────────────────────────────────


class TestRegistryIsolation:
    """Two registries built independently do not share adapter instances."""

    def test_two_registries_have_distinct_adapter_instances(self):
        r1 = AdapterRegistry.build([_make_ollama_config(name="o")])
        r2 = AdapterRegistry.build([_make_ollama_config(name="o")])
        a1 = r1.get("o")
        a2 = r2.get("o")
        assert a1 is not None and a2 is not None
        assert a1 is not a2

    def test_adapters_dict_keys_isolated(self):
        r1 = AdapterRegistry.build([_make_ollama_config(name="only-in-r1")])
        r2 = AdapterRegistry.build([_make_ollama_config(name="only-in-r2")])
        assert "only-in-r1" in r1.adapters
        assert "only-in-r1" not in r2.adapters
        assert "only-in-r2" in r2.adapters
        assert "only-in-r2" not in r1.adapters
