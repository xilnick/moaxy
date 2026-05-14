"""Tests for the moaxy plugin system.

Covers: plugin loading, initialization, validation, process invocation,
lifecycle management (init/validate/cleanup), and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from moaxy.plugins.base import Plugin
from moaxy.plugins.discovery import (
    PluginDiscoveryError,
    discover_plugins,
    load_plugin_instance,
)
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"


class _MinimalPlugin(Plugin):
    """Minimal concrete plugin for testing the base class lifecycle."""

    name = "minimal"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def process(self, context: dict) -> dict:
        return context


class _FailingValidatePlugin(_MinimalPlugin):
    """Plugin that always fails validation."""

    name = "failing"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def validate(self) -> list[str]:
        errors = super().validate()
        errors.append("intentional validation failure")
        return errors


class _ResourcePlugin(_MinimalPlugin):
    """Plugin that tracks resource lifecycle."""

    name = "resource_tracker"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def __init__(self) -> None:
        super().__init__()
        self.resource = None
        self.cleaned = False

    def init(self) -> None:
        super().init()
        self.resource = "acquired"

    def cleanup(self) -> None:
        self.cleaned = True
        super().cleanup()

    def process(self, context: dict) -> dict:
        return context


def _make_plugin_dir(tmp_path: Path, plugin_name: str, plugin_code: str) -> Path:
    """Create a temporary plugin package directory with the given code."""
    pkg = tmp_path / plugin_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "plugin.py").write_text(plugin_code)
    return tmp_path


class TestPluginBase:
    """Tests for the Plugin base class lifecycle."""

    def test_init_sets_initialized_flag(self):
        p = _MinimalPlugin()
        assert not p.initialized
        p.init()
        assert p.initialized

    def test_validate_returns_no_errors_for_valid_plugin(self):
        p = _MinimalPlugin()
        p.init()
        errors = p.validate()
        assert errors == []

    def test_validate_returns_errors(self):
        p = _FailingValidatePlugin()
        p.init()
        errors = p.validate()
        assert "intentional validation failure" in errors

    def test_cleanup_resets_flags(self):
        p = _MinimalPlugin()
        p.init()
        p.validate()
        assert p.initialized
        assert p.validated
        p.cleanup()
        assert not p.initialized
        assert not p.validated

    def test_full_lifecycle_init_validate_cleanup(self):
        p = _MinimalPlugin()
        p.init()
        assert p.initialized
        assert not p.validated

        errors = p.validate()
        assert errors == []
        assert p.validated

        p.cleanup()
        assert not p.initialized
        assert not p.validated

    def test_resource_plugin_acquires_and_releases(self):
        p = _ResourcePlugin()
        p.init()
        assert p.resource == "acquired"
        assert not p.cleaned
        p.cleanup()
        assert p.cleaned


class TestPluginDiscovery:
    """Tests for plugin discovery from a directory."""

    def test_discovers_valid_plugin(self, tmp_path):
        code = """
from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType

class TestR(Plugin):
    name = "test_router"
    version = "1.0"
    plugin_type = PluginType.ROUTER
    def process(self, context):
        return context
"""
        _make_plugin_dir(tmp_path, "test_router", code)
        plugins = discover_plugins(tmp_path)
        assert len(plugins) == 1
        assert plugins[0].name == "test_router"

    def test_discovers_multiple_plugins(self, tmp_path):
        for i, ptype in enumerate(["router", "transformer", "auth"], 1):
            code = f"""
from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType

class P{i}(Plugin):
    name = "p{i}"
    version = "1.0"
    plugin_type = PluginType.{ptype.upper()}
    def process(self, context):
        return context
"""
            _make_plugin_dir(tmp_path, f"p{i}", code)

        plugins = discover_plugins(tmp_path)
        assert len(plugins) == 3

    def test_skips_directories_without_plugin_py(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        plugins = discover_plugins(tmp_path)
        assert plugins == []

    def test_skips_hidden_directories(self, tmp_path):
        code = """
from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType

class HiddenPlugin(Plugin):
    name = "hidden"
    version = "1.0"
    plugin_type = PluginType.ROUTER
    def process(self, context):
        return context
"""
        _make_plugin_dir(tmp_path, "_hidden_plugin", code)
        plugins = discover_plugins(tmp_path)
        assert plugins == []

    def test_raises_for_missing_directory(self):
        with pytest.raises(PluginDiscoveryError, match="not found"):
            discover_plugins("/nonexistent/path/12345")

    def test_load_plugin_instance_with_config(self, tmp_path):
        code = """
from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType

class ConfigPlugin(Plugin):
    name = "config_plugin"
    version = "1.0"
    plugin_type = PluginType.ROUTER
    custom_field = "default"
    def process(self, context):
        return context
"""
        _make_plugin_dir(tmp_path, "config_plugin", code)
        sys.path.insert(0, str(tmp_path))
        try:
            plugins = discover_plugins(tmp_path)
            instance = load_plugin_instance(plugins[0], {"custom_field": "overridden"})
            assert instance.custom_field == "overridden"
        finally:
            sys.path.remove(str(tmp_path))


class TestPluginManager:
    """Tests for PluginManager lifecycle orchestration."""

    def test_loads_plugins_from_directory(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert errors == []
        assert mgr.plugin_count == 2
        assert mgr.loaded

    def test_list_plugins_returns_metadata(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        info = mgr.list_plugins()
        assert len(info) == 2
        names = {p["name"] for p in info}
        assert "demo_router" in names
        assert "demo_transformer" in names

    def test_get_plugin_by_name(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        assert router is not None
        assert router.name == "demo_router"
        assert router.plugin_type == PluginType.ROUTER

    def test_get_plugin_returns_none_for_unknown(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        assert mgr.get_plugin("nonexistent") is None

    def test_get_plugins_by_type(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        routers = mgr.get_plugins_by_type(PluginType.ROUTER)
        assert len(routers) == 1
        assert routers[0].name == "demo_router"

        transformers = mgr.get_plugins_by_type(PluginType.TRANSFORMER)
        assert len(transformers) == 1
        assert transformers[0].name == "demo_transformer"

    def test_shutdown_cleans_up_all(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        assert mgr.plugin_count == 2
        mgr.shutdown()
        assert mgr.plugin_count == 0
        assert not mgr.loaded

    def test_double_load_is_noop(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        mgr.load()
        assert mgr.plugin_count == 2

    def test_load_with_empty_dir(self, tmp_path):
        mgr = PluginManager(plugins_dir=tmp_path)
        errors = mgr.load()
        assert errors == []
        assert mgr.plugin_count == 0


class TestDemoRouterPlugin:
    """Integration tests for the demo router plugin."""

    def test_routes_non_matching_model_to_default(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        context = {"model": "llama3"}
        result = router.process(context)
        decision = result["route_decision"]
        assert decision.target == "http://localhost:11434"
        assert not decision.intercepted

    def test_intercepts_model_matching_prefix(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        context = {"model": "demo-gpt-4"}
        result = router.process(context)
        decision = result["route_decision"]
        assert decision.intercepted is True
        assert decision.target == "http://localhost:8080"
        assert decision.model == "gpt-4"
        assert decision.intercepted_by == "demo_router"

    def test_tracks_route_count(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        assert router.route_count == 0
        router.process({"model": "llama3"})
        router.process({"model": "demo-qwen"})
        assert router.route_count == 2

    def test_cleanup_resets_route_count(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        router.process({"model": "llama3"})
        router.cleanup()
        assert router.route_count == 0
        assert not router.initialized


class TestDemoTransformerPlugin:
    """Integration tests for the demo transformer plugin."""

    def test_injects_header_into_request(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        context = {"request": {"path": "/v1/chat/completions"}}
        result = transformer.process(context)
        headers = result["request"]["headers"]
        assert headers["X-Moaxy-Demo"] == "true"

    def test_modifies_response_body(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        context = {"response": {"choices": []}}
        result = transformer.process(context)
        assert result["response"]["proxy"] == "demo_transformer"
        assert result["response"]["transform_count"] == 1

    def test_sets_transformed_by_on_context(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        result = transformer.process({"request": {}, "response": {}})
        assert result["transformed_by"] == "demo_transformer"

    def test_tracks_transform_count(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        transformer.process({"response": {}})
        transformer.process({"response": {}})
        transformer.process({"response": {}})
        assert transformer.transform_count == 3

    def test_cleanup_resets_transform_count(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        transformer.process({"response": {}})
        transformer.cleanup()
        assert transformer.transform_count == 0
        assert not transformer.initialized


class TestManagerFullLifecycle:
    """End-to-end tests of the full load → process → shutdown cycle."""

    def test_orchestrates_both_plugins_in_sequence(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()

        router = mgr.get_plugin("demo_router")
        result = router.process({"model": "demo-codegen"})
        assert result["route_decision"].intercepted is True
        assert result["route_decision"].target == "http://localhost:8080"

        transformer = mgr.get_plugin("demo_transformer")
        result = transformer.process({"request": {}, "response": {}})
        assert "X-Moaxy-Demo" in result["request"]["headers"]
        assert result["response"]["proxy"] == "demo_transformer"

        mgr.shutdown()
        assert not router.initialized
        assert not transformer.initialized
