"""Tests for the Plugin base class async extension (process_async hook).

Covers: default no-op process_async, subclass overrides, validation of
plugin_type/name, and backwards compatibility with existing sync plugins.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from moaxy.plugins.base import Plugin
from moaxy.plugins.discovery import discover_plugins
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"


class _SyncOnlyPlugin(Plugin):
    """Concrete plugin that implements only the sync `process` hook."""

    name = "sync_only"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def process(self, context: dict) -> dict:
        return context


class _AsyncOverridePlugin(Plugin):
    """Concrete plugin that overrides process_async."""

    name = "async_override"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def __init__(self) -> None:
        super().__init__()
        self.async_call_count = 0
        self.last_async_context: dict | None = None

    def process(self, context: dict) -> dict:
        return context

    async def process_async(self, context: dict) -> dict:
        self.async_call_count += 1
        self.last_async_context = context
        context["processed_async_by"] = self.name
        return context


class _AsyncOnlyPlugin(Plugin):
    """Concrete plugin that does not override process at all.

    Used to verify that the abstract process() does NOT prevent subclassing
    when only the default no-op async path is exercised. This plugin will
    still fail to instantiate as an abstract class because process is
    abstract, so we override it with a no-op.
    """

    name = "async_only"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def process(self, context: dict) -> dict:
        return context

    async def process_async(self, context: dict) -> dict:
        await asyncio.sleep(0)
        return context


class _EmptyNamePlugin(Plugin):
    name = ""
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def process(self, context: dict) -> dict:
        return context


class _InvalidTypePlugin(Plugin):
    name = "bad_type"
    version = "0.1.0"
    plugin_type = "router"  # type: ignore[assignment]

    def process(self, context: dict) -> dict:
        return context

    def init(self) -> None:
        # Skip base init() logging because the invalid plugin_type would
        # crash the .value accessor. We still mark initialized for
        # validate() to mirror normal flow.
        self._initialized = True


class TestProcessAsyncDefault:
    """Tests for the default no-op process_async implementation."""

    def test_process_async_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(Plugin.process_async)

    @pytest.mark.asyncio
    async def test_default_process_async_returns_context_unchanged(self):
        p = _SyncOnlyPlugin()
        ctx = {"model": "minimax-m3:cloud", "messages": []}
        result = await p.process_async(ctx)
        assert result is ctx
        assert result == {"model": "minimax-m3:cloud", "messages": []}

    @pytest.mark.asyncio
    async def test_default_process_async_does_not_mutate_context(self):
        p = _SyncOnlyPlugin()
        ctx: dict = {"a": 1}
        await p.process_async(ctx)
        assert ctx == {"a": 1}

    @pytest.mark.asyncio
    async def test_default_process_async_with_empty_context(self):
        p = _SyncOnlyPlugin()
        ctx: dict = {}
        result = await p.process_async(ctx)
        assert result is ctx
        assert result == {}


class TestProcessAsyncOverride:
    """Tests for subclasses that override process_async."""

    @pytest.mark.asyncio
    async def test_overridden_process_async_is_invoked(self):
        p = _AsyncOverridePlugin()
        ctx = {"x": 1}
        result = await p.process_async(ctx)
        assert p.async_call_count == 1
        assert p.last_async_context is ctx
        assert result["processed_async_by"] == "async_override"

    @pytest.mark.asyncio
    async def test_overridden_process_async_can_be_awaited_multiple_times(self):
        p = _AsyncOverridePlugin()
        await p.process_async({"i": 0})
        await p.process_async({"i": 1})
        await p.process_async({"i": 2})
        assert p.async_call_count == 3

    @pytest.mark.asyncio
    async def test_overridden_process_async_can_await_other_coroutines(self):
        p = _AsyncOnlyPlugin()
        ctx = {"k": "v"}
        result = await p.process_async(ctx)
        assert result is ctx


class TestSyncProcessUnchanged:
    """Backwards-compatibility: sync process(context) -> dict still works."""

    def test_sync_plugin_still_callable(self):
        p = _SyncOnlyPlugin()
        ctx = {"model": "demo-gpt-4"}
        result = p.process(ctx)
        assert result is ctx

    def test_existing_demo_router_still_works(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        assert router is not None
        out = router.process({"model": "demo-x"})
        assert out["route_decision"].intercepted is True

    def test_existing_demo_transformer_still_works(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        assert transformer is not None
        out = transformer.process({"request": {"path": "/v1/chat/completions"}})
        assert out["request"]["headers"]["X-Moaxy-Demo"] == "true"


class TestValidationInvariants:
    """validate() must continue to enforce plugin_type and name checks."""

    def test_validate_passes_for_valid_plugin(self):
        p = _SyncOnlyPlugin()
        p.init()
        assert p.validate() == []
        assert p.validated is True

    def test_validate_flags_empty_name(self):
        p = _EmptyNamePlugin()
        p.init()
        errors = p.validate()
        assert "Plugin name must be set" in errors
        assert p.validated is False

    def test_validate_flags_non_plugin_type_plugin_type(self):
        p = _InvalidTypePlugin()
        p.init()
        errors = p.validate()
        assert "Plugin must declare a valid plugin_type" in errors
        assert p.validated is False

    def test_validate_flags_both_empty_name_and_bad_type(self):
        class _BothBad(Plugin):
            name = ""
            version = "0"
            plugin_type = "transformer"  # type: ignore[assignment]

            def process(self, context):
                return context

            def init(self) -> None:
                # Avoid the base init() logger that dereferences
                # plugin_type.value (would crash for a string).
                self._initialized = True

        p = _BothBad()
        p.init()
        errors = p.validate()
        assert "Plugin name must be set" in errors
        assert "Plugin must declare a valid plugin_type" in errors


class TestAsyncExtensionDoesNotBreakDiscovery:
    """Adding process_async must not break Plugin discovery or manager load."""

    def test_demo_plugins_still_discoverable(self):
        plugins = discover_plugins(PLUGINS_DIR)
        names = {p.name for p in plugins}
        assert "demo_router" in names
        assert "demo_transformer" in names

    def test_demo_plugins_still_load_via_manager(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert errors == []
        assert mgr.plugin_count >= 2

    def test_custom_plugin_with_async_hook_loads_via_manager(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert errors == []

        instance = _AsyncOverridePlugin()
        instance.init()
        assert instance.validate() == []
        mgr._plugins[instance.name] = instance

        assert mgr.get_plugin("async_override") is not None
        assert mgr.get_plugin("async_override").plugin_type == PluginType.MIDDLEWARE


class TestProcessAsyncWithExistingPlugins:
    """Existing built-in plugins have a no-op process_async by default."""

    @pytest.mark.asyncio
    async def test_demo_router_default_process_async_is_noop(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        assert router is not None
        ctx = {"model": "x"}
        result = await router.process_async(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_demo_transformer_default_process_async_is_noop(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        assert transformer is not None
        ctx = {"request": {}, "response": {}}
        result = await transformer.process_async(ctx)
        assert result is ctx
        # No side effects from default no-op
        assert "X-Moaxy-Demo" not in ctx.get("request", {}).get("headers", {})
