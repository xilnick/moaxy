"""Tests for the demo_reflector and demo_advisor built-in plugins.

Covers:
- Both plugin files exist on disk and define a concrete Plugin subclass.
- PluginManager.load() discovers 4 plugins total (router, transformer,
  reflector, advisor).
- Each plugin's process_async is a no-op pass-through that increments an
  internal counter.
- The full lifecycle (init() → validate() → process_async() → cleanup()) works
  for both plugins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moaxy.plugins.base import Plugin
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"


# -----------------------------------------------------------------------------
# Disk presence and importability
# -----------------------------------------------------------------------------


class TestPluginFilesExist:
    """The two new built-in plugin files must be on disk and importable."""

    def test_demo_reflector_plugin_file_exists(self):
        path = PLUGINS_DIR / "demo_reflector" / "plugin.py"
        assert path.is_file(), f"missing {path}"

    def test_demo_advisor_plugin_file_exists(self):
        path = PLUGINS_DIR / "demo_advisor" / "plugin.py"
        assert path.is_file(), f"missing {path}"

    def test_demo_reflector_package_init_exists(self):
        path = PLUGINS_DIR / "demo_reflector" / "__init__.py"
        assert path.is_file(), f"missing {path}"

    def test_demo_advisor_package_init_exists(self):
        path = PLUGINS_DIR / "demo_advisor" / "__init__.py"
        assert path.is_file(), f"missing {path}"

    def test_demo_reflector_plugin_class_importable(self):
        from plugins.demo_reflector.plugin import DemoReflectorPlugin

        assert issubclass(DemoReflectorPlugin, Plugin)
        assert not __import__("inspect").isabstract(DemoReflectorPlugin)

    def test_demo_advisor_plugin_class_importable(self):
        from plugins.demo_advisor.plugin import DemoAdvisorPlugin

        assert issubclass(DemoAdvisorPlugin, Plugin)
        assert not __import__("inspect").isabstract(DemoAdvisorPlugin)


# -----------------------------------------------------------------------------
# PluginManager discovery of all 4 built-in plugins
# -----------------------------------------------------------------------------


class TestFourBuiltInPluginsDiscovered:
    """After loading, the manager must have exactly 4 plugins."""

    def test_loads_four_plugins(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert errors == []
        assert mgr.plugin_count == 4

    def test_plugin_names_include_all_four_demos(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        names = {p["name"] for p in mgr.list_plugins()}
        assert names == {
            "demo_router",
            "demo_transformer",
            "demo_reflector",
            "demo_advisor",
        }

    def test_plugin_types_cover_router_transformer_reflector_advisor(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        types_seen = {p["type"] for p in mgr.list_plugins()}
        assert types_seen == {"router", "transformer", "reflector", "advisor"}

    def test_get_plugin_demo_reflector(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        assert plugin.plugin_type == PluginType.REFLECTOR

    def test_get_plugin_demo_advisor(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        assert plugin.plugin_type == PluginType.ADVISOR

    def test_get_plugins_by_type_reflector(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        reflectors = mgr.get_plugins_by_type(PluginType.REFLECTOR)
        assert len(reflectors) == 1
        assert reflectors[0].name == "demo_reflector"

    def test_get_plugins_by_type_advisor(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        advisors = mgr.get_plugins_by_type(PluginType.ADVISOR)
        assert len(advisors) == 1
        assert advisors[0].name == "demo_advisor"

    def test_list_plugins_metadata_after_load(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        info = mgr.list_plugins()
        assert len(info) == 4
        for entry in info:
            assert entry["initialized"] is True
            assert entry["validated"] is True
            assert entry["version"] == "1.0.0"


# -----------------------------------------------------------------------------
# DemoReflectorPlugin behaviour
# -----------------------------------------------------------------------------


class TestDemoReflectorPlugin:
    """Behavioural tests for the demo_reflector built-in."""

    def test_default_plugin_type_is_reflector(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        assert plugin.plugin_type is PluginType.REFLECTOR

    def test_lifecycle_init_validate(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        assert plugin.initialized is True
        assert plugin.validated is True

    def test_process_async_is_coroutine_function(self):
        import inspect

        from plugins.demo_reflector.plugin import DemoReflectorPlugin

        assert inspect.iscoroutinefunction(DemoReflectorPlugin.process_async)

    @pytest.mark.asyncio
    async def test_process_async_returns_context_unchanged(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        ctx = {"request": {"model": "minimax-m3:cloud"}, "step": "critique"}
        result = await plugin.process_async(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_process_async_is_no_op_pass_through(self):
        """The pass-through contract: every key in the input survives."""
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        ctx = {
            "request": {"model": "minimax-m3:cloud", "messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"content": "hello"}}]},
            "turn": 0,
        }
        result = await plugin.process_async(ctx)
        assert result["request"] == ctx["request"]
        assert result["response"] == ctx["response"]
        assert result["turn"] == 0

    @pytest.mark.asyncio
    async def test_process_async_increments_internal_counter(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        assert plugin.reflect_count == 0
        await plugin.process_async({})
        assert plugin.reflect_count == 1
        await plugin.process_async({})
        assert plugin.reflect_count == 2
        await plugin.process_async({})
        assert plugin.reflect_count == 3

    @pytest.mark.asyncio
    async def test_process_async_injects_provenance_marker(self):
        """The pass-through may add a small provenance tag, but does not
        drop or transform the client's payload."""
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        result = await plugin.process_async({"k": "v"})
        assert result["k"] == "v"
        assert result["reflected_by"] == "demo_reflector"

    def test_process_sync_is_no_op_pass_through(self):
        """The legacy sync process() is implemented as a pass-through too,
        so the plugin remains concrete (Plugin.process is abstract)."""
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        ctx = {"x": 1}
        result = plugin.process(ctx)
        assert result is ctx
        assert result == {"x": 1}

    def test_cleanup_resets_counter_and_flags(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_reflector")
        assert plugin is not None
        # Bump the counter before cleaning up.
        import asyncio

        asyncio.run(plugin.process_async({}))
        assert plugin.reflect_count == 1
        plugin.cleanup()
        assert plugin.reflect_count == 0
        assert plugin.initialized is False
        assert plugin.validated is False


# -----------------------------------------------------------------------------
# DemoAdvisorPlugin behaviour
# -----------------------------------------------------------------------------


class TestDemoAdvisorPlugin:
    """Behavioural tests for the demo_advisor built-in."""

    def test_default_plugin_type_is_advisor(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        assert plugin.plugin_type is PluginType.ADVISOR

    def test_lifecycle_init_validate(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        assert plugin.initialized is True
        assert plugin.validated is True

    def test_process_async_is_coroutine_function(self):
        import inspect

        from plugins.demo_advisor.plugin import DemoAdvisorPlugin

        assert inspect.iscoroutinefunction(DemoAdvisorPlugin.process_async)

    @pytest.mark.asyncio
    async def test_process_async_returns_context_unchanged(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        ctx = {"request": {"model": "deepseek-v4-pro:cloud"}, "step": "advice"}
        result = await plugin.process_async(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_process_async_is_no_op_pass_through(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        ctx = {
            "request": {"model": "minimax-m3:cloud", "messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"content": "hello"}}]},
            "turn": 0,
        }
        result = await plugin.process_async(ctx)
        assert result["request"] == ctx["request"]
        assert result["response"] == ctx["response"]
        assert result["turn"] == 0

    @pytest.mark.asyncio
    async def test_process_async_increments_internal_counter(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        assert plugin.advise_count == 0
        await plugin.process_async({})
        assert plugin.advise_count == 1
        await plugin.process_async({})
        assert plugin.advise_count == 2

    @pytest.mark.asyncio
    async def test_process_async_injects_provenance_marker(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        result = await plugin.process_async({"k": "v"})
        assert result["k"] == "v"
        assert result["advised_by"] == "demo_advisor"

    def test_process_sync_is_no_op_pass_through(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        ctx = {"x": 1}
        result = plugin.process(ctx)
        assert result is ctx
        assert result == {"x": 1}

    def test_cleanup_resets_counter_and_flags(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        plugin = mgr.get_plugin("demo_advisor")
        assert plugin is not None
        import asyncio

        asyncio.run(plugin.process_async({}))
        assert plugin.advise_count == 1
        plugin.cleanup()
        assert plugin.advise_count == 0
        assert plugin.initialized is False
        assert plugin.validated is False


# -----------------------------------------------------------------------------
# PluginManager.run dispatches the new plugins through process_async
# -----------------------------------------------------------------------------


class TestManagerDispatchesDemoReflectorAndAdvisor:
    """When the manager runs the new plugin types, their async hooks fire."""

    @pytest.mark.asyncio
    async def test_manager_run_reflector_uses_demo_reflector_process_async(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        ctx: dict = {"step": "reflect"}
        result = await mgr.run(ctx, plugin_types=[PluginType.REFLECTOR])
        # demo_reflector's process_async injected the marker
        assert result["reflected_by"] == "demo_reflector"
        assert result["step"] == "reflect"

    @pytest.mark.asyncio
    async def test_manager_run_advisor_uses_demo_advisor_process_async(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        ctx: dict = {"step": "advise"}
        result = await mgr.run(ctx, plugin_types=[PluginType.ADVISOR])
        assert result["advised_by"] == "demo_advisor"
        assert result["step"] == "advise"

    @pytest.mark.asyncio
    async def test_manager_run_reflector_then_advisor_increments_both(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        reflector = mgr.get_plugin("demo_reflector")
        advisor = mgr.get_plugin("demo_advisor")
        assert reflector is not None and advisor is not None

        await mgr.run({}, plugin_types=[PluginType.REFLECTOR, PluginType.ADVISOR])
        assert reflector.reflect_count == 1
        assert advisor.advise_count == 1
        await mgr.run({}, plugin_types=[PluginType.REFLECTOR, PluginType.ADVISOR])
        assert reflector.reflect_count == 2
        assert advisor.advise_count == 2
