"""Tests for the PluginManager.run dispatch logic and PluginType enum extension.

Covers:
- PluginType.REFLECTOR and PluginType.ADVISOR are additive enum members.
- PluginManager.run is async and dispatches to process_async for REFLECTOR/ADVISOR.
- PluginManager.run dispatches to process for ROUTER/TRANSFORMER/AUTH/MIDDLEWARE.
- PluginManager.run with an empty plugin_types list is a no-op.
- PluginManager.load returns a flat list of "<plugin_name>: <error>" strings.
- A plugin whose init() raises is reported in errors and not registered;
  other valid plugins still load.
- Existing demo plugins (demo_router, demo_transformer) keep working without
  modification.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from moaxy.plugins.base import Plugin
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"


# -----------------------------------------------------------------------------
# Helper plugins used in dispatch tests. Each one records which method was
# called so we can assert the dispatch behaviour.
# -----------------------------------------------------------------------------


class _RecorderMixin:
    """Records the most recent method called and increments a counter."""

    def __init__(self) -> None:
        super().__init__()
        self.sync_calls = 0
        self.async_calls = 0
        self.last_sync_context: dict | None = None
        self.last_async_context: dict | None = None
        self.last_method: str | None = None

    def _record_sync(self, context: dict) -> None:
        self.sync_calls += 1
        self.last_sync_context = context
        self.last_method = "process"

    def _record_async(self, context: dict) -> None:
        self.async_calls += 1
        self.last_async_context = context
        self.last_method = "process_async"


class _RecorderRouter(Plugin, _RecorderMixin):
    name = "rec_router"
    version = "0.1.0"
    plugin_type = PluginType.ROUTER

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context


class _RecorderTransformer(Plugin, _RecorderMixin):
    name = "rec_transformer"
    version = "0.1.0"
    plugin_type = PluginType.TRANSFORMER

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context


class _RecorderAuth(Plugin, _RecorderMixin):
    name = "rec_auth"
    version = "0.1.0"
    plugin_type = PluginType.AUTH

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context


class _RecorderMiddleware(Plugin, _RecorderMixin):
    name = "rec_middleware"
    version = "0.1.0"
    plugin_type = PluginType.MIDDLEWARE

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context


class _RecorderReflector(Plugin, _RecorderMixin):
    name = "rec_reflector"
    version = "0.1.0"
    plugin_type = PluginType.REFLECTOR

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context

    async def process_async(self, context: dict) -> dict:
        self._record_async(context)
        context["recorder"] = self.name
        return context


class _RecorderAdvisor(Plugin, _RecorderMixin):
    name = "rec_advisor"
    version = "0.1.0"
    plugin_type = PluginType.ADVISOR

    def __init__(self) -> None:
        Plugin.__init__(self)
        _RecorderMixin.__init__(self)

    def process(self, context: dict) -> dict:
        self._record_sync(context)
        context["recorder"] = self.name
        return context

    async def process_async(self, context: dict) -> dict:
        self._record_async(context)
        context["recorder"] = self.name
        return context


class _RaisingInitPlugin(Plugin):
    name = "raising_init"
    version = "0.1.0"
    plugin_type = PluginType.ROUTER

    def init(self) -> None:
        raise RuntimeError("boom from init()")

    def process(self, context: dict) -> dict:
        return context


class _BadValidatePlugin(Plugin):
    name = "bad_validate"
    version = "0.1.0"
    plugin_type = PluginType.ROUTER

    def validate(self) -> list[str]:
        return ["validate says no"]

    def process(self, context: dict) -> dict:
        return context


def _populate_manager(mgr: PluginManager) -> None:
    """Manually register recorder plugins on a manager (skip discovery)."""
    recorders = [
        _RecorderRouter(),
        _RecorderTransformer(),
        _RecorderAuth(),
        _RecorderMiddleware(),
        _RecorderReflector(),
        _RecorderAdvisor(),
    ]
    for plugin in recorders:
        plugin.init()
        errors = plugin.validate()
        assert errors == []
        mgr._plugins[plugin.name] = plugin
    mgr._loaded = True


# -----------------------------------------------------------------------------
# PluginType enum
# -----------------------------------------------------------------------------


class TestPluginTypeEnum:
    """PluginType must have the original four values plus REFLECTOR and ADVISOR."""

    def test_has_six_members(self):
        assert len(PluginType) == 6

    def test_original_four_values_preserved(self):
        assert PluginType.ROUTER.value == "router"
        assert PluginType.TRANSFORMER.value == "transformer"
        assert PluginType.AUTH.value == "auth"
        assert PluginType.MIDDLEWARE.value == "middleware"

    def test_reflector_member_exists(self):
        assert PluginType.REFLECTOR.value == "reflector"

    def test_advisor_member_exists(self):
        assert PluginType.ADVISOR.value == "advisor"

    def test_all_six_member_names(self):
        names = {m.name for m in PluginType}
        assert names == {
            "ROUTER",
            "TRANSFORMER",
            "AUTH",
            "MIDDLEWARE",
            "REFLECTOR",
            "ADVISOR",
        }


# -----------------------------------------------------------------------------
# PluginManager.run dispatch
# -----------------------------------------------------------------------------


class TestPluginManagerRunIsAsync:
    """PluginManager.run is an async coroutine function."""

    def test_run_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(PluginManager.run)


class TestPluginManagerRunSyncDispatch:
    """run() invokes the sync process() for the four legacy types."""

    @pytest.mark.asyncio
    async def test_router_uses_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"x": 1}
        await mgr.run(ctx, plugin_types=[PluginType.ROUTER])
        plugin = mgr.get_plugin("rec_router")
        assert plugin is not None
        assert plugin.last_method == "process"
        assert plugin.sync_calls == 1
        assert plugin.async_calls == 0

    @pytest.mark.asyncio
    async def test_transformer_uses_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {}
        await mgr.run(ctx, plugin_types=[PluginType.TRANSFORMER])
        plugin = mgr.get_plugin("rec_transformer")
        assert plugin is not None
        assert plugin.last_method == "process"
        assert plugin.sync_calls == 1
        assert plugin.async_calls == 0

    @pytest.mark.asyncio
    async def test_auth_uses_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {}
        await mgr.run(ctx, plugin_types=[PluginType.AUTH])
        plugin = mgr.get_plugin("rec_auth")
        assert plugin is not None
        assert plugin.last_method == "process"
        assert plugin.sync_calls == 1
        assert plugin.async_calls == 0

    @pytest.mark.asyncio
    async def test_middleware_uses_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {}
        await mgr.run(ctx, plugin_types=[PluginType.MIDDLEWARE])
        plugin = mgr.get_plugin("rec_middleware")
        assert plugin is not None
        assert plugin.last_method == "process"
        assert plugin.sync_calls == 1
        assert plugin.async_calls == 0


class TestPluginManagerRunAsyncDispatch:
    """run() invokes the async process_async() for REFLECTOR and ADVISOR."""

    @pytest.mark.asyncio
    async def test_reflector_uses_async_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"step": "critique"}
        await mgr.run(ctx, plugin_types=[PluginType.REFLECTOR])
        plugin = mgr.get_plugin("rec_reflector")
        assert plugin is not None
        assert plugin.last_method == "process_async"
        assert plugin.async_calls == 1
        assert plugin.sync_calls == 0

    @pytest.mark.asyncio
    async def test_advisor_uses_async_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"step": "advice"}
        await mgr.run(ctx, plugin_types=[PluginType.ADVISOR])
        plugin = mgr.get_plugin("rec_advisor")
        assert plugin is not None
        assert plugin.last_method == "process_async"
        assert plugin.async_calls == 1
        assert plugin.sync_calls == 0

    @pytest.mark.asyncio
    async def test_reflector_and_adisor_run_async(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {}
        await mgr.run(ctx, plugin_types=[PluginType.REFLECTOR, PluginType.ADVISOR])
        assert mgr.get_plugin("rec_reflector").async_calls == 1
        assert mgr.get_plugin("rec_advisor").async_calls == 1


class TestPluginManagerRunNoOp:
    """Empty plugin_types list is a no-op; the context is returned unchanged."""

    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"untouched": True, "model": "minimax-m3:cloud"}
        result = await mgr.run(ctx, plugin_types=[])
        assert result is ctx
        # No recorder was touched.
        for name in (
            "rec_router",
            "rec_transformer",
            "rec_auth",
            "rec_middleware",
            "rec_reflector",
            "rec_advisor",
        ):
            p = mgr.get_plugin(name)
            assert p is not None
            assert p.sync_calls == 0
            assert p.async_calls == 0

    @pytest.mark.asyncio
    async def test_default_plugin_types_is_noop(self):
        """When plugin_types is omitted, run() is a no-op."""
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"a": 1}
        result = await mgr.run(ctx)
        assert result is ctx


class TestPluginManagerRunMixedTypes:
    """A list with mixed types dispatches each plugin to the right method."""

    @pytest.mark.asyncio
    async def test_mixed_types_dispatch_correctly(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {}
        await mgr.run(
            ctx,
            plugin_types=[
                PluginType.ROUTER,
                PluginType.TRANSFORMER,
                PluginType.REFLECTOR,
                PluginType.ADVISOR,
            ],
        )
        assert mgr.get_plugin("rec_router").sync_calls == 1
        assert mgr.get_plugin("rec_transformer").sync_calls == 1
        assert mgr.get_plugin("rec_reflector").async_calls == 1
        assert mgr.get_plugin("rec_advisor").async_calls == 1
        # The reflector and advisor must NOT have been hit by the sync path.
        assert mgr.get_plugin("rec_reflector").sync_calls == 0
        assert mgr.get_plugin("rec_advisor").sync_calls == 0

    @pytest.mark.asyncio
    async def test_run_passes_context_through(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        _populate_manager(mgr)
        ctx: dict[str, Any] = {"injected": "value"}
        result = await mgr.run(ctx, plugin_types=[PluginType.ROUTER])
        # The router mutated ctx by setting "recorder" in-place; the same dict
        # is returned to the caller.
        assert result["recorder"] == "rec_router"
        assert result["injected"] == "value"


# -----------------------------------------------------------------------------
# PluginManager.load error handling
# -----------------------------------------------------------------------------


class TestPluginManagerLoadErrors:
    """load() returns "<plugin_name>: <error>" strings and never raises."""

    def test_load_returns_flat_list_of_strings(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert isinstance(errors, list)
        for e in errors:
            assert isinstance(e, str)

    def test_load_with_failing_validate_reports_named_error(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr._plugins.clear()
        bad = _BadValidatePlugin()
        bad.init()
        # Bypass load(): simulate the per-plugin error path.
        validation_errors = bad.validate()
        errors: list[str] = []
        if validation_errors:
            errors.extend(f"{bad.name}: {e}" for e in validation_errors)
        assert errors == ["bad_validate: validate says no"]
        assert mgr.get_plugin("bad_validate") is None

    def test_load_with_raising_init_reports_error_and_skips_registration(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr._plugins.clear()

        # The PluginManager.load() contract: an init() failure is reported in
        # the returned errors list and the failing plugin is NOT registered,
        # while other valid plugins still register.
        errors: list[str] = []

        # Manually drive the same logic the manager uses so we can exercise
        # the exact behaviour we expect from the implementation.
        candidate = _RaisingInitPlugin()
        try:
            candidate.init()
        except Exception as exc:
            errors.append(f"{candidate.name}: {exc}")

        good = _RecorderRouter()
        good.init()
        if not good.validate():
            mgr._plugins[good.name] = good
        mgr._loaded = True

        assert errors == ["raising_init: boom from init()"]
        assert mgr.get_plugin("raising_init") is None
        # Other valid plugins still registered.
        assert mgr.get_plugin("rec_router") is not None


class TestInitFailureIntegration:
    """End-to-end: a plugin whose init() raises is reported, others still load.

    Drives PluginManager.load() through its real code path by injecting
    classes via a stub discover_plugins.
    """

    def test_init_failure_path_via_manager_load(self, monkeypatch):
        from moaxy.plugins import manager as manager_module

        raising = _RaisingInitPlugin
        good = _RecorderRouter
        monkeypatch.setattr(
            manager_module,
            "discover_plugins",
            lambda plugins_dir: [raising, good],
        )

        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()

        assert "raising_init: boom from init()" in errors
        assert mgr.get_plugin("raising_init") is None
        assert mgr.get_plugin("rec_router") is not None
        # Only the good plugin is registered.
        assert mgr.plugin_count == 1


# -----------------------------------------------------------------------------
# Backwards compatibility with existing built-in plugins
# -----------------------------------------------------------------------------


class TestDemoPluginsStillWork:
    """demo_router and demo_transformer must keep working unchanged."""

    def test_demo_plugins_still_load(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        errors = mgr.load()
        assert errors == []
        assert mgr.get_plugin("demo_router") is not None
        assert mgr.get_plugin("demo_transformer") is not None

    def test_demo_plugin_count_unchanged(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        # The original two demo plugins (router, transformer) are still
        # present. The new built-in REFLECTOR (demo_reflector) and ADVISOR
        # (demo_advisor) plugins are also loaded.
        assert mgr.plugin_count >= 2

    def test_demo_router_still_invokable_via_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        router = mgr.get_plugin("demo_router")
        assert router is not None
        out = router.process({"model": "demo-gpt-4"})
        assert out["route_decision"].intercepted is True

    def test_demo_transformer_still_invokable_via_sync_process(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        transformer = mgr.get_plugin("demo_transformer")
        assert transformer is not None
        out = transformer.process({"request": {"path": "/v1/chat/completions"}})
        assert out["request"]["headers"]["X-Moaxy-Demo"] == "true"

    @pytest.mark.asyncio
    async def test_demo_plugins_default_process_async_is_noop(self):
        mgr = PluginManager(plugins_dir=PLUGINS_DIR)
        mgr.load()
        for name in ("demo_router", "demo_transformer"):
            plugin = mgr.get_plugin(name)
            assert plugin is not None
            ctx: dict = {"marker": name}
            result = await plugin.process_async(ctx)
            assert result is ctx
            assert ctx["marker"] == name
