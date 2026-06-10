"""Plugin manager — orchestrates plugin lifecycle (load, init, validate, run, cleanup)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from moaxy.plugins.base import Plugin
from moaxy.plugins.discovery import discover_plugins, load_plugin_instance
from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)

_ASYNC_PLUGIN_TYPES: frozenset[PluginType] = frozenset(
    {PluginType.REFLECTOR, PluginType.ADVISOR}
)


class PluginManager:
    """Manages the full lifecycle of all plugins.

    Loading order:
    1. discover — scan plugins_dir for Plugin subclasses
    2. init — call init() on each plugin instance
    3. validate — call validate() on each plugin, collect errors
    4. process — run plugins (routing, transforming, etc.)
    5. cleanup — call cleanup() on each plugin, release resources
    """

    def __init__(self, plugins_dir: str | Path = "plugins") -> None:
        self._plugins_dir = Path(plugins_dir)
        self._plugins: dict[str, Plugin] = {}
        self._loaded = False

    @property
    def plugins_dir(self) -> Path:
        return self._plugins_dir

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return metadata for all registered plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "type": p.plugin_type.value,
                "initialized": p.initialized,
                "validated": p.validated,
            }
            for p in self._plugins.values()
        ]

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[Plugin]:
        """Get all registered plugins of a given type."""
        return [p for p in self._plugins.values() if p.plugin_type == plugin_type]

    def load(self, plugin_configs: dict[str, dict[str, Any]] | None = None) -> list[str]:
        """Discover and initialize all plugins.

        Args:
            plugin_configs: Optional per-plugin config dict keyed by plugin class name.

        Returns:
            Flat list of error messages. Each message has the format
            ``"<plugin_name>: <error_message>"``. An empty list means all
            plugins loaded successfully. Failures during ``init()`` are
            captured here; the failing plugin is not registered but other
            plugins are still loaded.
        """
        if self._loaded:
            logger.warning("Plugins already loaded; skipping duplicate load()")
            return []

        plugin_classes = discover_plugins(self._plugins_dir)
        errors: list[str] = []

        for cls in plugin_classes:
            config = (plugin_configs or {}).get(cls.__name__, {})
            instance = load_plugin_instance(cls, config)
            try:
                instance.init()
            except Exception as exc:
                errors.append(f"{instance.name}: {exc}")
                logger.error(
                    "Plugin %s raised during init() and was not registered: %s",
                    instance.name,
                    exc,
                )
                continue
            validation_errors = instance.validate()
            if validation_errors:
                errors.extend(f"{instance.name}: {e}" for e in validation_errors)
            else:
                self._plugins[instance.name] = instance

        self._loaded = True
        logger.info("Loaded %d plugin(s)", self.plugin_count)
        return errors

    async def run(
        self,
        context: dict[str, Any],
        plugin_types: Iterable[PluginType] | None = None,
    ) -> dict[str, Any]:
        """Run the registered plugins of the requested types against ``context``.

        For each requested type, every registered plugin of that type is
        invoked. Plugins of type :class:`PluginType.REFLECTOR` and
        :class:`PluginType.ADVISOR` are dispatched to their async
        ``process_async`` hook; the four legacy types
        (ROUTER, TRANSFORMER, AUTH, MIDDLEWARE) are dispatched to the
        synchronous ``process`` hook.

        Args:
            context: Mutable context dict shared across all plugins in the
                run. Plugins are expected to return the same dict they
                receive (mutations are visible to subsequent plugins).
            plugin_types: Iterable of plugin types to run. When ``None`` or
                empty, this method is a no-op and the context is returned
                unchanged.

        Returns:
            The same ``context`` dict, after every selected plugin has run.
        """
        if plugin_types is None:
            return context
        types = list(plugin_types)
        if not types:
            return context

        for plugin_type in types:
            for plugin in self.get_plugins_by_type(plugin_type):
                if plugin.plugin_type in _ASYNC_PLUGIN_TYPES:
                    context = await plugin.process_async(context)
                else:
                    context = plugin.process(context)
        return context

    def shutdown(self) -> None:
        """Clean up all plugins and release resources."""
        for name, plugin in list(self._plugins.items()):
            try:
                plugin.cleanup()
            except Exception as exc:
                logger.error("Error cleaning up plugin %s: %s", name, exc)
        self._plugins.clear()
        self._loaded = False
        logger.info("All plugins shut down")
