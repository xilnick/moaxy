"""Plugin manager — orchestrates plugin lifecycle (load, init, validate, run, cleanup)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from moaxy.plugins.base import Plugin
from moaxy.plugins.discovery import discover_plugins, load_plugin_instance
from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)


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
            List of validation error messages (empty = all good).
        """
        if self._loaded:
            logger.warning("Plugins already loaded; skipping duplicate load()")
            return []

        plugin_classes = discover_plugins(self._plugins_dir)
        errors: list[str] = []

        for cls in plugin_classes:
            config = (plugin_configs or {}).get(cls.__name__, {})
            instance = load_plugin_instance(cls, config)
            instance.init()
            validation_errors = instance.validate()
            if validation_errors:
                errors.extend(f"{instance.name}: {e}" for e in validation_errors)
            else:
                self._plugins[instance.name] = instance

        self._loaded = True
        logger.info("Loaded %d plugin(s)", self.plugin_count)
        return errors

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
