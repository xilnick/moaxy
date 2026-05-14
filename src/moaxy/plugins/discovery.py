"""Plugin discovery — scan a directory for plugin packages and load them."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

from moaxy.plugins.base import Plugin

logger = logging.getLogger(__name__)


class PluginDiscoveryError(Exception):
    """Raised when plugin discovery or loading fails."""


def discover_plugins(plugins_dir: str | Path) -> list[type[Plugin]]:
    """Scan the plugins directory and load all plugin classes.

    Each subdirectory under plugins_dir is treated as a potential plugin package.
    A plugin package must contain a `plugin.py` module that exports a Plugin subclass
    as a module-level attribute (or the first Plugin subclass found).
    """
    plugins_dir = Path(plugins_dir).resolve()
    if not plugins_dir.is_dir():
        raise PluginDiscoveryError(f"Plugins directory not found: {plugins_dir}")

    discovered: list[type[Plugin]] = []

    plugins_dir_str = str(plugins_dir)
    if plugins_dir_str not in sys.path:
        sys.path.insert(0, plugins_dir_str)

    for entry in sorted(plugins_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue

        plugin_mod_path = entry / "plugin.py"
        if not plugin_mod_path.is_file():
            logger.debug("Skipping %s: no plugin.py found", entry.name)
            continue

        try:
            plugin_cls = _load_plugin_from_package(entry.name, plugin_mod_path)
            discovered.append(plugin_cls)
            logger.info("Discovered plugin %s from %s", plugin_cls.__name__, entry.name)
        except Exception as exc:
            logger.warning("Failed to load plugin from %s: %s", entry.name, exc)

    return discovered


def _load_plugin_from_package(package_name: str, module_path: Path) -> type[Plugin]:
    """Load a Plugin subclass from a plugin package."""
    spec = importlib.util.spec_from_file_location(
        f"moaxy_plugin_{package_name}",
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise PluginDiscoveryError(f"Cannot load module spec for {package_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Plugin:
            continue
        if issubclass(obj, Plugin) and not inspect.isabstract(obj):
            return obj

    raise PluginDiscoveryError(f"No concrete Plugin subclass found in {package_name}")


def load_plugin_instance(plugin_cls: type[Plugin], config: dict[str, Any] | None = None) -> Plugin:
    """Instantiate a discovered plugin class with optional config."""
    instance = plugin_cls()
    if config:
        for key, value in config.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
    return instance
