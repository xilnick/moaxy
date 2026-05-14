"""Plugin base class and lifecycle management."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from moaxy.plugins.types import PluginType

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for all moaxy plugins.

    Subclass this and implement the required hooks to create a plugin.
    Each plugin declares a type (router, transformer, auth, middleware)
    and can override lifecycle hooks: init(), validate(), cleanup().
    """

    name: str
    version: str
    plugin_type: PluginType

    def __init__(self) -> None:
        self._initialized = False
        self._validated = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def validated(self) -> bool:
        return self._validated

    def init(self) -> None:
        """Called once when the plugin is loaded and registered.

        Use this to set up resources, open connections, load config, etc.
        Override in subclasses, but call super().init() first.
        """
        self._initialized = True
        logger.info("Plugin %s (%s) initialized", self.name, self.plugin_type.value)

    def validate(self) -> list[str]:
        """Validate that the plugin is correctly configured and ready.

        Called after init(). Return a list of error messages (empty = valid).
        Override in subclasses to add custom validation logic.
        """
        errors: list[str] = []
        if not self.name:
            errors.append("Plugin name must be set")
        if not isinstance(self.plugin_type, PluginType):
            errors.append("Plugin must declare a valid plugin_type")
        if not errors:
            self._validated = True
        return errors

    def cleanup(self) -> None:
        """Called when the plugin is being unregistered or the proxy shuts down.

        Use this to close connections, flush buffers, release resources.
        Override in subclasses, but call super().cleanup() at the end.
        """
        self._initialized = False
        self._validated = False
        logger.info("Plugin %s (%s) cleaned up", self.name, self.plugin_type.value)

    @abstractmethod
    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process a request/response through the plugin.

        For routers: context contains request details, returns routing decision.
        For transformers: context contains request/response, returns modified version.
        For auth handlers: context contains credentials, returns auth result.
        For middleware: context contains full request context, returns modified context.

        Args:
            context: Plugin-specific context dict with request/response data.

        Returns:
            Processed context dict with modifications applied.
        """
        ...
