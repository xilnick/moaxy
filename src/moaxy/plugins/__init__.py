"""Plugin system for moaxy.

Extension points for custom routers, transformers, auth handlers, and middleware.
"""

from moaxy.plugins.base import Plugin
from moaxy.plugins.manager import PluginManager
from moaxy.plugins.types import PluginType

__all__ = ["Plugin", "PluginManager", "PluginType"]
