"""Plugin type enumeration."""

from enum import Enum


class PluginType(Enum):
    ROUTER = "router"
    TRANSFORMER = "transformer"
    AUTH = "auth"
    MIDDLEWARE = "middleware"
