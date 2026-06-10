"""Demo reflector plugin — a no-op REFLECTOR pass-through.

Proves that the async ``process_async`` hook is wired through the plugin
manager for ``PluginType.REFLECTOR`` and that the lifecycle (init,
validate, process_async, cleanup) holds end-to-end. Increments an
internal counter on every invocation so tests can verify dispatch.
"""

from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType


class DemoReflectorPlugin(Plugin):
    """Example REFLECTOR plugin that records how many times it ran.

    The base ``process`` hook is implemented as a pass-through to keep the
    plugin concrete; the async ``process_async`` hook is the real surface
    the orchestrator calls and is also a no-op pass-through that bumps a
    counter.
    """

    name = "demo_reflector"
    version = "1.0.0"
    plugin_type = PluginType.REFLECTOR

    def __init__(self) -> None:
        super().__init__()
        self.reflect_count = 0

    def init(self) -> None:
        super().init()
        self.reflect_count = 0

    def validate(self) -> list[str]:
        errors = super().validate()
        return errors

    def process(self, context: dict) -> dict:
        return context

    async def process_async(self, context: dict) -> dict:
        self.reflect_count += 1
        context["reflected_by"] = self.name
        return context

    def cleanup(self) -> None:
        self.reflect_count = 0
        super().cleanup()
