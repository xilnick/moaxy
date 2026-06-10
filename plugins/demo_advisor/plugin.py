"""Demo advisor plugin — a no-op ADVISOR pass-through.

Proves that the async ``process_async`` hook is wired through the plugin
manager for ``PluginType.ADVISOR`` and that the lifecycle (init,
validate, process_async, cleanup) holds end-to-end. Increments an
internal counter on every invocation so tests can verify dispatch.
"""

from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType


class DemoAdvisorPlugin(Plugin):
    """Example ADVISOR plugin that records how many times it ran.

    The base ``process`` hook is implemented as a pass-through to keep the
    plugin concrete; the async ``process_async`` hook is the real surface
    the orchestrator calls and is also a no-op pass-through that bumps a
    counter.
    """

    name = "demo_advisor"
    version = "1.0.0"
    plugin_type = PluginType.ADVISOR

    def __init__(self) -> None:
        super().__init__()
        self.advise_count = 0

    def init(self) -> None:
        super().init()
        self.advise_count = 0

    def validate(self) -> list[str]:
        errors = super().validate()
        return errors

    def process(self, context: dict) -> dict:
        return context

    async def process_async(self, context: dict) -> dict:
        self.advise_count += 1
        context["advised_by"] = self.name
        return context

    def cleanup(self) -> None:
        self.advise_count = 0
        super().cleanup()
