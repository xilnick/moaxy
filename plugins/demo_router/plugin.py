from dataclasses import dataclass, field

from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType


@dataclass
class RouteDecision:
    target: str = ""
    model: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    intercepted: bool = False
    intercepted_by: str = ""
    intercepted_reason: str = ""


class DemoRouterPlugin(Plugin):
    """Example router plugin that can intercept and redirect requests.

    Demonstrates the plugin interface: init, validate, process, cleanup.
    Routes requests based on a configurable model prefix and target.
    """

    name = "demo_router"
    version = "1.0.0"
    plugin_type = PluginType.ROUTER

    def __init__(self) -> None:
        super().__init__()
        self.model_prefix: str = "demo-"
        self.default_target: str = "http://localhost:11434"
        self.redirect_target: str = "http://localhost:8080"
        self.route_count = 0

    def init(self) -> None:
        super().init()
        self.route_count = 0

    def validate(self) -> list[str]:
        errors = super().validate()
        if not self.default_target.startswith(("http://", "https://")):
            errors.append("default_target must be a valid URL")
        if not self.redirect_target.startswith(("http://", "https://")):
            errors.append("redirect_target must be a valid URL")
        return errors

    def process(self, context: dict) -> dict:
        self.route_count += 1
        model = context.get("model", "")
        decision = RouteDecision()

        if model.startswith(self.model_prefix):
            decision.target = self.redirect_target
            decision.model = model[len(self.model_prefix) :]
            decision.intercepted = True
            decision.intercepted_by = self.name
            decision.intercepted_reason = (
                f"Model '{model}' matched prefix '{self.model_prefix}'"
            )
        else:
            decision.target = self.default_target
            decision.model = model

        context["route_decision"] = decision
        context["routed_by"] = self.name
        return context

    def cleanup(self) -> None:
        self.route_count = 0
        super().cleanup()
