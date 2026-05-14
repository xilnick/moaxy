from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType


class DemoTransformerPlugin(Plugin):
    """Example transformer plugin that modifies request/response payloads.

    Demonstrates the plugin interface with request header injection
    and response body modification capabilities.
    """

    name = "demo_transformer"
    version = "1.0.0"
    plugin_type = PluginType.TRANSFORMER

    def __init__(self) -> None:
        super().__init__()
        self.header_prefix: str = "X-Moaxy-"
        self.inject_header: str = "X-Moaxy-Demo"
        self.inject_value: str = "true"
        self.transform_count = 0

    def init(self) -> None:
        super().init()
        self.transform_count = 0

    def validate(self) -> list[str]:
        errors = super().validate()
        if not self.header_prefix:
            errors.append("header_prefix must not be empty")
        if not self.inject_header.startswith(self.header_prefix):
            errors.append(
                f"inject_header must start with prefix '{self.header_prefix}'"
            )
        return errors

    def process(self, context: dict) -> dict:
        self.transform_count += 1

        if "request" in context:
            req = context["request"]
            headers = req.setdefault("headers", {})
            headers[self.inject_header] = self.inject_value

        if "response" in context:
            resp = context["response"]
            if isinstance(resp, dict):
                resp["proxy"] = self.name
                resp["transform_count"] = self.transform_count

        context["transformed_by"] = self.name
        return context

    def cleanup(self) -> None:
        self.transform_count = 0
        super().cleanup()
