"""Adapters that bridge moaxy to upstream LLM providers.

Each adapter wraps one provider's HTTP API behind the uniform
:class:`moaxy.adapters.base.Adapter` interface. The
:class:`moaxy.adapters.ollama.OllamaAdapter` is the only concrete adapter
shipped in M1; the OpenAI-compatible adapter will be added in a later
milestone.

The :class:`moaxy.adapters.registry.AdapterRegistry` constructs
:class:`Adapter` instances from the ``backends`` list of a parsed
:class:`moaxy.models.config.MoaxyConfig`, keyed by each config's ``name``
field.
"""

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    Usage,
    UsageAccumulator,
)
from moaxy.adapters.ollama import OllamaAdapter
from moaxy.adapters.registry import (
    AdapterRegistry,
    DuplicateAdapterNameError,
    UnknownAdapterError,
    build_registry,
)

__all__ = [
    "Adapter",
    "AdapterRegistry",
    "ChatResponse",
    "DuplicateAdapterNameError",
    "Message",
    "OllamaAdapter",
    "UnknownAdapterError",
    "UpstreamError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "Usage",
    "UsageAccumulator",
    "build_registry",
]
