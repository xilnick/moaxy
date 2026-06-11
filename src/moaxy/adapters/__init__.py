"""Adapters that bridge moaxy to upstream LLM providers.

Each adapter wraps one provider's HTTP API behind the uniform
:class:`moaxy.adapters.base.Adapter` interface. The
:class:`moaxy.adapters.ollama.OllamaAdapter` is the M1 Ollama adapter;
:class:`moaxy.adapters.openrouter.OpenRouterAdapter` is the M6 OpenRouter
adapter. The OpenAI-compatible adapter is reserved for a future milestone.

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
from moaxy.adapters.openrouter import OpenRouterAdapter, OpenRouterConfigError
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
    "OpenRouterAdapter",
    "OpenRouterConfigError",
    "UnknownAdapterError",
    "UpstreamError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "Usage",
    "UsageAccumulator",
    "build_registry",
]
