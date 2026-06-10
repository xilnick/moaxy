"""Adapters that bridge moaxy to upstream LLM providers.

Each adapter wraps one provider's HTTP API behind the uniform
:class:`moaxy.adapters.base.Adapter` interface. The
:class:`moaxy.adapters.ollama.OllamaAdapter` is the only concrete adapter
shipped in M1; the OpenAI-compatible adapter will be added in a later
milestone.
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

__all__ = [
    "Adapter",
    "ChatResponse",
    "Message",
    "OllamaAdapter",
    "UpstreamError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "Usage",
    "UsageAccumulator",
]
