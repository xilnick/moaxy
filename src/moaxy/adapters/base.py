"""Adapter base types and the upstream-error hierarchy.

Concrete adapters (e.g. :class:`moaxy.adapters.ollama.OllamaAdapter`) inherit
from :class:`Adapter` and implement the async ``chat`` and ``stream`` methods.
Callers catch :class:`UpstreamError` (and its subclasses
:class:`UpstreamTimeoutError` and :class:`UpstreamUnavailableError`) to
distinguish transport-level failures from application errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
        content: Message text. May be the empty string.
    """

    role: str
    content: str


@dataclass
class Usage:
    """Token usage for a single upstream call.

    The default values of zero model the case where the upstream does not
    report usage; the adapter still produces a valid :class:`Usage` instance.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """A normalised chat-completion response.

    Mirrors the shape of OpenAI's ``chat.completion`` object: the model that
    produced the answer, the chosen message, the finish reason, and token
    usage. Adapters parse the upstream JSON into this shape so the rest of
    moaxy can stay backend-agnostic.
    """

    id: str
    model: str
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


class UsageAccumulator:
    """Sums token usage across multiple upstream calls.

    The pipeline orchestrator (M2+) holds a :class:`UsageAccumulator` per
    request and adds each adapter call's usage to it; the snapshot is
    exposed in the final response.
    """

    def __init__(self) -> None:
        self._prompt = 0
        self._completion = 0
        self._total = 0

    def add(self, usage: Usage) -> None:
        """Add a :class:`Usage` sample to the running totals."""
        self._prompt += usage.prompt_tokens
        self._completion += usage.completion_tokens
        self._total += usage.total_tokens
        if usage.total_tokens == 0 and (usage.prompt_tokens or usage.completion_tokens):
            # Some upstreams (e.g. certain Ollama builds) omit ``total_tokens``.
            # Mirror the prompt + completion sum so the snapshot invariant
            # ``total == prompt + completion`` still holds.
            self._total += usage.prompt_tokens + usage.completion_tokens

    def snapshot(self) -> Usage:
        """Return the current totals as a :class:`Usage` value object."""
        return Usage(
            prompt_tokens=self._prompt,
            completion_tokens=self._completion,
            total_tokens=self._total,
        )

    def reset(self) -> None:
        """Reset the running totals to zero."""
        self._prompt = 0
        self._completion = 0
        self._total = 0


class UpstreamError(Exception):
    """Raised when the upstream returns an error or its response is unusable.

    The ``status_code`` attribute carries the HTTP status (when available).
    The ``body`` attribute carries the raw response body (when available),
    useful for surfacing the upstream's error message in the proxy's own
    error responses without losing fidelity.

    Subclasses :class:`UpstreamTimeoutError` and
    :class:`UpstreamUnavailableError` cover transport-level failures; they
    are caught by the fallback walker in :mod:`moaxy.pipeline.fallback`.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out (read, connect, or pool)."""

    def __init__(self, message: str = "upstream timeout") -> None:
        super().__init__(message, status_code=None, body=None)


class UpstreamUnavailableError(UpstreamError):
    """Raised when the upstream is unreachable (connection refused, DNS, etc.)."""

    def __init__(self, message: str = "upstream unavailable") -> None:
        super().__init__(message, status_code=None, body=None)


class Adapter(ABC):
    """Abstract base for all backend adapters.

    A concrete adapter wraps one upstream provider (Ollama, OpenAI, etc.)
    and exposes a uniform async interface so the rest of moaxy is
    provider-agnostic.
    """

    name: str = ""

    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat-completion request.

        Args:
            model: The model identifier understood by the upstream.
            messages: OpenAI-shaped messages list. Each item is a dict with
                at least ``role`` and ``content`` keys.
            **kwargs: Provider-specific request parameters forwarded
                verbatim (``max_tokens``, ``temperature``, ``top_p``,
                ``stop``, ``stream``, etc.).

        Returns:
            A :class:`ChatResponse` normalised to the OpenAI
            ``chat.completion`` shape.

        Raises:
            UpstreamError: The upstream returned a 4xx/5xx or its response
                could not be decoded.
            UpstreamTimeoutError: The upstream timed out.
            UpstreamUnavailableError: The upstream could not be reached.
        """
        ...

    @abstractmethod
    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat-completion request.

        Yields text deltas as they arrive from the upstream. The default
        implementation in concrete adapters is a sync method that returns
        an async generator. Adapters MUST yield plain ``str`` chunks so the
        SSE layer can serialise them without further parsing.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the adapter (e.g. the HTTP client)."""
        ...


__all__ = [
    "Adapter",
    "ChatResponse",
    "Message",
    "UpstreamError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "Usage",
    "UsageAccumulator",
]
