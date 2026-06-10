"""Test fixtures shared across the test suite.

The fixtures here are deliberately small and in-process so that the
tests stay hermetic and do not require a running uvicorn or Ollama.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from moaxy.adapters.ollama import OllamaAdapter
from moaxy.models.config import (
    AdapterConfig,
    MoaxyConfig,
    RouteConfig,
)
from moaxy.models.config import (
    RouteMatch as ConfigRouteMatch,
)


class _FakeOllamaTransport(httpx.AsyncBaseTransport):
    """A programmable httpx transport that returns scripted responses."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._handler(request)


def make_json_response(
    payload: dict[str, Any], status_code: int = 200
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def make_ollama_payload(
    *,
    content: str = "hello",
    model: str = "minimax-m2.7:cloud",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    total_tokens: int | None = None,
    finish_reason: str = "stop",
    chatcmpl_id: str = "chatcmpl-abc123",
) -> dict[str, Any]:
    return {
        "id": chatcmpl_id,
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
            if total_tokens is not None
            else (prompt_tokens + completion_tokens),
        },
    }


def make_ollama_adapter(
    handler,
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 5.0,
) -> OllamaAdapter:
    """Build an :class:`OllamaAdapter` wired to an in-process transport."""
    return OllamaAdapter(
        base_url=base_url,
        timeout=timeout,
        _transport=_FakeOllamaTransport(handler),
    )


def make_config(
    *,
    routes: list[RouteConfig] | None = None,
    backends: list[AdapterConfig] | None = None,
) -> MoaxyConfig:
    """Build a minimal :class:`MoaxyConfig` for tests."""
    return MoaxyConfig(
        backends=backends
        or [AdapterConfig(name="olloma-local", adapter="ollama", base_url="http://127.0.0.1:11434")],
        routes=routes
        or [
            RouteConfig(
                name="r",
                match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                backend="olloma-local",
            )
        ],
    )


__all__ = [
    "make_config",
    "make_json_response",
    "make_ollama_adapter",
    "make_ollama_payload",
]
