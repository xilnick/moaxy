"""OpenRouterAdapter — wraps the OpenRouter.ai OpenAI-compatible API.

OpenRouter (https://openrouter.ai) routes requests to 300+ upstream models
(Anthropic, OpenAI, Google, Meta, Mistral, etc.) behind a single
OpenAI-shaped ``/api/v1/chat/completions`` endpoint. This adapter mirrors
the structure of :class:`moaxy.adapters.ollama.OllamaAdapter` and adds the
OpenRouter-specific affordances:

* ``Authorization: Bearer ${OPENROUTER_API_KEY}`` on every request (the
  key is read from the environment at construction time; the adapter
  refuses to start when missing).
* ``HTTP-Referer`` and ``X-Title`` headers when configured (used by
  OpenRouter for app attribution and ranking).
* ``transforms`` array in the request body when configured (OpenRouter
  message-postprocessing pipeline; e.g. ``["middle-out"]``).
* Server-sent-events streaming (``text/event-stream``) with
  ``data: {...}\n\n`` frames and a ``[DONE]`` terminator.

The adapter's :meth:`__repr__` redacts the API key so it is safe to
include the adapter in debug logs and error messages without leaking
the secret.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    Usage,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_S = 60.0
CHAT_COMPLETIONS_PATH = "/chat/completions"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


class OpenRouterConfigError(RuntimeError):
    """Raised when the OpenRouter adapter cannot be configured.

    Examples: the ``OPENROUTER_API_KEY`` env var is unset; the constructor
    is called with an obviously bad ``base_url``. Mirrors the
    :class:`RuntimeError` contract documented in the M6 spec; a
    domain-specific subclass lets callers ``except OpenRouterConfigError``
    without colliding with generic :class:`RuntimeError` consumers.
    """


class OpenRouterAdapter(Adapter):
    """Adapter for OpenRouter.ai's OpenAI-compatible chat-completion API.

    Args:
        base_url: Root URL of the OpenRouter API. Defaults to
            ``https://openrouter.ai/api/v1``. Trailing slashes are stripped.
        timeout: Per-call timeout in seconds for the underlying
            ``httpx.AsyncClient``. Defaults to 60 seconds (OpenRouter can
            be slow on the first request to a cold upstream model).
        http_referer: Optional URL sent as the ``HTTP-Referer`` header
            on every request. Used by OpenRouter for app attribution.
        x_title: Optional app title sent as the ``X-Title`` header on
            every request. Used by OpenRouter for app ranking.
        transforms: Optional list of OpenRouter message transforms
            (e.g. ``["middle-out"]``) included in the request body.
        _transport: Internal/test hook. Lets unit tests inject an
            in-process :class:`httpx.AsyncBaseTransport`. Production
            code leaves this ``None`` so a real network client is used.

    Raises:
        OpenRouterConfigError: The ``OPENROUTER_API_KEY`` environment
            variable is missing or empty. Raised at construction time
            (NOT at request time) so misconfiguration fails fast.
    """

    name = "openrouter"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        http_referer: str | None = None,
        x_title: str | None = None,
        transforms: list[str] | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key or not api_key.strip():
            raise OpenRouterConfigError(
                f"{API_KEY_ENV_VAR} environment variable is not set or is empty; "
                f"the OpenRouterAdapter requires a valid OpenRouter API key"
            )
        self._api_key: str = api_key
        self._raw_base_url = base_url
        self.base_url = self._normalise_base_url(base_url)
        self.timeout = float(timeout)
        self.http_referer = http_referer
        self.x_title = x_title
        self.transforms = list(transforms) if transforms else None
        self._transport = _transport
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        return base_url.rstrip("/") if base_url else DEFAULT_BASE_URL

    @property
    def endpoint(self) -> str:
        """Full URL of the ``/v1/chat/completions`` endpoint."""
        return f"{self.base_url}{CHAT_COMPLETIONS_PATH}"

    def _get_client(self) -> httpx.AsyncClient:
        """Return the lazily-initialised ``httpx.AsyncClient``."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            if self.http_referer:
                headers["HTTP-Referer"] = self.http_referer
            if self.x_title:
                headers["X-Title"] = self.x_title
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` and release its socket."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the JSON body for a chat-completion request.

        The OpenRouter-specific ``transforms`` field is appended when
        configured. Other provider-specific kwargs (``max_tokens``,
        ``temperature``, ``top_p``, ``stop``, ``stream``) are forwarded
        verbatim via ``**kwargs``.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)
        if self.transforms:
            payload["transforms"] = list(self.transforms)
        return payload

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat-completion request and parse the reply.

        Returns:
            A :class:`ChatResponse` with id/model/message/usage/finish_reason
            populated.

        Raises:
            UpstreamTimeoutError: The request timed out.
            UpstreamUnavailableError: The upstream is unreachable or closed
                the connection mid-response.
            UpstreamError: The upstream returned a 4xx/5xx or a malformed
                body.
        """
        payload = self._build_payload(model=model, messages=messages, **kwargs)

        client = self._get_client()
        try:
            response = await client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning("OpenRouterAdapter timeout for model=%s: %s", model, exc)
            raise UpstreamTimeoutError(f"timeout talking to {self.base_url}") from exc
        except httpx.ConnectError as exc:
            logger.warning("OpenRouterAdapter connect error for model=%s: %s", model, exc)
            raise UpstreamUnavailableError(
                f"cannot connect to {self.base_url}"
            ) from exc
        except httpx.RemoteProtocolError as exc:
            logger.warning(
                "OpenRouterAdapter remote protocol error for model=%s: %s", model, exc
            )
            raise UpstreamUnavailableError(
                f"upstream {self.base_url} closed the connection"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("OpenRouterAdapter request error for model=%s: %s", model, exc)
            raise UpstreamUnavailableError(
                f"transport error talking to {self.base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            body_text = response.text
            message = self._extract_error_message(body_text) or (
                f"upstream returned HTTP {response.status_code}"
            )
            raise UpstreamError(
                message,
                status_code=response.status_code,
                body=body_text,
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise UpstreamError(
                f"failed to decode upstream response: {exc}",
                status_code=response.status_code,
                body=response.text,
            ) from exc

        return self._parse_chat_response(data)

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text deltas from a streaming chat-completion request.

        OpenRouter speaks the OpenAI SSE format: ``data: {...}\\n\\n``
        frames, each a partial ``chat.completion.chunk`` object, followed
        by a final ``data: [DONE]\\n\\n`` terminator. We pull the
        ``choices[0].delta.content`` field out of each chunk and yield it
        as a plain ``str``.

        The ``[DONE]`` line is consumed silently and never yielded.
        ``finish_reason: "stop"`` chunks (with no content) are also
        consumed silently so the orchestrator sees only the content
        deltas, in line with the M5 SSE parser contract.
        """
        payload = self._build_payload(
            model=model, messages=messages, stream=True, **kwargs
        )
        # ``stream`` may have been forwarded via **kwargs already, but ensure
        # it is set so the upstream returns text/event-stream.
        payload["stream"] = True

        client = self._get_client()
        try:
            async with client.stream("POST", self.endpoint, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise UpstreamError(
                        f"upstream returned HTTP {response.status_code}",
                        status_code=response.status_code,
                        body=body.decode("utf-8", errors="replace"),
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # SSE lines start with ``data: ``; tolerate leading whitespace.
                    stripped = line.lstrip()
                    if not stripped.startswith("data:"):
                        # Could be a comment line (``:``) or a non-data
                        # event; ignore silently.
                        continue
                    payload_text = stripped[len("data:") :].lstrip()
                    if payload_text == "[DONE]":
                        # Terminator; stop reading further.
                        break
                    try:
                        chunk = json.loads(payload_text)
                    except json.JSONDecodeError:
                        logger.debug(
                            "OpenRouterAdapter.stream: ignoring non-JSON SSE payload"
                        )
                        continue
                    delta = self._extract_delta(chunk)
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"timeout talking to {self.base_url}"
            ) from exc
        except httpx.ConnectError as exc:
            raise UpstreamUnavailableError(
                f"cannot connect to {self.base_url}"
            ) from exc
        except httpx.RemoteProtocolError as exc:
            raise UpstreamUnavailableError(
                f"upstream {self.base_url} closed the connection"
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamUnavailableError(
                f"transport error talking to {self.base_url}: {exc}"
            ) from exc

    @staticmethod
    def _extract_delta(chunk: dict[str, Any]) -> str:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        delta = first.get("delta") or {}
        return str(delta.get("content") or "")

    @staticmethod
    def _parse_chat_response(data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or []
        first_choice = choices[0] if choices else {}
        message_data = first_choice.get("message") or {}
        message = Message(
            role=str(message_data.get("role", "assistant")),
            content=str(message_data.get("content", "")),
        )
        usage_data = data.get("usage") or {}
        prompt_tokens = int(usage_data.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_data.get("completion_tokens", 0) or 0)
        total_tokens_raw = usage_data.get("total_tokens")
        if total_tokens_raw is None or total_tokens_raw == 0:
            # Mirror the OllamaAdapter behaviour: if the upstream omits
            # ``total_tokens`` (or reports it as 0), fall back to the sum.
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = int(total_tokens_raw)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return ChatResponse(
            id=str(data.get("id", "")),
            model=str(data.get("model", "")),
            message=message,
            usage=usage,
            finish_reason=first_choice.get("finish_reason"),
        )

    @staticmethod
    def _extract_error_message(body: str) -> str | None:
        if not body:
            return None
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return body.strip() or None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                msg = error.get("message")
                if msg:
                    return str(msg)
            msg = payload.get("message")
            if msg:
                return str(msg)
        return body.strip() or None

    def __repr__(self) -> str:
        """Return a safe representation that redacts the API key.

        The key is shown as ``<first 6 chars>...`` (e.g. ``sk-or-...``)
        or as ``<redacted>`` when shorter than 6 characters. The
        ``base_url``, configured headers, and ``transforms`` are
        included for debugging.
        """
        return (
            f"OpenRouterAdapter(base_url={self.base_url!r}, "
            f"http_referer={self.http_referer!r}, "
            f"x_title={self.x_title!r}, "
            f"transforms={self.transforms!r}, "
            f"timeout={self.timeout!s}, "
            f"api_key={self._redact_key(self._api_key)!r})"
        )

    @staticmethod
    def _redact_key(key: str) -> str:
        if not key:
            return "<redacted>"
        if len(key) <= 6:
            return "<redacted>"
        return f"{key[:6]}..."


__all__ = [
    "API_KEY_ENV_VAR",
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "OpenRouterAdapter",
    "OpenRouterConfigError",
]
