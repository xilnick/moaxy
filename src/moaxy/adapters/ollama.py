"""OllamaAdapter — wraps an Ollama (OpenAI-compatible) backend.

The adapter posts to ``${base_url}/v1/chat/completions`` with the standard
OpenAI request body (``model``, ``messages``, plus optional
``max_tokens``/``temperature``/``top_p``/``stop``/``stream`` forwarded via
``**kwargs``). The upstream JSON response is parsed into a normalised
:class:`ChatResponse`.

Failure modes map to typed exceptions:

* :class:`UpstreamError` for 4xx/5xx responses and for non-decodable
  response bodies.
* :class:`UpstreamTimeoutError` for read/connect/pool timeouts.
* :class:`UpstreamUnavailableError` for connection errors and
  remote-protocol errors.

The adapter is intentionally minimal: it does not interpret route-level
configuration, alias tables, or fallback chains. Those concerns live in
:mod:`moaxy.routing.matcher` and :mod:`moaxy.pipeline.fallback` (M2+).
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

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

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 30.0
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class OllamaAdapter(Adapter):
    """Adapter for an Ollama instance speaking the OpenAI chat-completions API.

    Args:
        base_url: Root URL of the Ollama server. Defaults to
            ``http://127.0.0.1:11434``. Trailing slashes are stripped.
        timeout: Per-call timeout in seconds for the underlying
            ``httpx.AsyncClient``. Defaults to 30 seconds.
        api_key: Optional bearer token. Forwarded as ``Authorization`` when
            set; Ollama itself does not require it, but the same field is
            useful for proxying to authenticated OpenAI-compatible backends.
        _transport: Internal/test hook. Lets unit tests inject an in-process
            :class:`httpx.AsyncBaseTransport`. Production code leaves this
            ``None`` so a real network client is used.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._raw_base_url = base_url
        self.base_url = self._normalise_base_url(base_url)
        self.timeout = float(timeout)
        self.api_key = api_key
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
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
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
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)

        client = self._get_client()
        try:
            response = await client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning("OllamaAdapter timeout for model=%s: %s", model, exc)
            raise UpstreamTimeoutError(f"timeout talking to {self.base_url}") from exc
        except httpx.ConnectError as exc:
            logger.warning("OllamaAdapter connect error for model=%s: %s", model, exc)
            raise UpstreamUnavailableError(
                f"cannot connect to {self.base_url}"
            ) from exc
        except httpx.RemoteProtocolError as exc:
            logger.warning(
                "OllamaAdapter remote protocol error for model=%s: %s", model, exc
            )
            raise UpstreamUnavailableError(
                f"upstream {self.base_url} closed the connection"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("OllamaAdapter request error for model=%s: %s", model, exc)
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

        The Ollama OpenAI-compatible endpoint emits newline-delimited JSON
        (``application/x-ndjson``), where each line is a partial
        ``chat.completion.chunk`` object. We pull the ``choices[0].delta.content``
        field out of each chunk and yield it.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        payload.update(kwargs)
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
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("OllamaAdapter.stream: ignoring non-JSON line")
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
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_data.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_data.get("total_tokens", 0) or 0),
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


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "OllamaAdapter",
]
