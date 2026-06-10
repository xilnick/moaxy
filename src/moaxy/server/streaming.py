"""Server-Sent Events (SSE) encoding for streaming chat completions.

This module is the data-formatting half of M4 streaming. The proxy
emits an OpenAI-compatible ``text/event-stream`` response for
``stream: true`` requests: incremental ``data: {chunk}`` lines during
the initial answer, ``event: revision`` lines carrying the post-
reflection / post-advisor revised text, and a final ``data: [DONE]``
terminator.

The format matches the SSE wire format documented in
https://html.spec.whatwg.org/multipage/server-sent-events.html and
the OpenAI streaming guide (https://platform.openai.com/docs/api-reference/chat/streaming):

* Each event is one or more ``field: value`` lines terminated by a
  blank line (``\\n\\n``).
* ``data:`` lines carry the JSON payload; multiple ``data:`` lines
  in the same event are concatenated with newlines (we use exactly
  one ``data:`` line per event).
* ``event:`` lines (optional) name the event; the default event
  type is ``"message"`` when the field is absent.
* The stream ends with the literal payload ``[DONE]`` on a
  ``data:`` line; clients detect end-of-stream by either receiving
  ``[DONE]`` or by the connection closing.

Helpers in this module are pure functions (no I/O) so they are easy
to unit-test in isolation. The streaming orchestrator composes them
into an :class:`AsyncIterator` that uvicorn's :class:`StreamingResponse`
serialises byte-for-byte.
"""

from __future__ import annotations

import json
import time
from typing import Any

SSE_DONE_PAYLOAD = "[DONE]"
SSE_DONE_EVENT = "data: [DONE]\n\n"
SSE_TERMINATOR_BYTES = SSE_DONE_EVENT.encode("utf-8")
SSE_LINE_SEP = b"\n"
SSE_FIELD_SEP = b": "
SSE_EVENT_SEP = b"\n\n"


def _sse_bytes(field: str, value: str) -> bytes:
    """Encode a single SSE ``field: value`` line without the terminator.

    The caller appends ``\\n\\n`` (encoded as :data:`SSE_EVENT_SEP`)
    to close the event. We do not include the closing blank line
    here so multiple fields can be combined into a single event by
    concatenating the per-field byte chunks.
    """
    # SSE values may not contain a literal newline; OpenAI chunk JSON
    # is always single-line, so this guard is sufficient in practice.
    if "\n" in value or "\r" in value:
        raise ValueError(
            f"SSE field {field!r} contains a newline; SSE values must be single-line"
        )
    return f"{field}: {value}\n".encode()


def format_sse_data(payload: Any) -> bytes:
    """Return the SSE event bytes for a ``data:`` payload.

    Args:
        payload: A JSON-serialisable object (typically a dict) or a
            raw string. Dicts are serialised via :func:`json.dumps`
            with the default ``ensure_ascii=False`` so non-ASCII
            content streams as UTF-8 instead of escaped ``\\uXXXX``
            sequences.

    Returns:
        The bytes ``b"data: <value>\\n\\n"`` ready to be written to
        the streaming response.
    """
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False)
    return _sse_bytes("data", text) + b"\n"



def format_sse_event(event: str, payload: Any) -> bytes:
    """Return the SSE event bytes for a named event with a data payload.

    Used to send custom event types such as ``event: revision`` for
    the post-reflection / post-advisor revised answer.

    Args:
        event: The event name (e.g. ``"revision"``). Must be a
            non-empty string without newlines.
        payload: The data payload (typically a dict).

    Returns:
        The bytes ``b"event: <name>\\ndata: <json>\\n\\n"`` ready
        to be written to the streaming response.
    """
    if not isinstance(event, str) or not event:
        raise ValueError("SSE event name must be a non-empty string")
    if "\n" in event or "\r" in event:
        raise ValueError("SSE event name must not contain newlines")
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False)
    return (
        _sse_bytes("event", event) + _sse_bytes("data", text) + b"\n"
    )


def format_sse_done() -> bytes:
    """Return the SSE terminator bytes for ``data: [DONE]\\n\\n``.

    The literal payload ``[DONE]`` is the OpenAI streaming
    end-of-stream marker; clients (and the validation contract)
    detect end-of-stream when they receive this line.
    """
    return SSE_TERMINATOR_BYTES


def build_chat_completion_chunk(
    *,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-stream",
    created: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-shaped ``chat.completion.chunk`` dict.

    The chunk object mirrors OpenAI's streaming wire format: an
    ``id``, ``object: "chat.completion.chunk"``, ``created`` (unix
    seconds), ``model``, and a single-element ``choices`` array whose
    ``delta`` carries the incremental content. The first chunk
    typically carries ``role: "assistant"`` in ``delta``; subsequent
    chunks carry the content pieces; the final chunk carries an empty
    ``delta`` and the ``finish_reason``.

    The ``delta`` dict always includes a ``content`` key (defaulting
    to an empty string) so the wire shape is consistent across all
    chunks of a completion. OpenAI's own streaming API does the
    same — even the final empty-delta chunk carries ``content: ""``
    so consumers can rely on the key being present.

    Args:
        model: The model name the chunk is attributed to (the
            alias-resolved real name; the response layer may override
            it with the client alias for the first chunk).
        delta: The delta payload. ``None`` is treated as an empty
            dict. Typical values are ``{"role": "assistant"}`` for
            the first chunk, ``{"content": "Hello"}`` for content
            chunks, and ``{}`` for the final chunk.
        finish_reason: The OpenAI ``finish_reason`` string. ``None``
            for intermediate chunks; ``"stop"`` (or ``"length"``,
            ``"tool_calls"``) for the terminal chunk.
        chunk_id: The id shared across all chunks of a single
            completion. OpenAI uses ``chatcmpl-<random>``; the
            default mirrors that.
        created: The unix-seconds ``created`` timestamp. ``None``
            means "use the current time" — the caller may also pass
            a fixed value to keep all chunks of one completion
            timestamped identically.

    Returns:
        A dict matching the OpenAI ``chat.completion.chunk`` shape
        and ready to be serialised by :func:`format_sse_data`.
    """
    delta_dict = dict(delta) if delta else {}
    # Always include ``content`` in the delta (defaulting to empty
    # string) so consumers can rely on the key being present. This
    # matches OpenAI's own streaming wire format.
    delta_dict.setdefault("content", "")
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(created if created is not None else time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta_dict,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_revision_payload(
    *,
    model: str,
    text: str,
    turn: int | None = None,
    chunk_id: str = "chatcmpl-stream",
    created: int | None = None,
) -> dict[str, Any]:
    """Build a revision event payload for ``event: revision``.

    The revision event is the M4 streaming hook for the
    post-reflection / post-advisor revised answer. The client sees
    it as a named SSE event (``event: revision``) whose ``data:``
    field is a JSON dict with the revised text and a few
    identifying fields. The OpenAI streaming protocol does not
    define a ``revision`` event; this is a moaxy extension that
    mirrors the architecture's stated M4 contract.

    Args:
        model: The alias-resolved real model name the revision was
            produced by.
        text: The full revised text. Sent verbatim (not split into
            deltas) because the reflection/advisor stages do not
            currently stream their LLM calls.
        turn: The reflection turn index (0-based) when the event
            comes from a reflection revision. ``None`` when the
            event comes from the advisor stage or any other
            non-turn-indexed revision.
        chunk_id: The id shared with the initial chunks of this
            completion; useful for client-side correlation.
        created: The unix-seconds timestamp for the revision.
            ``None`` means "use the current time".

    Returns:
        A dict ready to be serialised by :func:`format_sse_event`
        with ``event="revision"``.
    """
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.revision",
        "created": int(created if created is not None else time.time()),
        "model": model,
        "text": text,
    }
    if turn is not None:
        payload["turn"] = turn
    return payload


__all__ = [
    "SSE_DONE_EVENT",
    "SSE_DONE_PAYLOAD",
    "SSE_EVENT_SEP",
    "SSE_FIELD_SEP",
    "SSE_LINE_SEP",
    "SSE_TERMINATOR_BYTES",
    "build_chat_completion_chunk",
    "build_revision_payload",
    "format_sse_data",
    "format_sse_done",
    "format_sse_event",
]
