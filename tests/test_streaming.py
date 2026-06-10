"""Tests for the SSE encoding helpers in :mod:`moaxy.server.streaming`.

The streaming module is the data-formatting half of M4 SSE. It
exposes pure functions (no I/O) for encoding individual events and
building the OpenAI-shaped chunk payloads that the streaming
response is composed from. This test file is hermetic: no
FastAPI app, no httpx, no real adapters.

The contract pinned here matches the validation contract entries
"VAL-CROSS-018" and "VAL-CROSS-019" (SSE wire format and OpenAI
chunk shape) at the unit level.
"""

from __future__ import annotations

import json

import pytest

from moaxy.server.streaming import (
    SSE_DONE_PAYLOAD,
    SSE_TERMINATOR_BYTES,
    build_chat_completion_chunk,
    build_revision_payload,
    format_sse_data,
    format_sse_done,
    format_sse_event,
)

# ────────────────────────────────────────────────────────────────────
# format_sse_data
# ────────────────────────────────────────────────────────────────────


class TestFormatSSEData:
    """``format_sse_data`` encodes a ``data:`` SSE event as bytes."""

    def test_encodes_dict_as_json(self):
        payload = {"foo": "bar"}
        result = format_sse_data(payload)
        assert result == b'data: {"foo": "bar"}\n\n'

    def test_encodes_string_as_raw(self):
        """String payloads are written verbatim (no JSON wrapping)."""
        result = format_sse_data("hello world")
        assert result == b"data: hello world\n\n"

    def test_uses_ensure_ascii_false_for_non_ascii(self):
        """Non-ASCII content is preserved as UTF-8 (not \\uXXXX)."""
        payload = {"text": "héllo"}
        result = format_sse_data(payload)
        # The UTF-8 bytes for é are 0xC3 0xA9; if json.dumps used
        # ``ensure_ascii=True`` (the default) it would emit the
        # ASCII fallback ``h\\u00e9llo``.
        assert b"h\\u00e9llo" not in result
        assert "éllo".encode() in result

    def test_terminates_each_event_with_double_newline(self):
        result = format_sse_data({"k": 1})
        assert result.endswith(b"\n\n")

    def test_rejects_newlines_in_string_payload(self):
        """SSE field values must be single-line."""
        with pytest.raises(ValueError, match="contains a newline"):
            format_sse_data("line1\nline2")

    def test_handles_complex_nested_payload(self):
        payload = {
            "id": "chatcmpl-x",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"}}
            ],
        }
        result = format_sse_data(payload)
        # Round-trips through json.loads.
        line = result.split(b"\n")[0]
        assert line.startswith(b"data: ")
        json_bytes = line[len(b"data: ") :]
        decoded = json.loads(json_bytes.decode("utf-8"))
        assert decoded == payload


# ────────────────────────────────────────────────────────────────────
# format_sse_event
# ────────────────────────────────────────────────────────────────────


class TestFormatSSEEvent:
    """``format_sse_event`` encodes a named SSE event with a data payload."""

    def test_encodes_revision_event(self):
        payload = {"text": "revised answer"}
        result = format_sse_event("revision", payload)
        # SSE format: event: name\ndata: json\n\n
        assert result.startswith(b"event: revision\n")
        assert b"\ndata: " in result
        assert result.endswith(b"\n\n")

    def test_event_name_appears_in_correct_field(self):
        result = format_sse_event("ping", {"k": 1})
        lines = result.split(b"\n")
        assert lines[0] == b"event: ping"
        assert lines[1] == b'data: {"k": 1}'
        assert lines[2] == b""

    def test_string_payload(self):
        """String payloads bypass JSON serialisation."""
        result = format_sse_event("heartbeat", "ok")
        assert result == b"event: heartbeat\ndata: ok\n\n"

    def test_rejects_empty_event_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            format_sse_event("", {"k": 1})

    def test_rejects_newline_in_event_name(self):
        with pytest.raises(ValueError, match="must not contain newlines"):
            format_sse_event("two\nlines", {"k": 1})

    def test_rejects_newline_in_string_payload(self):
        with pytest.raises(ValueError, match="contains a newline"):
            format_sse_event("e", "line1\nline2")


# ────────────────────────────────────────────────────────────────────
# format_sse_done
# ────────────────────────────────────────────────────────────────────


class TestFormatSSEDone:
    """``format_sse_done`` returns the canonical end-of-stream bytes."""

    def test_done_payload_is_data_done_double_newline(self):
        assert format_sse_done() == SSE_TERMINATOR_BYTES
        assert SSE_TERMINATOR_BYTES == b"data: [DONE]\n\n"

    def test_done_payload_constant(self):
        assert SSE_DONE_PAYLOAD == "[DONE]"


# ────────────────────────────────────────────────────────────────────
# build_chat_completion_chunk
# ────────────────────────────────────────────────────────────────────


class TestBuildChatCompletionChunk:
    """``build_chat_completion_chunk`` builds an OpenAI-shaped chunk dict."""

    def test_basic_chunk_shape(self):
        chunk = build_chat_completion_chunk(
            model="minimax-m3:cloud",
            delta={"content": "hi"},
        )
        assert chunk["id"].startswith("chatcmpl-")
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "minimax-m3:cloud"
        assert isinstance(chunk["created"], int)
        assert chunk["choices"] == [
            {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
        ]

    def test_first_chunk_includes_role(self):
        chunk = build_chat_completion_chunk(
            model="m",
            delta={"role": "assistant", "content": ""},
        )
        assert chunk["choices"][0]["delta"]["role"] == "assistant"

    def test_finish_reason_is_propagated(self):
        chunk = build_chat_completion_chunk(
            model="m",
            delta={},
            finish_reason="stop",
        )
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_none_delta_becomes_content_empty_dict(self):
        """``None`` delta becomes ``{"content": ""}`` (the canonical empty delta)."""
        chunk = build_chat_completion_chunk(model="m", delta=None)
        assert chunk["choices"][0]["delta"] == {"content": ""}

    def test_custom_id_and_created(self):
        chunk = build_chat_completion_chunk(
            model="m",
            delta={"content": "x"},
            chunk_id="chatcmpl-abc",
            created=1234567890,
        )
        assert chunk["id"] == "chatcmpl-abc"
        assert chunk["created"] == 1234567890

    def test_default_chunk_id_starts_with_chatcmpl(self):
        chunk = build_chat_completion_chunk(model="m", delta={"content": "x"})
        assert chunk["id"].startswith("chatcmpl-")


# ────────────────────────────────────────────────────────────────────
# build_revision_payload
# ────────────────────────────────────────────────────────────────────


class TestBuildRevisionPayload:
    """``build_revision_payload`` builds a revision event payload."""

    def test_basic_revision_shape(self):
        payload = build_revision_payload(model="m", text="revised")
        assert payload["object"] == "chat.completion.revision"
        assert payload["model"] == "m"
        assert payload["text"] == "revised"
        assert "id" in payload
        assert "created" in payload
        # No ``turn`` key by default.
        assert "turn" not in payload

    def test_turn_included_when_provided(self):
        payload = build_revision_payload(model="m", text="t", turn=1)
        assert payload["turn"] == 1

    def test_turn_none_omitted(self):
        payload = build_revision_payload(model="m", text="t", turn=None)
        assert "turn" not in payload

    def test_custom_id_and_created(self):
        payload = build_revision_payload(
            model="m",
            text="t",
            chunk_id="chatcmpl-x",
            created=1700000000,
        )
        assert payload["id"] == "chatcmpl-x"
        assert payload["created"] == 1700000000


# ────────────────────────────────────────────────────────────────────
# Composed event sequence (smoke)
# ────────────────────────────────────────────────────────────────────


class TestComposedSSEStream:
    """The composed event sequence matches the expected wire format."""

    def test_chat_chunk_then_done(self):
        """A minimal stream: one role chunk + one content chunk + DONE."""
        # First chunk: role assignment.
        first = format_sse_data(
            build_chat_completion_chunk(
                model="m",
                delta={"role": "assistant", "content": "He"},
            )
        )
        # Second chunk: content piece.
        second = format_sse_data(
            build_chat_completion_chunk(
                model="m",
                delta={"content": "llo"},
            )
        )
        done = format_sse_done()
        combined = first + second + done
        # Each event ends with ``\n\n``.
        assert combined.count(b"\n\n") == 3
        # The last event is ``data: [DONE]``.
        assert combined.endswith(b"data: [DONE]\n\n")

    def test_revision_event_in_stream(self):
        rev = format_sse_event(
            "revision",
            build_revision_payload(model="m", text="revised"),
        )
        # The event must start with ``event: revision`` and end with
        # a blank line so SSE clients see the event boundary.
        assert rev.startswith(b"event: revision\n")
        assert rev.endswith(b"\n\n")
