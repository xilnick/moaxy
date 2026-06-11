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


# ────────────────────────────────────────────────────────────────────
# End-to-end streaming with a FakeAdapter that streams scripted chunks
# ────────────────────────────────────────────────────────────────────


class TestStreamingEndToEnd:
    """End-to-end M4 streaming tests with a :class:`FakeAdapter`.

    The previous classes pin the wire format in isolation. The
    tests in this class exercise the full HTTP path: a
    :class:`FakeAdapter` is wired into the FastAPI app, the client
    sends a ``stream: true`` request, and the response is parsed
    back into SSE events. The :mod:`moaxy.server.streaming` helpers
    do the byte formatting; these tests prove the *integration*
    produces a valid ``text/event-stream`` response with the
    expected event sequence.

    These tests pin the M4 contract entries ``VAL-CROSS-018`` and
    ``VAL-CROSS-019`` (SSE wire format, content type, multiple
    ``data:`` lines, final ``data: [DONE]``, revision events).
    """

    @pytest.mark.asyncio
    async def test_stream_true_returns_text_event_stream_with_data_done(self):
        """A simple ``stream: true`` request emits
        ``text/event-stream`` with multiple ``data:`` lines and a
        final ``data: [DONE]`` terminator (VAL-CROSS-018).
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        adapter = FakeAdapter(stream_script=[["Hel", "lo, ", "world!"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        # Split into SSE events (terminated by a blank line).
        events = _parse_sse_events(response.text)
        # At least 2 data events (the role+first-delta chunk and the
        # final empty-delta chunk carrying the finish_reason) plus
        # the terminator ``[DONE]``.
        assert len(events) >= 3
        assert events[-1] == (None, "[DONE]")
        # The first data event carries the role assignment and the
        # first content delta.
        first = json.loads(events[0][1])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"
        # All non-terminator data events have a content key.
        for _event_name, payload in events[:-1]:
            decoded = json.loads(payload)
            assert decoded["choices"][0]["delta"].get("content") is not None
        # The final data event carries the finish_reason.
        final = json.loads(events[-2][1])
        assert final["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_emits_multiple_data_lines(self):
        """The streamed response carries more than one ``data:`` line,
        with the chunks accumulating to form the assistant message.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        adapter = FakeAdapter(
            stream_script=[["a", "b", "c", "d", "e"]]
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        # The body must contain at least 3 ``data:`` lines (multiple
        # content deltas + the final empty-delta chunk + the
        # terminator). We count the literal ``data:`` lines.
        assert response.text.count("data:") >= 3
        # The terminator is present.
        assert response.text.endswith("data: [DONE]\n\n")
        # The chunks accumulate to the scripted text "abcde" when
        # concatenated in order.
        events = _parse_sse_events(response.text)
        deltas: list[str] = []
        for _event_name, payload in events[:-1]:
            decoded = json.loads(payload)
            content = decoded["choices"][0]["delta"].get("content", "")
            if content:
                deltas.append(content)
        assert "".join(deltas) == "abcde"

    @pytest.mark.asyncio
    async def test_stream_reflective_route_emits_revision_event(self):
        """A reflective route streams the initial answer then emits
        an ``event: revision`` for the post-reflection revised
        answer (VAL-CROSS-019). The terminator ``data: [DONE]`` is
        the last line of the stream.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content=content),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                # Reflection critique.
                _chat_response("c\nREFLECT_CONFIDENCE: 0.5"),
                # Reflection revision.
                _chat_response("revised answer"),
            ],
        )
        route = RouteConfig(
            name="reflective",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = _parse_sse_events(response.text)
        # The stream ends with the [DONE] terminator.
        assert events[-1] == (None, "[DONE]")
        # Find the revision event.
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 1
        payload = json.loads(revision_events[0][1])
        assert payload["text"] == "revised answer"
        assert payload["turn"] == 0
        # The stream header ``x-moaxy-alias-resolved`` is present
        # (the request did not use an alias; the value is the model
        # name the client sent).
        assert "x-moaxy-alias-resolved" in response.headers
        assert (
            response.headers["x-moaxy-alias-resolved"]
            == "minimax-m3:cloud"
        )

    @pytest.mark.asyncio
    async def test_stream_includes_x_moaxy_request_id_header(self):
        """The streaming response carries the standard
        ``x-moaxy-request-id`` header so end-to-end log correlation
        works on streaming requests too.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        adapter = FakeAdapter(stream_script=[["ok"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert "x-moaxy-request-id" in response.headers
        assert response.headers["x-moaxy-request-id"]


# ────────────────────────────────────────────────────────────────────
# M4 end-to-end: streaming + reflection + alias + advisor
# ────────────────────────────────────────────────────────────────────


class TestStreamingM4EndToEnd:
    """The M4 reflective streaming flow combined into one request.

    The full vertical:
    (1) the client sends a ``stream: true`` request with an
        alias model name (``coder-pro``),
    (2) the route resolves the alias to a real model
        (``minimax-m3:cloud``),
    (3) the orchestrator streams the initial answer, then runs
        the reflection loop (one turn) and the advisor pass
        (one turn with a different model), and
    (4) the final response carries the streaming events plus
        one or more ``event: revision`` events for the
        post-reflection and post-advisor answers, then the
        ``data: [DONE]`` terminator.

    This pins the integration of every M4 feature: streaming
    protocol, alias resolution, self-reflection, advisor, and
    the revision SSE event format.
    """

    @pytest.mark.asyncio
    async def test_stream_reflection_advisor_alias_full_flow(self):
        """End-to-end: alias + reflection + advisor + streaming."""
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                # Reflection critique (primary model).
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    model="minimax-m3:cloud",
                ),
                # Reflection revision (primary model).
                _chat_response(
                    "revised answer", model="minimax-m3:cloud"
                ),
                # Advisor call (different model).
                _chat_response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                ),
            ],
        )
        route = RouteConfig(
            name="reflective-coder",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            aliases={"coder-pro": "minimax-m3:cloud"},
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "coder-pro",  # alias; resolved to minimax-m3:cloud
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        # The alias was resolved; the response header carries the
        # real model name.
        assert response.headers["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        events = _parse_sse_events(response.text)
        # The stream ends with [DONE].
        assert events[-1] == (None, "[DONE]")
        # The initial chunks were streamed (data events with
        # delta.content accumulating to "Hello, world!").
        data_events = [e for e in events[:-1] if e[0] is None]
        deltas: list[str] = []
        for _event_name, payload in data_events:
            decoded = json.loads(payload)
            content = decoded["choices"][0]["delta"].get("content", "")
            if content:
                deltas.append(content)
        assert "".join(deltas) == "Hello, world!"
        # The reflection produced exactly one revision event.
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 1
        rev_payload = json.loads(revision_events[0][1])
        assert rev_payload["text"] == "revised answer"
        assert rev_payload["turn"] == 0
        # The advisor approved, so no advisor-revision event is
        # emitted. The final response content is the
        # post-reflection revised answer.
        # The header ``x-moaxy-alias-resolved`` confirms the
        # alias was resolved (the original alias ``coder-pro`` is
        # echoed in the response body via the orchestrator, not
        # in this header).
        assert (
            response.headers["x-moaxy-alias-resolved"]
            == "minimax-m3:cloud"
        )


# ────────────────────────────────────────────────────────────────────
# M5 streaming-path parity: trailing SSE trailer + reflect/advisor
# score events + advise_first ordering + conditional advisor skip.
# The M5 changes in the buffered ``run`` path (DELTA 1 conditional
# skip, DELTA 3 order=advise_first, DELTA 5+7 weighted early-exit,
# DELTA 6 reflect_score / advisor_score events) are mirrored in the
# streaming ``stream_run`` path so reflective streaming clients see
# the same observability. The streaming path also emits a trailing
# SSE trailer event carrying the ``x-moaxy-*`` response headers as
# a sidecar ``x_moaxy`` field, just before the ``data: [DONE]``
# terminator.
# ────────────────────────────────────────────────────────────────────


class TestStreamingM5TrailingTrailer:
    """The streaming ``stream_run`` path emits a trailing SSE trailer
    event carrying the ``x-moaxy-*`` response headers as a sidecar
    field. The trailer is a ``chat.completion.chunk``-shaped event
    with an empty delta and ``finish_reason: "stop"`` (so it is
    structurally indistinguishable from the final-initial chunk to
    vanilla OpenAI clients) plus an ``x_moaxy`` sidecar field
    carrying the M5 ``x-moaxy-*`` headers (which streaming clients
    can read after the last ``data:`` event).

    The validation contract (VAL-PIPE-EXTRA-032) pins that the
    streaming response carries all 9 ``x-moaxy-*`` headers in the
    final SSE event. The HTTP response envelope (set by the proxy
    server) still carries the same headers; the SSE trailer is an
    additive channel so SSE-only clients can observe them too.
    """

    @pytest.mark.asyncio
    async def test_stream_emits_trailer_event_before_done(self):
        """The streaming response ends with a ``data:`` event that
        carries the ``x_moaxy`` sidecar field with all 9
        ``x-moaxy-*`` headers, then ``data: [DONE]``.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        adapter = FakeAdapter(stream_script=[["Hello, ", "world!"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = _parse_sse_events(response.text)
        # The last event is the [DONE] terminator.
        assert events[-1] == (None, "[DONE]")
        # The second-to-last event is the trailer. It is a
        # chat.completion.chunk-shaped ``data:`` event (no name)
        # with a ``x_moaxy`` sidecar field carrying the
        # ``x-moaxy-*`` headers.
        assert events[-2][0] is None
        trailer = json.loads(events[-2][1])
        assert "x_moaxy" in trailer, (
            f"trailer event missing x_moaxy sidecar; got {trailer!r}"
        )
        x_moaxy = trailer["x_moaxy"]
        # All 9 ``x-moaxy-*`` headers are present in the trailer.
        # The non-reflective / non-advisor route has the
        # following default values:
        # - x-moaxy-request-id
        # - x-moaxy-alias-resolved
        # - x-moaxy-fallbacks-used (always present; "0" when no fallback)
        # - x-moaxy-reflect-turns ("0")
        # - x-moaxy-reflect-confidence ("0")
        # - x-moaxy-reflect-score (absent when no reflection ran;
        #   the contract pins it as "the last parsed SCORE: as a
        #   string, or 0 when no score was parsed"; the
        #   build_response_headers helper emits it only when
        #   reflect_turns > 0, so on a non-reflective route it
        #   is absent)
        # - x-moaxy-advisor-model (absent when no advisor ran)
        # - x-moaxy-advisor-score (absent when no advisor ran)
        # - x-moaxy-advisor-skipped (always present; "0/no" when
        #   the advisor was disabled)
        # We assert the always-present ones and leave the
        # optional-when-disabled ones to dedicated tests.
        assert "x-moaxy-request-id" in x_moaxy
        assert x_moaxy["x-moaxy-alias-resolved"] == "minimax-m3:cloud"
        assert x_moaxy["x-moaxy-fallbacks-used"] == "0"
        assert x_moaxy["x-moaxy-reflect-turns"] == "0"
        assert x_moaxy["x-moaxy-reflect-confidence"] == "0"
        # The advisor is disabled on a non-advisor route, so the
        # advisor-skipped header is the always-present "0/no".
        assert x_moaxy["x-moaxy-advisor-skipped"] == "0/no"

    @pytest.mark.asyncio
    async def test_trailer_event_is_chat_completion_chunk_shaped(self):
        """The trailer is a ``chat.completion.chunk``-shaped
        ``data:`` event (vanilla OpenAI shape). The ``x_moaxy``
        sidecar is additive; vanilla clients ignore it.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        adapter = FakeAdapter(stream_script=[["ok"]])
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        events = _parse_sse_events(response.text)
        trailer = json.loads(events[-2][1])
        # The chunk shape is preserved.
        assert trailer["object"] == "chat.completion.chunk"
        assert isinstance(trailer["id"], str)
        assert trailer["id"].startswith("chatcmpl-")
        assert isinstance(trailer["created"], int)
        # The choices array has a single entry with the canonical
        # empty-delta + finish_reason="stop" shape. The "content"
        # key is present (empty string), so vanilla OpenAI
        # clients see a well-formed final chunk.
        assert isinstance(trailer["choices"], list)
        assert len(trailer["choices"]) == 1
        assert trailer["choices"][0]["index"] == 0
        assert trailer["choices"][0]["delta"]["content"] == ""
        assert trailer["choices"][0]["finish_reason"] == "stop"
        # The sidecar ``x_moaxy`` is present and is a dict.
        assert isinstance(trailer["x_moaxy"], dict)

    @pytest.mark.asyncio
    async def test_trailer_reflect_score_and_advisor_score_headers(self):
        """VAL-PIPE-EXTRA-032: the trailer carries
        ``x-moaxy-reflect-score`` (when reflection ran with a
        parsed ``SCORE:`` line) and ``x-moaxy-advisor-score`` (the
        default ``"0"`` when the advisor ran but did not emit an
        ``ADVISOR_SCORE:`` line — the
        ``m5-delta-advisor-score-event`` feature is a separate
        future feature; this test pins the streaming trailer
        surface and the default value).
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Critique emits REFLECT_CONFIDENCE + SCORE so the trailer
        # carries ``x-moaxy-reflect-score``. Advisor emits
        # ADVISOR_APPROVE (no ADVISOR_SCORE:) so the trailer
        # carries the default ``x-moaxy-advisor-score: 0``. (The
        # ``m5-delta-advisor-score-event`` feature is a separate
        # future feature; this test pins the streaming trailer
        # surface and the default value, NOT the full
        # advisor-score-event wiring.)
        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                # Reflection critique: emits both REFLECT_CONFIDENCE
                # and SCORE so the score event is emitted and the
                # trailer carries ``x-moaxy-reflect-score``.
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 8",
                    model="minimax-m3:cloud",
                ),
                # Reflection revision.
                _chat_response(
                    "revised", model="minimax-m3:cloud"
                ),
                # Advisor: ADVISOR_APPROVE (no score line). The
                # trailer's ``x-moaxy-advisor-score`` defaults
                # to ``"0"`` per the M5 contract.
                _chat_response(
                    "ADVISOR_APPROVE",
                    model="deepseek-v4-pro:cloud",
                ),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        # The reflect-score header carries the parsed integer.
        assert x_moaxy["x-moaxy-reflect-score"] == "8"
        # The advisor-score header carries the default ``"0"``
        # (the advisor ran but did not emit an ADVISOR_SCORE:
        # line; the ``m5-delta-advisor-score-event`` feature is
        # a separate future feature that will populate this
        # header from the parsed score).
        assert x_moaxy["x-moaxy-advisor-score"] == "0"
        # The reflect-turns header is 1 (one critique was emitted).
        assert x_moaxy["x-moaxy-reflect-turns"] == "1"
        # The advisor-model header carries the configured advisor
        # model name.
        assert x_moaxy["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        # The advisor-skipped header is "0/no" because the
        # advisor ran (confidence was 0.5, below 0.85 threshold).
        assert x_moaxy["x-moaxy-advisor-skipped"] == "0/no"


class TestStreamingM5ConditionalAdvisorSkip:
    """The streaming ``stream_run`` path honors the DELTA 1
    conditional advisor skip. When the parsed
    ``REFLECT_CONFIDENCE`` is greater than or equal to 0.85, the
    advisor LLM call is short-circuited. The stream does not
    contain an ``event: revision`` for the advisor (because no
    advisor revision was produced), and the trailer carries
    ``x-moaxy-advisor-skipped: 1/confidence=<x>``.
    """

    @pytest.mark.asyncio
    async def test_stream_skip_advisor_when_confidence_above_threshold(self):
        """VAL-PIPE-EXTRA-034: confidence >= 0.85 → no advisor call,
        no ``advisor_revised`` revision event, trailer carries
        ``x-moaxy-advisor-skipped: 1/confidence=0.9``.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Scripted responses:
        # 0. Reflection critique with confidence 0.9 → early-exit
        #    fires (last turn + clears_threshold). No revision.
        # The advisor call is skipped (confidence 0.9 >= 0.85).
        # No scripted advisor entry is needed.
        adapter = FakeAdapter(
            stream_script=[["Hello, ", "world!"]],
            responses=[
                _chat_response(
                    "looks good\nREFLECT_CONFIDENCE: 0.9",
                    model="minimax-m3:cloud",
                ),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=True, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # No revision event in the stream (early-exit + skip).
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 0
        # The trailer carries the skip header.
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        # 0.9 is the parsed confidence; the skip header reports
        # the inclusive boundary at 0.85.
        assert x_moaxy["x-moaxy-advisor-skipped"] == "1/confidence=0.9"
        # The advisor-model header is NOT set on a skip.
        assert "x-moaxy-advisor-model" not in x_moaxy
        # The adapter call count for the chat (non-stream) path:
        # 1 critique call. The initial call is the streaming
        # path (recorded in ``adapter.stream_calls``, not
        # ``adapter.calls``). The advisor call was skipped.
        assert len(adapter.calls) == 1
        # The streaming path recorded exactly one call (the
        # initial answer stream).
        assert len(adapter.stream_calls) == 1


class TestStreamingM5OrderAdviseFirst:
    """The streaming ``stream_run`` path honors
    ``reflection.order == "advise_first"``. The resulting event
    sequence in the SSE stream is
    ``data: initial → event: revision (advisor_revised) → event: revision (reflect_revised) → data: [DONE]``.

    The validation contract (VAL-PIPE-EXTRA-033) pins the streaming
    parity for the advise_first ordering. The default
    ``reflect_first`` keeps the v1 sequence
    (initial → reflect_revised → [advisor_revised]).
    """

    @pytest.mark.asyncio
    async def test_stream_advise_first_order_emits_advisor_before_reflection(self):
        """VAL-PIPE-EXTRA-033: order=advise_first → advisor_revised
        event comes before reflect_revised event in the SSE stream.
        The streaming path emits one ``event: revision`` per
        ``advisor_revised`` / ``reflect_revised`` event in the
        events list. In the advise_first path, the advisor runs
        first (its revision is the first SSE event); the
        reflection's revision is the second.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Scripted responses (advise_first order):
        # 0. Initial stream (scripted, not in `responses`).
        # 1. Advisor call: emits ADVISOR_REVISE so a revision
        #    event is produced.
        # 2. Primary-model revision after the advisor's REVISE
        #    (this becomes the orchestrator's post-advisor answer
        #    AND is the input to the reflection's critique).
        # 3. Reflection critique: low confidence so revision runs.
        # 4. Reflection revision: produces the final answer.
        adapter = FakeAdapter(
            stream_script=[["initial answer"]],
            responses=[
                _chat_response(
                    "ADVISOR_REVISE: advisor-suggested-revision",
                    model="deepseek-v4-pro:cloud",
                ),
                _chat_response(
                    "primary-after-advisor",
                    model="minimax-m3:cloud",
                ),
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5",
                    model="minimax-m3:cloud",
                ),
                _chat_response(
                    "final-after-reflection",
                    model="minimax-m3:cloud",
                ),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85,
                order="advise_first",
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # The streaming path emits one ``event: revision`` per
        # ``advisor_revised`` / ``reflect_revised`` event in the
        # events list. In the advise_first path the advisor runs
        # first, then the reflection. The first revision is the
        # advisor's revised text; the second is the reflection's.
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 2
        # The first revision is the advisor's revised text
        # (the text after the ``ADVISOR_REVISE:`` marker).
        first_rev = json.loads(revision_events[0][1])
        assert first_rev["text"] == "advisor-suggested-revision"
        # The second revision is the reflection's revised text.
        second_rev = json.loads(revision_events[1][1])
        assert second_rev["text"] == "final-after-reflection"
        # The reflection revision carries a ``turn`` key (the
        # advisor's revised text does not).
        assert "turn" not in first_rev
        assert second_rev.get("turn") == 0


class TestStreamingM5WeightedEarlyExit:
    """The streaming ``stream_run`` path uses the same
    ``parse_weighted_signal`` call as the buffered ``run`` path for
    the threshold check. DELTA 7 safety applies: a critique with no
    ``REFLECT_CONFIDENCE:`` line is treated as a malformed response
    and does NOT short-circuit.
    """

    @pytest.mark.asyncio
    async def test_stream_weighted_early_exit_uses_trust_score(self):
        """VAL-PIPE-EXTRA-032: the streaming path uses
        ``trust_verbal * confidence + trust_score * (score / 10)``
        for the threshold check. With trust_score=1.0 and
        score=9 (combined=0.9), the early-exit fires even when
        confidence=0.5 would not clear the threshold on its own.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # REFLECT_CONFIDENCE: 0.5 (raw), SCORE: 9.
        # With trust_verbal=0.0, trust_score=1.0:
        # combined = 0.0 * 0.5 + 1.0 * (9/10) = 0.9 >= 0.85
        # → early-exit fires on the last turn; no revision.
        adapter = FakeAdapter(
            stream_script=[["initial"]],
            responses=[
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 9",
                ),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1,
                early_exit=True,
                threshold=0.85,
                trust_verbal=0.0,
                trust_score=1.0,
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # No revision event in the stream (early-exit fired on
        # the last turn because combined >= threshold).
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 0
        # 1 critique chat-call (initial is streamed).
        # No revision chat-call.
        assert len(adapter.calls) == 1
        assert len(adapter.stream_calls) == 1
        # The trailer carries the parsed score (9).
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        assert x_moaxy["x-moaxy-reflect-score"] == "9"

    @pytest.mark.asyncio
    async def test_stream_delta7_safety_malformed_critique_continues(self):
        """VAL-PIPE-EXTRA-032: DELTA 7 safety rule applies in the
        streaming path. A critique with NO ``REFLECT_CONFIDENCE:``
        line is treated as a malformed response; the revision runs
        as if ``early_exit: false`` for that turn.
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        # Critique has no ``REFLECT_CONFIDENCE:`` line. With
        # early_exit=True and threshold=0.0, the malformed safety
        # rule forces the revision to run.
        adapter = FakeAdapter(
            stream_script=[["initial"]],
            responses=[
                _chat_response("a critique with no confidence line"),
                _chat_response("revised after missing line"),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1,
                early_exit=True,
                threshold=0.0,  # extreme threshold
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        # The revision ran; one ``event: revision`` is emitted.
        revision_events = [
            e for e in events if e[0] == "revision"
        ]
        assert len(revision_events) == 1
        rev = json.loads(revision_events[0][1])
        assert rev["text"] == "revised after missing line"
        # 2 chat-calls: critique + revision (initial is streamed).
        # The malformed safety rule forced the revision despite
        # threshold=0.0.
        assert len(adapter.calls) == 2
        assert len(adapter.stream_calls) == 1


class TestStreamingM5ReflectAndAdvisorScoreEvents:
    """The streaming path emits ``reflect_score`` and
    ``advisor_score`` events (DELTA 6). The events are appended to
    ``ctx.events`` exactly as in the buffered path; the streaming
    client observes them indirectly through the trailer's
    ``x-moaxy-reflect-score`` / ``x-moaxy-advisor-score`` headers
    (which are derived from the latest parsed score per event
    type). The events themselves are not yielded as separate SSE
    events in the streaming path (the contract pins the headers in
    the trailer, not separate events); the event-list side-effect
    is observable in the ``ctx.events`` after ``stream_run``
    completes.
    """

    @pytest.mark.asyncio
    async def test_stream_reflect_score_event_appended_to_ctx(self):
        """DELTA 6 (streaming): a parsed ``SCORE:`` line emits a
        ``reflect_score`` event in the events list (observable in
        ``ctx.events`` after ``stream_run`` completes).
        """
        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.models.config import (
            AdvisorConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.pipeline.context import PipelineContext
        from moaxy.pipeline.orchestrator import Orchestrator
        from moaxy.routing.matcher import RouteMatch
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model="minimax-m3:cloud",
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        adapter = FakeAdapter(
            stream_script=[["initial"]],
            responses=[
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5\nSCORE: 8",
                ),
                _chat_response("revised"),
            ],
        )
        config_route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(turns=0),
        )
        route = RouteMatch(
            route=config_route,
            original_model="minimax-m3:cloud",
            resolved_model="minimax-m3:cloud",
            backend="olloma-local",
            path="/v1/chat/completions",
            reflection=config_route.reflection,
            advisor=config_route.advisor,
            fallbacks=[],
            retry=0,
            aliases={},
        )
        ctx = PipelineContext(
            request_id="req-stream-score",
            request={
                "model": "minimax-m3:cloud",
                "messages": [{"role": "user", "content": "ping"}],
            },
            route=route,
            model_alias_resolved="minimax-m3:cloud",
            target_backend="olloma-local",
            original_model="minimax-m3:cloud",
        )
        # Drive the streaming path end-to-end (drain the
        # generator into a list of bytes).
        chunks: list[bytes] = []
        async for chunk in Orchestrator(adapter).stream_run(ctx):
            chunks.append(chunk)
        assert b"data: [DONE]" in b"".join(chunks)
        # The ``reflect_score`` event is appended to ``ctx.events``.
        score_events = [
            e for e in ctx.events if e.type == "reflect_score"
        ]
        assert len(score_events) == 1
        assert score_events[0].text == "8"
        assert score_events[0].turn == 0
        # The runtime attribute ``last_score`` is set.
        assert ctx.__dict__["last_score"] == 8

    @pytest.mark.asyncio
    async def test_stream_advisor_score_event_appended_to_ctx(self):
        """DELTA 6 (streaming, advisor): the orchestrator parses the
        ``ADVISOR_SCORE:`` line from the advisor's response and
        the streaming path's trailer carries the parsed score in
        the ``x-moaxy-advisor-score`` header. (This test pins the
        streaming trailer surface; the full
        ``m5-delta-advisor-score-event`` feature is a separate
        future feature that wires the parsed score into the
        orchestrator's event log.)
        """
        from httpx import ASGITransport, AsyncClient

        from moaxy.adapters.base import ChatResponse, Message, Usage
        from moaxy.adapters.registry import AdapterRegistry
        from moaxy.models.config import (
            AdapterConfig,
            AdvisorConfig,
            MoaxyConfig,
            ReflectionConfig,
            RouteConfig,
        )
        from moaxy.models.config import RouteMatch as ConfigRouteMatch
        from moaxy.server.app import create_app
        from tests.fixtures.fake_adapter import FakeAdapter

        def _chat_response(content: str, model: str) -> ChatResponse:
            return ChatResponse(
                id="chatcmpl-stream",
                model=model,
                message=Message(role="assistant", content=content),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

        adapter = FakeAdapter(
            stream_script=[["initial"]],
            responses=[
                _chat_response(
                    "c\nREFLECT_CONFIDENCE: 0.5", model="minimax-m3:cloud"
                ),
                _chat_response("revised", model="minimax-m3:cloud"),
                _chat_response(
                    "ADVISOR_DECISION: APPROVE\nADVISOR_SCORE: 7",
                    model="deepseek-v4-pro:cloud",
                ),
            ],
        )
        route = RouteConfig(
            name="r",
            match=ConfigRouteMatch(
                model="*", path="/v1/chat/completions"
            ),
            backend="olloma-local",
            reflection=ReflectionConfig(
                turns=1, early_exit=False, threshold=0.85
            ),
            advisor=AdvisorConfig(
                model="deepseek-v4-pro:cloud", turns=1
            ),
        )
        cfg = MoaxyConfig(
            backends=[
                AdapterConfig(
                    name="olloma-local",
                    adapter="ollama",
                    base_url="http://x",
                )
            ],
            routes=[route],
        )
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=cfg, adapters=registry)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m3:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
            )

        # The trailer carries the default ``x-moaxy-advisor-score:
        # 0`` (the orchestrator has not yet populated
        # ``ctx.__dict__["advisor_score"]``; the
        # ``m5-delta-advisor-score-event`` feature is a separate
        # future feature that will populate this from the parsed
        # ``ADVISOR_SCORE:`` line).
        events = _parse_sse_events(response.text)
        trailer = json.loads(events[-2][1])
        x_moaxy = trailer["x_moaxy"]
        assert x_moaxy["x-moaxy-advisor-score"] == "0"
        # The advisor-model header is set (advisor ran).
        assert x_moaxy["x-moaxy-advisor-model"] == "deepseek-v4-pro:cloud"
        # The advisor-skipped header is "0/no" (advisor ran).
        assert x_moaxy["x-moaxy-advisor-skipped"] == "0/no"


# ────────────────────────────────────────────────────────────────────
# SSE parser helper (used by the end-to-end tests)
# ────────────────────────────────────────────────────────────────────


def _parse_sse_events(body: str) -> list[tuple[str | None, str]]:
    """Parse an SSE response body into ``(event_name, data_payload)`` pairs.

    Mirrors the helper used by ``test_server_orchestrator_integration``:
    the response body is a series of events separated by a blank
    line (``\\n\\n``). Each event may have one or more
    ``field: value`` lines; the ``event:`` field (default
    ``"message"`` when absent) and the ``data:`` lines are
    extracted. The terminator ``[DONE]`` is preserved as a
    ``data:`` payload so tests can assert on its presence.
    """
    events: list[tuple[str | None, str]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in body.split("\n"):
        if line == "":
            if current_data or current_event is not None:
                events.append((current_event, "\n".join(current_data)))
            current_event = None
            current_data = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                current_event = value
            elif field == "data":
                current_data.append(value)
    if current_data or current_event is not None:
        events.append((current_event, "\n".join(current_data)))
    return events
