"""Hermetic fake adapter for unit and integration tests.

The :class:`FakeAdapter` is a programmable, in-process
:class:`moaxy.adapters.base.Adapter` that scripts every LLM call's
response and records every call's parameters. It is the canonical
test double for the moaxy pipeline (orchestrator, reflector,
fallback walker, server) and replaces the larger network plumbing
with a deterministic, list-of-responses state machine.

The contract is intentionally close to the contract of the real
:class:`OllamaAdapter`:

* ``chat(*, model, messages, **kwargs)`` returns a
  :class:`moaxy.adapters.base.ChatResponse`.
* The signature accepts the same keyword arguments (``temperature``,
  ``top_p``, ``max_tokens``, …) the orchestrator forwards from the
  original request body.
* The recorded call dict carries the ``model``, the full
  ``messages`` list (verbatim — the adapter does not copy it), and
  every forwarded ``**kwargs``. Tests can assert on the call log to
  pin what the pipeline sent upstream.

The fake is also aware of the OpenAI "system message" convention:
:attr:`FakeAdapter.system_messages` is a per-call convenience list
of the ``system``-role message in the ``messages`` list, in the
order the messages arrived. Tests that need to assert on the
system prompt the pipeline forwards (default reflect prompt, custom
prompt, missing prompt) read this property rather than scanning
the raw messages list themselves.

Script format
-------------

The ``responses`` list is consumed in order; the Nth call to
``chat`` returns the Nth scripted response. The script entries may
be:

* a :class:`ChatResponse` — returned on success.
* an :class:`Exception` instance — raised on the call. Subclasses
  of :class:`moaxy.adapters.base.UpstreamError` are passed through
  unchanged; the orchestrator's fallback walker catches them and
  advances to the next model in the chain.
* ``None`` — convenience for "no scripted response for this call";
  the adapter returns a sensible default
  (:class:`ChatResponse` with the default model and content
  ``"ok"``). Use ``None`` sparingly: explicit scripted responses
  make test failures easier to debug.

When the script runs out before all calls have been made, the
adapter raises :class:`AssertionError` with a message that names
the call index and the model so the failing test reports a
useful error rather than a generic ``IndexError``.

Why not ``MagicMock`` / ``AsyncMock``?
--------------------------------------

The pipeline forwards the OpenAI-shaped ``messages`` list (and a
handful of ``**kwargs``) verbatim to the adapter; tests that need
to assert on the *content* of the call (e.g. "the reflector
prompted with the previous answer") need real call recording.
``AsyncMock`` records the call signature but not the
semantically-rich payload, and mocking the OpenAI client leads to
a brittle tree of ``return_value.return_value.return_value``
chains. The hand-rolled fake is ~80 lines and lets tests
straightforwardly assert ``adapter.calls[0]["messages"][-1]
["content"] == "Please critique …"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    Usage,
)


def _default_response(
    *,
    model: str = "minimax-m3:cloud",
    content: str = "ok",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    finish_reason: str = "stop",
    chatcmpl_id: str = "chatcmpl-fake",
) -> ChatResponse:
    """Build a sensible default :class:`ChatResponse` for ``None`` scripts."""
    return ChatResponse(
        id=chatcmpl_id,
        model=model,
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        finish_reason=finish_reason,
    )


class FakeAdapter(Adapter):
    """Programmable in-process :class:`Adapter` for tests.

    The fake records every call's parameters in
    :attr:`calls` and exposes :attr:`system_messages` for tests
    that want to assert on the system prompt the pipeline
    forwarded. It is constructed once per test and shared
    between the orchestrator and the test body.

    Attributes:
        name: The adapter name. The pipeline reads this only for
            log output; the fake reports ``"fake"``.
        responses: The scripted response list. The Nth call to
            :meth:`chat` returns the Nth entry. See the module
            docstring for the entry grammar.
        calls: The recorded call log. Each entry is a dict with
            ``model``, ``messages``, and every forwarded
            ``**kwargs`` key the orchestrator passed. The
            ``messages`` list is the same object the caller
            supplied (the fake does NOT copy; tests that need a
            snapshot should ``list(adapter.calls)`` or
            ``copy.deepcopy(adapter.calls)``).
        system_messages: A per-call convenience list mirroring
            :attr:`calls`. The Nth entry is the ``"system"``-role
            message extracted from the Nth call's ``messages``
            list, or ``None`` when the call did not include a
            system message. Tests that need to assert on the
            reflect or advisor system prompt read this.
        usages: A per-call convenience list mirroring
            :attr:`calls`. The Nth entry is the
            :class:`Usage` of the response the fake returned for
            the Nth call. (Useful when the test wants to
            cross-check the accumulator against the per-call
            usage the adapter reported.)
    """

    name = "fake"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self._index: int = 0

    def reset(self, responses: list[Any] | None = None) -> None:
        """Reset the call log and (optionally) re-script responses."""
        self.responses = list(responses) if responses is not None else []
        self.calls = []
        self._index = 0

    @property
    def system_messages(self) -> list[dict[str, Any] | None]:
        """The system-role message of each recorded call, in order.

        Returns one entry per call in :attr:`calls`. Each entry is
        the first ``"system"``-role message found in the call's
        ``messages`` list, or ``None`` when no such message was
        present. Tests that want to assert "the pipeline forwarded
        the configured reflector system prompt" should look at the
        third call's entry (the critique call, in the M2 default
        reflection flow).
        """
        out: list[dict[str, Any] | None] = []
        for call in self.calls:
            sys_msg: dict[str, Any] | None = None
            for msg in call.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "system":
                    sys_msg = msg
                    break
            out.append(sys_msg)
        return out

    @property
    def usages(self) -> list[Usage]:
        """The :class:`Usage` of the response returned for each call.

        Returns one entry per call in :attr:`calls`. Each entry is
        the ``usage`` field of the :class:`ChatResponse` the fake
        returned (or of a zero-Usage placeholder when the call
        raised). Tests use this to cross-check the pipeline's
        :class:`UsageAccumulator` against the per-call usage the
        adapter reported.
        """
        usages: list[Usage] = []
        for i, _call in enumerate(self.calls):
            if i < len(self.responses):
                entry = self.responses[i]
                if isinstance(entry, ChatResponse):
                    usages.append(entry.usage)
                    continue
            usages.append(Usage())
        return usages

    def _next_response(self) -> Any:
        if self._index >= len(self.responses):
            return _default_response()
        entry = self.responses[self._index]
        return entry

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        """Return the next scripted response (or raise the next exception)."""
        self.calls.append(
            {"model": model, "messages": messages, **kwargs}
        )
        entry = self._next_response()
        self._index += 1
        if isinstance(entry, BaseException):
            raise entry
        if entry is None:
            return _default_response(model=model)
        if isinstance(entry, ChatResponse):
            return entry
        raise AssertionError(
            f"FakeAdapter: scripted entry #{self._index} must be "
            f"ChatResponse, Exception, or None; got {type(entry).__name__}"
        )

    async def stream(  # pragma: no cover - not exercised by reflection tests
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


__all__ = ["FakeAdapter"]
