"""Tests for :mod:`moaxy.pipeline.fallback`.

The :func:`call_with_fallbacks` walker is the M2 fallback policy. It
walks a model list, retrying each model on transient errors (5xx,
timeouts, connection errors), then advancing to the next. When the
list is exhausted it raises :class:`UpstreamExhaustedError`. The
tests below pin every property the validation contract asserts and
every edge case the orchestrator will rely on.

The tests use a hand-rolled :class:`ScriptedAdapter` rather than the
real :class:`OllamaAdapter`. The walker only depends on
``adapter.chat(model=..., messages=..., **kwargs) -> ChatResponse``;
the scripted adapter records every call and returns either a
scripted response or a scripted exception, giving the tests
hermetic, deterministic control over the adapter's behaviour.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    Message,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    Usage,
)
from moaxy.pipeline.fallback import (
    UpstreamExhaustedError,
    call_with_fallbacks,
)

# ────────────────────────────────────────────────────────────────────
# ScriptedAdapter
# ────────────────────────────────────────────────────────────────────


class ScriptedAdapter(Adapter):
    """An :class:`Adapter` whose ``chat`` is driven by a script.

    The script is a list of either:

    * a :class:`ChatResponse` to return on success, or
    * an :class:`Exception` instance to raise.

    Calls are matched to script entries in order. If the script runs
    out, every subsequent ``chat`` raises an :class:`AssertionError`
    so the test fails loudly with a clear message.

    The class records every call (model + kwargs) in ``self.calls``,
    indexed by the order in which ``chat`` was invoked.
    """

    name = "scripted"

    def __init__(self, script: list[Any]) -> None:
        self._script: list[Any] = list(script)
        self._index: int = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self._index >= len(self._script):
            raise AssertionError(
                f"ScriptedAdapter: no more scripted responses "
                f"(call #{self._index + 1} for model={model})"
            )
        entry = self._script[self._index]
        self._index += 1
        if isinstance(entry, BaseException):
            raise entry
        if not isinstance(entry, ChatResponse):
            raise AssertionError(
                f"ScriptedAdapter: script entry must be ChatResponse or "
                f"Exception, got {type(entry).__name__}"
            )
        return entry

    async def stream(  # pragma: no cover - not exercised by fallback tests
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        if False:
            yield ""

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


def _ok(content: str = "ok", model: str = "m1", **usage_kwargs: int) -> ChatResponse:
    """Build a successful :class:`ChatResponse` with optional usage knobs."""
    base_usage = {
        "prompt_tokens": usage_kwargs.get("prompt_tokens", 1),
        "completion_tokens": usage_kwargs.get("completion_tokens", 2),
        "total_tokens": usage_kwargs.get(
            "total_tokens",
            usage_kwargs.get("prompt_tokens", 1) + usage_kwargs.get("completion_tokens", 2),
        ),
    }
    return ChatResponse(
        id="chatcmpl-test",
        model=model,
        message=Message(role="assistant", content=content),
        usage=Usage(**base_usage),
        finish_reason="stop",
    )


def _5xx(model: str | None = None, message: str = "upstream 500") -> UpstreamError:
    return UpstreamError(message, status_code=500, body=message)


def _4xx(model: str | None = None, message: str = "upstream 400") -> UpstreamError:
    return UpstreamError(message, status_code=400, body=message)


def _timeout(message: str = "upstream timeout") -> UpstreamTimeoutError:
    return UpstreamTimeoutError(message)


def _unavailable(message: str = "upstream unavailable") -> UpstreamUnavailableError:
    return UpstreamUnavailableError(message)


# ────────────────────────────────────────────────────────────────────
# Function shape and export
# ────────────────────────────────────────────────────────────────────


class TestFallbackModuleExports:
    """The :mod:`moaxy.pipeline.fallback` module exports the documented names."""

    def test_call_with_fallbacks_is_callable(self):
        assert callable(call_with_fallbacks)

    def test_call_with_fallbacks_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(call_with_fallbacks)

    def test_upstream_exhausted_error_is_subclass_of_upstream_error(self):
        assert issubclass(UpstreamExhaustedError, UpstreamError)

    def test_upstream_exhausted_error_attributes(self):
        e = UpstreamExhaustedError("msg", models=["a", "b"], last_error=ValueError("x"))
        assert e.models == ["a", "b"]
        assert isinstance(e.last_error, ValueError)
        assert e.status_code is None
        assert e.body is None

    def test_pipeline_package_re_exports_fallback(self):
        from moaxy.pipeline import (
            UpstreamExhaustedError as PipelineUpstreamExhaustedError,
        )
        from moaxy.pipeline import (
            call_with_fallbacks as PipelineCallWithFallbacks,
        )

        assert PipelineUpstreamExhaustedError is UpstreamExhaustedError
        assert PipelineCallWithFallbacks is call_with_fallbacks


# ────────────────────────────────────────────────────────────────────
# Happy path
# ────────────────────────────────────────────────────────────────────


class TestHappyPath:
    """The walker returns the primary's response on a clean first try."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_first_try(self):
        adapter = ScriptedAdapter([_ok("hi", model="minimax-m3:cloud")])
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["minimax-m3:cloud"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "hi"
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_fallbacks_used_is_empty_when_primary_succeeds(self):
        adapter = ScriptedAdapter([_ok("hi", model="m1")])
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2", "m3"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_adapter(self):
        """Sampling parameters are forwarded verbatim to ``adapter.chat``."""
        adapter = ScriptedAdapter([_ok("hi")])
        await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
            stop=["END"],
        )
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.9
        assert call["max_tokens"] == 128
        assert call["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_model_kwarg_comes_from_list_not_kwargs(self):
        """The walker injects the current model from ``models``; callers
        must NOT pass ``model=`` in kwargs (doing so would shadow the
        per-iteration value and cause silent bugs)."""
        adapter = ScriptedAdapter([_ok("ok", model="primary")])
        await call_with_fallbacks(
            adapter,
            models=["primary"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        # The model recorded on the adapter call came from the list.
        assert adapter.calls[0]["model"] == "primary"

    @pytest.mark.asyncio
    async def test_model_kwarg_overridden_per_iteration(self):
        """Each entry in ``models`` is used as the ``model`` kwarg for
        its iteration. To observe the fallback, the primary must
        fail (a 5xx); the walker then advances and the adapter
        records the second iteration's model name."""
        adapter = ScriptedAdapter([_5xx(), _ok("ok from fallback")])
        await call_with_fallbacks(
            adapter,
            models=["primary", "fallback"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert [c["model"] for c in adapter.calls] == ["primary", "fallback"]

    @pytest.mark.asyncio
    async def test_model_in_kwargs_is_rejected_with_typeerror(self):
        """A caller that mistakenly passes ``model=`` in ``kwargs``
        (in addition to the ``models`` list) is rejected with a
        clear ``TypeError`` at call time. The walker manages the
        model name from the list; an explicit ``model=`` in kwargs
        would be ambiguous (the per-iteration value would shadow
        it, but silently — and the call would also raise a
        confusing ``TypeError`` deep in the adapter). The walker
        raises the TypeError up front with a clear message."""
        adapter = ScriptedAdapter([])
        with pytest.raises(TypeError, match="does not accept a 'model' kwarg"):
            await call_with_fallbacks(
                adapter,
                models=["primary"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
                model="WRONG",
            )
        # The adapter was never called.
        assert adapter.calls == []


# ────────────────────────────────────────────────────────────────────
# Retry on transient errors
# ────────────────────────────────────────────────────────────────────


class TestRetryOnTransient:
    """5xx, timeout, and connection errors trigger a retry on the same model."""

    @pytest.mark.asyncio
    async def test_5xx_retries_then_succeeds(self):
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _ok("recovered"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "recovered"
        assert fallbacks_used == []
        assert len(adapter.calls) == 3

    @pytest.mark.asyncio
    async def test_timeout_retries_then_succeeds(self):
        adapter = ScriptedAdapter(
            [
                _timeout(),
                _ok("after timeout"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "after timeout"
        assert fallbacks_used == []
        assert len(adapter.calls) == 2

    @pytest.mark.asyncio
    async def test_unavailable_retries_then_succeeds(self):
        adapter = ScriptedAdapter(
            [
                _unavailable(),
                _ok("after unavailable"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "after unavailable"
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_zero_retry_means_one_attempt(self):
        """``retry=0`` means the walker tries the model exactly once
        and advances (or gives up) on the first transient failure."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _ok("from fallback", model="m2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from fallback"
        assert fallbacks_used == ["m2"]
        assert len(adapter.calls) == 2

    @pytest.mark.asyncio
    async def test_negative_retry_treated_as_zero(self):
        """A negative ``retry`` is clamped to zero; the walker still
        tries each model once."""
        adapter = ScriptedAdapter(
            [
                _ok("ok"),
            ]
        )
        response, _ = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=-3,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "ok"
        assert len(adapter.calls) == 1

    @pytest.mark.asyncio
    async def test_retry_budget_exhausted_then_fallback(self):
        """The walker makes exactly ``retry + 1`` attempts on a
        model, then advances to the next."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _5xx(),
                _ok("from fallback", model="m2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        # m1 was tried 3 times (1 initial + 2 retries) then m2 succeeded.
        assert response.message.content == "from fallback"
        assert fallbacks_used == ["m2"]
        assert [c["model"] for c in adapter.calls] == ["m1", "m1", "m1", "m2"]

    @pytest.mark.asyncio
    async def test_retry_mixed_transient_types(self):
        """5xx, timeout, and unavailable errors are all transient and
        should each consume one attempt of the retry budget."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _timeout(),
                _unavailable(),
                _ok("after mixed failures"),
            ]
        )
        response, _ = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "after mixed failures"
        assert len(adapter.calls) == 4

    @pytest.mark.asyncio
    async def test_first_failure_then_primary_succeeds_on_retry(self):
        """A primary that fails once but succeeds on its second
        attempt yields ``fallbacks_used = []``: the primary itself
        is never in the list."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _ok("primary recovered"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "fallback"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "primary recovered"
        assert fallbacks_used == []


# ────────────────────────────────────────────────────────────────────
# Permanent (4xx) errors
# ────────────────────────────────────────────────────────────────────


class TestPermanentErrors:
    """4xx errors raise immediately without retry and without consulting fallbacks."""

    @pytest.mark.asyncio
    async def test_4xx_raises_immediately(self):
        adapter = ScriptedAdapter([_4xx(message="bad request")])
        with pytest.raises(UpstreamError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1"],
                retry=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.status_code == 400
        assert "bad request" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_4xx_does_not_consult_fallbacks(self):
        """A 4xx on the primary bubbles up; the fallback list is
        ignored because the request itself is malformed."""
        adapter = ScriptedAdapter(
            [
                _4xx(),
                _ok("should never be called", model="m2"),
            ]
        )
        with pytest.raises(UpstreamError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2"],
                retry=2,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.status_code == 400
        # Only the primary was called; the fallback was never invoked.
        assert [c["model"] for c in adapter.calls] == ["m1"]

    @pytest.mark.asyncio
    async def test_4xx_on_fallback_bubbles_up(self):
        """A 4xx on a fallback model (not the primary) also bubbles
        up immediately. The walker does not try the next model in
        the chain, because the request itself is the problem."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _4xx(),
                _ok("should never be called", model="m3"),
            ]
        )
        with pytest.raises(UpstreamError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2", "m3"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.status_code == 400
        # m1 (1 attempt), m2 (1 attempt), m3 never called.
        assert [c["model"] for c in adapter.calls] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_4xx_does_not_count_against_retry_budget(self):
        """A 4xx raises on the very first attempt; the ``retry``
        budget is irrelevant."""
        adapter = ScriptedAdapter([_4xx()])
        with pytest.raises(UpstreamError):
            await call_with_fallbacks(
                adapter,
                models=["m1"],
                retry=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        # Exactly one call, not retry + 1.
        assert len(adapter.calls) == 1


# ────────────────────────────────────────────────────────────────────
# Fallback chain walking
# ────────────────────────────────────────────────────────────────────


class TestFallbackChain:
    """The walker advances through the model list, retrying each."""

    @pytest.mark.asyncio
    async def test_primary_fails_uses_first_fallback(self):
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _ok("from fallback 1", model="f1"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from fallback 1"
        assert fallbacks_used == ["f1"]

    @pytest.mark.asyncio
    async def test_walks_full_chain_to_third_model(self):
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _ok("from fallback 2", model="f2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1", "f2"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from fallback 2"
        assert fallbacks_used == ["f1", "f2"]
        assert [c["model"] for c in adapter.calls] == ["primary", "f1", "f2"]

    @pytest.mark.asyncio
    async def test_each_fallback_is_independently_retried(self):
        """A transient failure on a fallback consumes its own retry
        budget; it does not bleed into the next model.

        Script: primary fails twice → f1 is tried (retry=1) twice,
        both fail → f2 is tried once and succeeds. ``fallbacks_used``
        records both f1 and f2 because both were invoked as
        fallbacks (f1 failed and the walker advanced, f2 succeeded)."""
        adapter = ScriptedAdapter(
            [
                _5xx(),  # primary attempt 1
                _5xx(),  # primary attempt 2
                _5xx(),  # f1 attempt 1
                _5xx(),  # f1 attempt 2
                _ok("from f2"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1", "f2"],
            retry=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from f2"
        assert fallbacks_used == ["f1", "f2"]
        assert [c["model"] for c in adapter.calls] == [
            "primary",
            "primary",
            "f1",
            "f1",
            "f2",
        ]

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_unchanged_to_every_call(self):
        """Sampling parameters are forwarded to every adapter call,
        not just the first. The walker doesn't transform kwargs."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _ok("ok"),
            ]
        )
        await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.42,
            top_p=0.88,
        )
        for call in adapter.calls:
            assert call["temperature"] == 0.42
            assert call["top_p"] == 0.88


# ────────────────────────────────────────────────────────────────────
# Exhaustion
# ────────────────────────────────────────────────────────────────────


class TestExhaustion:
    """When every model in the list has failed, raise UpstreamExhaustedError."""

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_upstream_exhausted(self):
        adapter = ScriptedAdapter([_5xx()] * 6)
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2", "m3"],
                retry=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "all backends failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_exhausted_error_carries_models(self):
        """The raised exception records the full chain that was tried."""
        adapter = ScriptedAdapter([_5xx()] * 9)
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2", "m3"],
                retry=2,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.models == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_exhausted_error_carries_last_error(self):
        """The raised exception records the final transient error
        raised by the last attempt; the caller can inspect it."""
        last_exc = _5xx(message="final straw")
        adapter = ScriptedAdapter([_5xx(message="first"), last_exc])
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.last_error is last_exc

    @pytest.mark.asyncio
    async def test_exhausted_is_subclass_of_upstream_error(self):
        """UpstreamExhaustedError is an UpstreamError so callers that
        already handle the upstream-error family can catch it
        generically."""
        adapter = ScriptedAdapter([_5xx()])
        with pytest.raises(UpstreamError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert isinstance(excinfo.value, UpstreamExhaustedError)

    @pytest.mark.asyncio
    async def test_empty_model_list_raises_exhausted(self):
        """An empty model list raises UpstreamExhaustedError with
        the 'all backends failed' substring, even though no
        adapter call was attempted."""
        adapter = ScriptedAdapter([])
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=[],
                retry=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "all backends failed" in str(excinfo.value)
        assert excinfo.value.models == []
        assert excinfo.value.last_error is None
        # The adapter was never called.
        assert adapter.calls == []

    @pytest.mark.asyncio
    async def test_exhausted_message_contains_all_backends_failed(self):
        """The validation contract (VAL-RT-013) pins the substring
        'all backends failed' on the exhaustion error message."""
        adapter = ScriptedAdapter([_timeout()])
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1"],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "all backends failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_exhausted_with_mixed_transient_failures(self):
        """The walker exhausts on any mix of transient failures
        (5xx + timeout + unavailable), in any order."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _timeout(),
                _unavailable(),
                _5xx(),
                _5xx(),
                _5xx(),
            ]
        )
        with pytest.raises(UpstreamExhaustedError) as excinfo:
            await call_with_fallbacks(
                adapter,
                models=["m1", "m2"],
                retry=2,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.models == ["m1", "m2"]


# ────────────────────────────────────────────────────────────────────
# fallbacks_used tracking
# ────────────────────────────────────────────────────────────────────


class TestFallbacksUsedTracking:
    """The returned list is correct under every scenario."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_with_no_fallbacks_used(self):
        adapter = ScriptedAdapter([_ok("ok")])
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "m2", "m3"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_primary_succeeds_after_retry_no_fallbacks_used(self):
        """The primary itself is not in fallbacks_used, even when it
        succeeded only after one or more retries. Only models called
        AFTER the primary failed appear."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _ok("primary recovered"),
            ]
        )
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1", "f2"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_fallbacks_used_records_actually_invoked_fallbacks(self):
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _5xx(),
                _ok("from f1", model="f1"),
            ]
        )
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1", "f2"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        # Only f1 was actually invoked as a fallback; f2 never ran.
        assert fallbacks_used == ["f1"]

    @pytest.mark.asyncio
    async def test_fallbacks_used_preserves_order(self):
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _5xx(),
                _5xx(),
                _ok("from f2", model="f2"),
            ]
        )
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["primary", "f1", "f2", "f3"],
            retry=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        # f1 was tried (and failed), f2 was tried (and succeeded);
        # f3 was never invoked. Order matches the model list.
        assert fallbacks_used == ["f1", "f2"]

    @pytest.mark.asyncio
    async def test_fallbacks_used_is_a_list(self):
        """The contract types fallbacks_used as ``list[str]``; a
        concrete list is required (not a generator, not None)."""
        adapter = ScriptedAdapter([_ok("ok")])
        _, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(fallbacks_used, list)
        assert all(isinstance(name, str) for name in fallbacks_used)


# ────────────────────────────────────────────────────────────────────
# Interaction with the route / model-defaults override pattern
# ────────────────────────────────────────────────────────────────────


class TestOverrideSemantics:
    """The function takes the resolved list and retry; the override
    resolution itself lives in the orchestrator / routing layer.

    The walker itself is unaware of ``models.fallbacks[model]`` or
    ``route.fallbacks``. The tests below document the contract: the
    caller is responsible for picking the right list and the right
    retry value, and the walker honours whatever it's given.
    """

    @pytest.mark.asyncio
    async def test_route_fallbacks_used_when_caller_passes_them(self):
        """When the caller passes the route-level ``fallbacks`` list
        (e.g. ``["kimi-k2.6:cloud"]``), the walker uses exactly
        that — the model-defaults value is irrelevant to the walker.
        """
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _ok("from kimi", model="kimi-k2.6:cloud"),
            ]
        )
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["minimax-m3:cloud", "kimi-k2.6:cloud"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from kimi"
        assert fallbacks_used == ["kimi-k2.6:cloud"]

    @pytest.mark.asyncio
    async def test_route_retry_used_when_caller_passes_it(self):
        """When the caller passes the route-level ``retry`` value
        (e.g. ``0``), the walker uses exactly that — the
        model-defaults retry is irrelevant."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _ok("from fallback", model="f1"),
            ]
        )
        # Caller passes retry=0 even though the model-default retry[model] is 5.
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["m1", "f1"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "from fallback"
        # m1 was tried exactly once, not 6 times.
        assert [c["model"] for c in adapter.calls] == ["m1", "f1"]
        assert fallbacks_used == ["f1"]


# ────────────────────────────────────────────────────────────────────
# Edge cases
# ────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and unusual input shapes."""

    @pytest.mark.asyncio
    async def test_single_model_no_fallbacks(self):
        """A list with a single model behaves like a normal call:
        the walker tries it, retrying on transient errors, and
        either returns or raises UpstreamExhaustedError."""
        adapter = ScriptedAdapter([_ok("ok")])
        response, fallbacks_used = await call_with_fallbacks(
            adapter,
            models=["only"],
            retry=3,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.message.content == "ok"
        assert fallbacks_used == []

    @pytest.mark.asyncio
    async def test_single_model_fails_all_retries_raises(self):
        adapter = ScriptedAdapter([_5xx()] * 3)
        with pytest.raises(UpstreamExhaustedError):
            await call_with_fallbacks(
                adapter,
                models=["only"],
                retry=2,
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_messages_kwarg_forwarded(self):
        """The ``messages`` kwarg is forwarded verbatim to the adapter."""
        adapter = ScriptedAdapter([_ok("ok")])
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "ping"},
        ]
        await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=0,
            messages=messages,
        )
        assert adapter.calls[0]["messages"] is messages or adapter.calls[0]["messages"] == messages

    @pytest.mark.asyncio
    async def test_response_returned_is_chat_response(self):
        adapter = ScriptedAdapter([_ok("ok")])
        response, _ = await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(response, ChatResponse)
        assert response.message.role == "assistant"
        assert response.message.content == "ok"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 1
        assert response.usage.completion_tokens == 2

    @pytest.mark.asyncio
    async def test_call_count_matches_attempts(self):
        """The total number of adapter calls equals the number of
        attempts actually made (1 + retries on transient, before
        the success or the next model)."""
        adapter = ScriptedAdapter(
            [
                _5xx(),
                _5xx(),
                _ok("ok"),
            ]
        )
        await call_with_fallbacks(
            adapter,
            models=["m1"],
            retry=2,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(adapter.calls) == 3  # 2 failed + 1 success

    @pytest.mark.asyncio
    async def test_no_calls_when_model_list_empty(self):
        adapter = ScriptedAdapter([])
        with pytest.raises(UpstreamExhaustedError):
            await call_with_fallbacks(
                adapter,
                models=[],
                retry=0,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert adapter.calls == []
