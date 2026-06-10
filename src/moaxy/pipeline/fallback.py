"""Per-model fallback walker for the moaxy pipeline.

The :func:`call_with_fallbacks` coroutine implements the M2 fallback
policy that the orchestrator runs at every LLM call site (initial
generation, each reflection critique and revision, the advisor pass).
It walks a list of model identifiers, retrying each on transient
failures before moving to the next, and reports the models that were
actually used as fallbacks.

Retry policy
------------

For every model in the ``models`` list, the walker makes up to
``retry + 1`` attempts. A *transient* failure — a 5xx response, a
read/connect timeout, or a connection error — increments the attempt
count and triggers another try on the same model. A *permanent*
failure — a 4xx response — is treated as a client error: it is
re-raised immediately without retrying and without consulting the
fallback list. This matches the validation contract: the proxy must
not silently route a malformed request to a different model.

The 4xx branch distinguishes the OpenAI-compatible ``UpstreamError``
subclasses (the adapter raises :class:`UpstreamError` with a populated
``status_code``). Any other :class:`UpstreamError` that lacks a
``status_code`` is conservatively treated as transient — adapter
implementations raise the typed timeouts/unavailable with ``None``
status for exactly that case.

Exhaustion
----------

When every model in the list has exhausted its retry budget without
success, :func:`call_with_fallbacks` raises
:class:`UpstreamExhaustedError`. The error message includes the
substring ``"all backends failed"`` (the validation contract pins
this; see VAL-RT-013). Callers that need to surface a structured HTTP
response (e.g. the FastAPI handler) translate this to ``502 Bad
Gateway`` with the underlying models and last error in the error
details.

``fallbacks_used`` semantics
----------------------------

The returned ``fallbacks_used`` list contains the model identifiers
that were invoked AFTER the primary model failed. The primary model
itself is never in the list, even when it succeeded only after one or
more retries. The list is ordered by invocation: ``fallbacks_used[0]``
is the first fallback the walker tried, and so on. A response served
by the primary on the first attempt yields ``fallbacks_used = []``.
"""

from __future__ import annotations

import logging
from typing import Any

from moaxy.adapters.base import (
    Adapter,
    ChatResponse,
    UpstreamError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)


class UpstreamExhaustedError(UpstreamError):
    """Raised when every model in a fallback chain has been exhausted.

    The exception is raised by :func:`call_with_fallbacks` after the
    last model in the chain has failed ``retry + 1`` times. The
    message is required to contain the substring ``"all backends
    failed"`` so that HTTP error responses can be detected by both
    the validation contract and operators reading the proxy's logs.
    Subclasses :class:`UpstreamError` so that callers that already
    catch upstream errors continue to work after the fallback chain
    is exhausted.

    Attributes:
        models: The list of model identifiers that were tried, in
            order. The first element is the primary; the rest are
            fallbacks. This is the same list the caller passed to
            :func:`call_with_fallbacks`.
        last_error: The final exception raised by the last attempt,
            or ``None`` when no attempt ever ran (e.g. the model list
            was empty).
    """

    def __init__(
        self,
        message: str,
        *,
        models: list[str] | None = None,
        last_error: BaseException | None = None,
    ) -> None:
        super().__init__(message, status_code=None, body=None)
        self.models: list[str] = list(models) if models else []
        self.last_error: BaseException | None = last_error


def _is_transient(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` should trigger a retry on the same model.

    Transient failures are:

    * :class:`UpstreamTimeoutError` — read/connect/pool timeouts.
    * :class:`UpstreamUnavailableError` — connection refused, DNS,
      remote protocol errors.
    * :class:`UpstreamError` with ``status_code >= 500`` — upstream
      returned a 5xx.
    * :class:`UpstreamError` with no ``status_code`` — adapter chose
      not to surface one (typically a transport-level failure that
      was already classified into the typed subclasses above; this
      branch is a safety net for adapters that raise a bare
      :class:`UpstreamError` for an indeterminate error).

    Permanent failures (4xx and other :class:`UpstreamError` with a
    status in the 4xx range) are *not* transient: the client request
    is wrong and retrying on a different model will not help.
    """
    if isinstance(exc, (UpstreamTimeoutError, UpstreamUnavailableError)):
        return True
    if isinstance(exc, UpstreamError):
        status = exc.status_code
        if status is None:
            return True
        return status >= 500
    return False


async def call_with_fallbacks(
    adapter: Adapter,
    models: list[str],
    retry: int,
    **kwargs: Any,
) -> tuple[ChatResponse, list[str]]:
    """Walk ``models`` in order, retrying each on transient errors.

    Args:
        adapter: The :class:`moaxy.adapters.base.Adapter` instance to
            dispatch every LLM call through. The walker calls
            ``adapter.chat(model=current_model, **kwargs)`` once per
            attempt.
        models: Ordered list of model identifiers. The first element
            is the primary model the orchestrator requested; the
            remaining elements are fallbacks. The walker tries them
            in order, advancing to the next entry only when the
            current one has exhausted its retry budget. An empty
            ``models`` list raises :class:`UpstreamExhaustedError`
            immediately.
        retry: Per-model retry budget. The walker makes up to
            ``retry + 1`` attempts on each model (one initial call
            plus ``retry`` retries on transient failures). A
            negative value is treated as zero. Permanent (4xx)
            failures bypass the budget.
        **kwargs: Keyword arguments forwarded verbatim to
            ``adapter.chat``. Must NOT contain ``model`` (the
            walker injects the current model name itself); any
            other OpenAI body field is allowed (``messages``,
            ``temperature``, ``top_p``, ``max_tokens``, etc.).

    Returns:
        A ``(response, fallbacks_used)`` tuple. ``response`` is the
        :class:`ChatResponse` returned by the successful attempt.
        ``fallbacks_used`` is a list of fallback model identifiers
        (i.e. every model in ``models[1:]`` that the walker tried
        before finding a successful attempt). The primary model
        itself is never included, even when it succeeded only after
        one or more retries. The list is empty when the primary
        model succeeded on its first attempt.

    Raises:
        UpstreamError: A permanent (4xx) failure on any model. The
            exception bubbles up unchanged; the walker does not
            advance to the next model because the request itself is
            invalid.
        UpstreamExhaustedError: Every model in ``models`` has
            exhausted its retry budget without success. The
            exception's message contains the substring
            ``"all backends failed"``; the ``models`` attribute
            carries the chain that was tried; the ``last_error``
            attribute carries the most recent transient error
            raised by the final attempt.
    """
    if not models:
        raise UpstreamExhaustedError(
            "all backends failed: empty model list",
            models=[],
            last_error=None,
        )

    if "model" in kwargs:
        raise TypeError(
            "call_with_fallbacks() does not accept a 'model' kwarg; "
            "the walker manages the model name from the `models` list. "
            "Remove `model=` from the call site."
        )

    attempts_per_model = max(int(retry), 0) + 1
    fallbacks_used: list[str] = []
    last_error: BaseException | None = None

    for index, model in enumerate(models):
        is_fallback = index > 0
        for attempt in range(attempts_per_model):
            try:
                response = await adapter.chat(model=model, **kwargs)
            except UpstreamError as exc:
                last_error = exc
                if not _is_transient(exc):
                    logger.info(
                        "fallback walker: permanent error on model=%s "
                        "(status=%s); re-raising without retrying",
                        model,
                        exc.status_code,
                    )
                    raise
                logger.warning(
                    "fallback walker: transient error on model=%s "
                    "(attempt %d/%d): %s",
                    model,
                    attempt + 1,
                    attempts_per_model,
                    exc,
                )
                if attempt + 1 < attempts_per_model:
                    continue
                break
            if is_fallback:
                fallbacks_used.append(model)
            return response, fallbacks_used
        if is_fallback:
            fallbacks_used.append(model)

    last_error_str = str(last_error) if last_error is not None else "no attempts"
    raise UpstreamExhaustedError(
        f"all backends failed; no model in the fallback chain succeeded "
        f"(last error: {last_error_str})",
        models=models,
        last_error=last_error,
    )


__all__ = [
    "UpstreamExhaustedError",
    "call_with_fallbacks",
]
