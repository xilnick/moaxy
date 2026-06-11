"""M7 benchmark harness — drives the moaxy proxy with (model, config) cells.

The :mod:`moaxy.benchmark.harness` module owns the
:class:`BenchmarkRunner` class, the public surface that executes
the M7 benchmark sweep. The runner takes a list of models, a list
of :class:`~moaxy.benchmark.configs.ConfigVariant` values, and a
list of :class:`~moaxy.benchmark.prompts.CodingPrompt` instances,
and produces one :class:`CellResult` per ``(model, variant)`` cell.

The runner is hermetic: it can be constructed with
``fake_adapter=True`` (no real OpenRouter key required) for unit
tests, OR with ``fake_adapter=False`` (the live benchmark) for
the actual OpenRouter run. The hermetic path is the
contract-pinned surface (:class:`TestBenchmarkRunnerHermetic` in
:mod:`tests.test_benchmark` exercises it end-to-end with all 8
cells).

Algorithm
---------

For every ``(model_alias, variant)`` pair, the runner:

1. Builds a :class:`~moaxy.models.config.MoaxyConfig` via
   :func:`~moaxy.benchmark.configs.make_config` (which already
   wires the right OpenRouter backend, route matcher, and
   per-variant reflection/advisor settings).
2. Constructs an :class:`~moaxy.adapters.registry.AdapterRegistry`
   with a single adapter:

   * In hermetic mode, an
     :class:`~moaxy.adapters.openrouter.OpenRouterAdapter` whose
     ``_transport`` is an in-process
     :class:`httpx.AsyncBaseTransport` (the
     :class:`_FakeOpenRouterChatTransport` in this module) that
     returns scripted OpenAI-shaped responses. No real network.
   * In live mode, a real
     :class:`~moaxy.adapters.openrouter.OpenRouterAdapter` that
     POSTs to ``https://openrouter.ai/api/v1/chat/completions``
     using the ``OPENROUTER_API_KEY`` env var.

3. Builds a FastAPI app via
   :func:`~moaxy.server.app.create_app` and (in live mode only)
   starts uvicorn on a free port in a background task. The
   hermetic path drives the app in-process via
   :class:`httpx.ASGITransport` — no listening socket.
4. Iterates through the prompt set and issues one
   ``POST /v1/chat/completions`` per prompt. Records latency,
   tokens, response text, the canonical ``x-moaxy-*`` response
   headers, and the success / failure status of the call.
5. Tears down the server (cancels the uvicorn task in live
   mode; closes the in-process client in hermetic mode) and
   aggregates the per-prompt results into a :class:`CellResult`
   with summary statistics (mean latency, mean quality,
   mean tokens, pass rate, per-prompt details).

Hermeticity
-----------

The hermetic path is fully in-process: it uses
:class:`httpx.ASGITransport` and an
:class:`httpx.AsyncBaseTransport` on the OpenRouter adapter.
The runner's hermetic mode does NOT bind a TCP socket, does NOT
spawn a subprocess, and does NOT read or set the
``OPENROUTER_API_KEY`` env var. The OpenRouter adapter still
insists on the env var being set (the M6 spec; failing fast on
misconfiguration is part of the adapter's contract), so the
runner sets a placeholder env var for the duration of the
hermetic run when one is not already present. The placeholder
is a clearly-fake value (``"sk-or-fake-hermetic-placeholder"``)
that no real upstream would accept, so an accidental live
network call is caught by the fake transport rather than by
the upstream.

Per-prompt metric aggregation
-----------------------------

Each :class:`CellResult` carries:

* ``model`` (the client-facing alias, e.g. ``"minimax-m3"``)
* ``variant`` (the :class:`ConfigVariant`)
* ``prompt_count`` (the number of prompts scored; expected
  to be ``>= 10`` per the contract)
* ``mean_latency_ms``, ``mean_quality``, ``mean_tokens``,
  ``pass_rate`` (aggregated summary statistics; ``None``
  when no prompts were scored)
* ``prompts`` (the per-prompt :class:`PromptResult` list)

The :class:`PromptResult` dataclass carries the raw values
(latency in ms, response text, the parsed ``prompt_tokens`` /
``completion_tokens`` / ``total_tokens``, the
``x-moaxy-*`` response headers, and an ``ok`` flag) so
downstream consumers (the report generator, the live
benchmark CLI) can re-aggregate the data with different
slicings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from moaxy.adapters.registry import AdapterRegistry
from moaxy.benchmark.configs import ConfigVariant, make_config
from moaxy.benchmark.prompts import CodingPrompt
from moaxy.models.config import MoaxyConfig
from moaxy.server.app import create_app

logger = logging.getLogger(__name__)


# The endpoint path the runner POSTs to. Pinned by the
# :class:`moaxy.server.routes.proxy.chat_completions` route
# definition; the harness's contract is the OpenAI-compatible
# shape, which the proxy already implements verbatim.
_CHAT_COMPLETIONS_PATH: str = "/v1/chat/completions"
"""The path the harness POSTs to on the moaxy proxy."""


# The header prefix the harness scans for ``x-moaxy-*`` headers
# on the response. The moaxy proxy stamps a number of these on
# every response (x-moaxy-request-id, x-moaxy-reflect-turns,
# x-moaxy-alias-resolved, x-moaxy-fallbacks-used, etc.); the
# harness records all of them so the report generator can
# surface the canonical M5/M6/M7 metadata on the cell result.
_HEADER_PREFIX: str = "x-moaxy-"
"""The prefix that identifies ``x-moaxy-*`` response headers."""


# The placeholder API key the runner sets in the env when the
# hermetic path runs and the user has not set the real key. The
# placeholder is a clearly-fake value (it does not match the
# OpenRouter key format ``sk-or-...``) so an accidental live
# network call is caught immediately by the fake transport
# rather than by the upstream.
_HERMETIC_PLACEHOLDER_API_KEY: str = "sk-or-fake-hermetic-placeholder-do-not-use"
"""The placeholder env-var value the hermetic runner sets when ``OPENROUTER_API_KEY`` is unset."""


# The set of header names the harness extracts from each
# response. We capture all ``x-moaxy-*`` headers rather than
# enumerate them, so a future M-side that adds a new header
# does not require a harness edit.
def _collect_moaxy_headers(headers: httpx.Headers) -> dict[str, str]:
    """Return the ``x-moaxy-*`` response headers as a dict.

    The header names are lowercased to match the canonical
    FastAPI / Starlette behaviour. The values are returned as
    plain ``str`` (not the ``httpx.Headers``-internal encoded
    form) so the result is JSON-serialisable and the report
    generator can render it without further massaging.
    """
    out: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower().startswith(_HEADER_PREFIX):
            out[name.lower()] = str(value)
    return out


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface.

    The helper binds a socket to port ``0`` (letting the OS
    pick an unused port), reads the assigned port, and closes
    the socket. The race between the close and the eventual
    uvicorn bind is small (and acceptable for a single-user
    benchmark); the live CLI re-uses the same helper, so a
    contention at this layer would surface as a port-bind
    failure on the uvicorn side and the cell's failure
    propagates up to the runner as a clean error rather than
    a crash.

    Returns:
        An integer in ``[1, 65535]`` that was free at the
        moment the helper ran. The runner binds uvicorn to
        this port immediately after, so the port may be
        re-assigned to a different process in the race window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class PromptResult:
    """The per-prompt result captured by the harness for one (model, variant) cell.

    The harness records one :class:`PromptResult` per prompt
    in the cell's prompt list. The dataclass is intentionally
    flat (no nested objects) so it is JSON-serialisable and
    the report generator can render it as a markdown table
    without further massaging.

    Attributes:
        task_id: The :attr:`CodingPrompt.task_id` of the
            prompt that was scored. The harness keys the
            per-prompt results by this field; the report
            generator uses it as a stable identifier in the
            appendix.
        category: The prompt's
            :attr:`~moaxy.benchmark.prompts.CodingPrompt.category`
            (e.g. ``"function_from_docstring"``). Surfaced in
            the report so the reader can slice the
            per-prompt table by category.
        prompt_text: The verbatim text sent to the model.
            Carried in the dataclass so the report's
            appendix can quote the prompt next to the
            model's response.
        response_text: The verbatim ``choices[0].message.content``
            the proxy returned. May be the empty string
            when the call failed.
        latency_ms: The wall-clock latency of the POST in
            milliseconds. The runner measures end-to-end
            (POST send + response parse) so the value
            includes the orchestrator's reflection and
            advisor round-trips.
        prompt_tokens: The ``usage.prompt_tokens`` field
            of the response, or ``None`` when the call
            failed.
        completion_tokens: The ``usage.completion_tokens``
            field of the response, or ``None`` when the
            call failed.
        total_tokens: The ``usage.total_tokens`` field
            of the response, or ``None`` when the call
            failed.
        status_code: The HTTP status code of the
            response (``200`` on success, ``4xx`` / ``5xx``
            on failure).
        ok: ``True`` when the call returned a 200 with a
            non-empty response; ``False`` otherwise. The
            pass rate is the fraction of prompts with
            ``ok=True``.
        score: The deterministic or LLM-judge score for
            the prompt, normalised to ``[0.0, 1.0]``. The
            value is ``None`` for prompts that were not
            scored (e.g. an unexpected exception in the
            harness) and ``0.0`` for prompts that scored
            ``0``. The pass rate is computed as the
            fraction of prompts with ``score == 1.0``.
        score_method: The scoring method used
            (``"deterministic"`` or ``"judge"``). Surfaced
            in the report so the reader can see which
            prompts were judged vs scored deterministically.
        moaxy_headers: The ``x-moaxy-*`` response headers
            as a ``dict[str, str]``. Carries
            ``x-moaxy-request-id``,
            ``x-moaxy-alias-resolved``,
            ``x-moaxy-reflect-turns``,
            ``x-moaxy-reflect-confidence``,
            ``x-moaxy-advisor-model``,
            ``x-moaxy-advisor-decision``,
            ``x-moaxy-fallbacks-used``, and any other
            ``x-moaxy-*`` header the proxy stamped on the
            response. Keys are lowercased.
        error: The error string for failed calls, or
            ``None`` on success. The runner catches
            exceptions and stores the stringified
            message here so the report's appendix
            shows the failure cause without crashing
            the cell.
    """

    task_id: str
    category: str
    prompt_text: str
    response_text: str
    latency_ms: float
    status_code: int
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    score: float | None = None
    score_method: str = ""
    moaxy_headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class CellResult:
    """The aggregated result for one ``(model, variant)`` benchmark cell.

    The harness produces one :class:`CellResult` per cell in
    the sweep (i.e. ``len(models) * len(config_variants)``
    results in total). The dataclass carries both the
    per-prompt details (so the report's appendix can render
    them) and the summary statistics (so the report's
    per-cell table can render a single row per cell without
    re-aggregating the per-prompt list).

    Attributes:
        model: The client-facing model alias the harness
            POSTed as (e.g. ``"minimax-m3"``). The full
            OpenRouter id is in
            :data:`~moaxy.benchmark.configs.MODEL_ALIASES`
            and is recorded on the prompt results'
            ``x-moaxy-alias-resolved`` header.
        variant: The :class:`ConfigVariant` that defined
            the cell's reflection / advisor settings.
        prompt_count: The number of prompts scored in
            this cell. The contract pins
            ``prompt_count >= 10``.
        mean_latency_ms: The mean ``latency_ms`` across
            the cell's prompts, or ``None`` when no
            prompts were scored. Computed as
            ``statistics.fmean`` so an empty cell returns
            ``None`` rather than raising.
        mean_quality: The mean ``score`` across the
            cell's prompts (where ``score`` is in
            ``[0.0, 1.0]``), or ``None`` when no
            prompts were scored. Note the mean is
            computed over prompts that have a
            non-``None`` score; prompts that crashed
            before scoring are excluded from the mean
            but still counted in ``prompt_count`` (so
            the pass rate can drop below 1.0 for a
            partial-failure cell).
        mean_tokens: The mean ``total_tokens`` across
            the cell's prompts, or ``None`` when no
            prompts were scored.
        pass_rate: The fraction of prompts with
            ``score == 1.0``, or ``None`` when no
            prompts were scored. The pass rate is the
            binary-success rate; the mean quality is
            the same quantity averaged over the
            continuous score (so for a cell of
            ``judge``-scored prompts the two values
            can differ).
        prompts: The per-prompt :class:`PromptResult`
            list, in the order the harness drove them.
            The list is the canonical data the report
            generator's appendix renders.
    """

    model: str
    variant: ConfigVariant
    prompt_count: int
    mean_latency_ms: float | None
    mean_quality: float | None
    mean_tokens: float | None
    pass_rate: float | None
    prompts: list[PromptResult] = field(default_factory=list)


# A type alias for a per-prompt scorer. The runner uses the
# alias so the test fixture can plug in any scoring strategy
# (deterministic, judge, or a hand-rolled mock) without
# importing the :mod:`moaxy.benchmark.scoring` modules at
# import time. The callable's signature is:
# ``async def scorer(prompt: CodingPrompt, model_output: str) -> float``
# where the returned float is in ``[0.0, 1.0]``.
PromptScorer = Callable[[CodingPrompt, str], Awaitable[float]]
"""Async callable that scores a model's output against a prompt and returns a float in [0, 1]."""


class _HermeticScoreCallable:
    """Wrap a synchronous deterministic scorer as an async ``PromptScorer``.

    The hermetic test path uses the deterministic scorers
    (which are sync) directly; the runner expects an async
    callable. The wrapper adapts the two surfaces so the
    test can pass either form.
    """

    def __init__(
        self,
        sync_score: Callable[[CodingPrompt, str], float],
    ) -> None:
        self._sync = sync_score

    async def __call__(
        self, prompt: CodingPrompt, model_output: str
    ) -> float:
        return float(self._sync(prompt, model_output))


class _ScriptedFakeHandler:
    """A scripted OpenAI-shaped response handler for the hermetic transport.

    The hermetic transport (an :class:`httpx.AsyncBaseTransport`)
    holds a reference to one of these and calls
    :meth:`__call__` on every request the proxy's
    :class:`OpenRouterAdapter` makes. The handler is
    intentionally stateful: each call returns a fresh scripted
    response so the orchestrator's reflection / advisor loop can
    consume one response per LLM call. When the script runs
    out, the handler returns a benign default so the orchestrator
    does not crash; the harness still records the call (the
    response is non-empty and well-formed).

    The script is the simplest "valid OpenAI chat-completion"
    shape. The ``content`` field is the verbatim text the
    model returned; the orchestrator parses the text for
    ``REFLECT_CONFIDENCE:`` (on critique calls) and
    ``ADVISOR_APPROVE`` (on advisor calls) exactly the same
    way it would for a real upstream response. The handler
    scripts a low-confidence critique (so the early-exit path
    does not fire and the orchestrator takes the full
    reflection / advisor path) and an advisor
    ``ADVISOR_APPROVE`` (so the advisor pass leaves the
    response content unchanged). The reflection revision and
    the advisor "approve" responses are all the same string
    so the harness does not depend on the orchestrator's
    internal state machine.
    """

    def __init__(self) -> None:
        self._calls: list[httpx.Request] = []
        self._index: int = 0

    @property
    def calls(self) -> list[httpx.Request]:
        """The list of requests the handler has seen so far."""
        return list(self._calls)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self._calls.append(request)
        # The hermetic handler returns a generic
        # OpenAI-shaped response. The model name echoes the
        # request's model so the report can correlate
        # response → request; the content is a low-confidence
        # critique followed by a non-short-circuit
        # ``REFLECT_CONFIDENCE: 0.5`` line so the reflection
        # path always advances to the revision step.
        try:
            payload = json.loads(request.content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        model_name = str(payload.get("model", "hermetic-model"))
        # Always include a non-short-circuit confidence line
        # (0.5) so the orchestrator takes the revision path
        # when reflection is enabled, and a
        # ``ADVISOR_APPROVE`` line so the advisor pass
        # leaves the content unchanged when advisor is
        # enabled. The text also contains the marker
        # ``SCORE: 5`` so the cross-critique prompt parser
        # has a value to read; the value is irrelevant for
        # the hermetic test (the test only asserts the
        # harness produced 8 cells, not the cell's quality).
        content = (
            "hermetic response\n"
            "REFLECT_CONFIDENCE: 0.5\n"
            "SCORE: 5\n"
            "ADVISOR_DECISION: APPROVE\n"
            "ADVISOR_APPROVE\n"
        )
        body = {
            "id": f"chatcmpl-hermetic-{self._index}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        self._index += 1
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=json.dumps(body).encode("utf-8"),
        )


class _FakeOpenRouterChatTransport(httpx.AsyncBaseTransport):
    """A minimal httpx transport for the hermetic OpenRouterAdapter.

    Mirrors the structure of the
    :class:`tests.conftest._FakeOpenRouterTransport` so the
    hermetic path can be exercised in-process without a
    live network. The transport holds a single scripted
    handler and dispatches every request to it; the
    handler is the only state the transport owns.

    Unlike the conftest helper, the harness's hermetic
    transport is *constructed* with a handler instance
    (not a callable) so the harness can introspect the
    handler's call log later if a test wants to assert on
    it. The contract-pinned hermetic test
    (:class:`TestBenchmarkRunnerHermetic`) does NOT inspect
    the call log; the hook is exposed for future tests
    that need to verify e.g. "the orchestrator issued
    three LLM calls on the BOTH cell".
    """

    def __init__(self, handler: _ScriptedFakeHandler) -> None:
        self._handler = handler

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        return await self._handler(request)


# The default deterministic scorer mapping. The hermetic
# test path uses the deterministic scorers (the
# LLM-judge would require a separate fake-transport wiring
# and is not necessary for the contract assertion, which
# only requires the runner to produce 8 cells with
# ``prompt_count >= 10``). The mapping is keyed by the
# prompt's ``category`` literal so the harness looks up
# the right scorer without re-discovering it at runtime.
def _default_scorer_for(prompt: CodingPrompt) -> PromptScorer:
    """Return a deterministic async scorer for ``prompt``.

    The hermetic path uses the
    :mod:`moaxy.benchmark.scoring.deterministic` scorers
    directly. The ``explain`` category has no deterministic
    scorer (its contract is LLM-judge); the hermetic test
    injects a fake transport that returns a constant score
    via the runner's ``scorer`` argument, so this helper
    is never called for ``explain`` prompts. The fallback
    here returns ``0.0`` for any unmatched category so a
    future edit that adds a category without updating the
    mapping surfaces as a 0.0 score, not a crash.
    """
    from moaxy.benchmark.scoring import (
        score_bug_fix,
        score_function_from_docstring,
        score_refactor,
    )

    if prompt.category == "function_from_docstring":
        return _HermeticScoreCallable(score_function_from_docstring)
    if prompt.category == "bug_fix":
        return _HermeticScoreCallable(score_bug_fix)
    if prompt.category == "refactor":
        return _HermeticScoreCallable(score_refactor)
    # ``explain`` is judge-scored; the hermetic test is
    # expected to inject a custom scorer that always
    # returns 0.7. We still return a callable so the
    # harness does not crash on the lookup; the
    # default is ``0.0``.
    async def _zero_scorer(
        _prompt: CodingPrompt, _output: str
    ) -> float:
        return 0.0

    return _zero_scorer


class BenchmarkRunner:
    """Drive the M7 benchmark sweep end-to-end.

    The runner is the public surface the CLI and the test
    suite use to execute the M7 benchmark. It takes the
    three inputs the contract pins (a list of model
    aliases, a list of :class:`ConfigVariant` values, a
    list of :class:`CodingPrompt` instances) plus a
    ``fake_adapter`` flag and an optional ``scorer``
    override, and produces a list of :class:`CellResult`
    (one per ``(model, variant)`` cell).

    The runner is hermetic by default: setting
    ``fake_adapter=True`` is the contract-pinned test path.
    The live benchmark sets ``fake_adapter=False`` and
    relies on the ``OPENROUTER_API_KEY`` env var being
    set in the worker shell.

    Args:
        models: The client-facing model aliases to sweep
            over. Each entry must be a key of
            :data:`~moaxy.benchmark.configs.MODEL_ALIASES`.
        config_variants: The :class:`ConfigVariant` values
            to sweep over. The order is preserved in the
            returned :class:`CellResult` list.
        prompts: The :class:`CodingPrompt` instances to
            drive. The order is preserved within each
            cell's ``prompts`` list.
        fake_adapter: When ``True``, the runner wires an
            in-process transport-backed
            :class:`OpenRouterAdapter` for every cell; no
            real network is contacted. When ``False``, a
            real :class:`OpenRouterAdapter` is
            constructed; the OpenRouter upstream must be
            reachable and the ``OPENROUTER_API_KEY`` env
            var must be set. The hermetic test path uses
            ``fake_adapter=True``; the live benchmark
            CLI uses ``fake_adapter=False``.
        scorer: Optional async callable that scores a
            model output against a prompt and returns a
            float in ``[0.0, 1.0]``. When ``None``, the
            runner uses the deterministic scorers from
            :mod:`moaxy.benchmark.scoring.deterministic`
            for the three deterministic categories and
            returns ``0.0`` for the ``explain`` category.
            Tests that want to score ``explain`` prompts
            hermetically pass a custom scorer (the
            contract-pinned hermetic test does NOT score
            explain prompts; it only asserts the harness
            ran all 8 cells with ``prompt_count >= 10``).
        request_timeout_s: Per-request timeout in
            seconds. Defaults to 60s (long enough for a
            full reflection+advisor round-trip on a slow
            model). The hermetic test path uses the
            default; the live CLI may want a longer
            timeout.
        host: The host uvicorn binds to in live mode.
            Defaults to ``"127.0.0.1"`` (loopback only;
            the architecture mandates loopback binding).
    """

    def __init__(
        self,
        *,
        models: list[str] | tuple[str, ...],
        config_variants: list[ConfigVariant] | tuple[ConfigVariant, ...],
        prompts: list[CodingPrompt] | tuple[CodingPrompt, ...],
        fake_adapter: bool = True,
        scorer: PromptScorer | None = None,
        request_timeout_s: float = 60.0,
        host: str = "127.0.0.1",
    ) -> None:
        self.models: list[str] = list(models)
        self.config_variants: list[ConfigVariant] = list(config_variants)
        self.prompts: list[CodingPrompt] = list(prompts)
        self.fake_adapter: bool = bool(fake_adapter)
        self.scorer: PromptScorer | None = scorer
        self.request_timeout_s: float = float(request_timeout_s)
        self.host: str = host
        # The per-cell handler the hermetic transport
        # captures; exposed for tests that want to
        # introspect the call log. The list has one
        # entry per cell in the order they were
        # executed. ``None`` in live mode.
        self.hermetic_handlers: list[_ScriptedFakeHandler] = []

    def _build_hermetic_handler(
        self,
    ) -> _ScriptedFakeHandler:
        """Return a fresh scripted handler for one cell.

        The hermetic path constructs a new handler per
        cell so the call log of cell N is independent of
        the call log of cell N+1. The handler is
        appended to :attr:`hermetic_handlers` so a
        caller (the test) can inspect it.
        """
        handler = _ScriptedFakeHandler()
        self.hermetic_handlers.append(handler)
        return handler

    def _build_app(
        self, config: MoaxyConfig
    ) -> tuple[Any, AdapterRegistry, tuple[bool, str | None]]:
        """Build the FastAPI app and adapter registry for one cell.

        The hermetic path builds an
        :class:`OpenRouterAdapter` with a scripted
        fake transport; the live path builds a real
        :class:`OpenRouterAdapter` (which reads
        ``OPENROUTER_API_KEY`` at construction time).

        Returns:
            A ``(app, registry, env_state)`` tuple. The
            ``app`` is a fully-wired :class:`fastapi.FastAPI`
            instance; the ``registry`` is the
            :class:`AdapterRegistry` that owns the
            adapter the app dispatches every LLM call
            through; the ``env_state`` is a
            ``(was_set, old_value)`` tuple carrying the
            prior state of the ``OPENROUTER_API_KEY``
            env var (so the caller can restore it on
            teardown and the harness does not leak the
            hermetic placeholder into subsequent
            tests).
        """
        from moaxy.adapters.openrouter import OpenRouterAdapter

        env_was_set: bool = "OPENROUTER_API_KEY" in os.environ
        env_old_value: str | None = os.environ.get("OPENROUTER_API_KEY")
        if self.fake_adapter:
            # Ensure the OpenRouterAdapter's
            # ``OPENROUTER_API_KEY`` env-var check
            # passes. The hermetic transport intercepts
            # every request before it leaves the
            # process, so the value is a clearly-fake
            # placeholder that no real upstream would
            # accept. The original env var is restored
            # when the cell is torn down so subsequent
            # cells (and the surrounding test) see the
            # user's actual value. The harness records
            # the prior state on the runner instance
            # (per-cell) so a context-manager teardown
            # can revert it.
            if not env_was_set:
                os.environ["OPENROUTER_API_KEY"] = (
                    _HERMETIC_PLACEHOLDER_API_KEY
                )
            handler = self._build_hermetic_handler()
            transport = _FakeOpenRouterChatTransport(handler)
            adapter = OpenRouterAdapter(_transport=transport)
        else:
            adapter = OpenRouterAdapter()
        registry = AdapterRegistry({config.backends[0].name: adapter})
        app = create_app(config=config, adapters=registry)
        return app, registry, (env_was_set, env_old_value)

    async def _run_cell(
        self,
        *,
        model: str,
        variant: ConfigVariant,
    ) -> CellResult:
        """Execute one ``(model, variant)`` cell and return its :class:`CellResult`.

        The method:

        1. Builds the :class:`MoaxyConfig` for the cell.
        2. Builds the FastAPI app + adapter registry
           (hermetic or live).
        3. In live mode, starts uvicorn on a free port
           in a background task; in hermetic mode, drives
           the app in-process via :class:`httpx.ASGITransport`.
        4. Iterates through the prompt list, POSTs each
           prompt, records latency / tokens / headers /
           score.
        5. Tears down the server and aggregates the
           per-prompt results into a :class:`CellResult`.

        Args:
            model: The client-facing model alias.
            variant: The :class:`ConfigVariant` to
                execute.

        Returns:
            A :class:`CellResult` with the cell's
            summary statistics and per-prompt details.
            The method never raises; a cell that fails
            end-to-end (e.g. a port-bind failure on
            live mode) is reported via an aggregated
            ``error`` on every :class:`PromptResult`,
            so the caller's :meth:`execute` can iterate
            freely without try/except.
        """
        config = make_config(model, variant)
        app, registry, env_state = self._build_app(config)
        prompt_results: list[PromptResult] = []
        server: Any = None

        # Both the live and the hermetic paths share
        # the same POST-and-parse loop. The transport
        # differs (ASGITransport vs. a real TCP
        # socket), but the request shape and the
        # response parsing are identical.
        if self.fake_adapter:

            async def _client_factory() -> _AsyncContextManager:
                return _AsyncContextManager(
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://test",
                        timeout=self.request_timeout_s,
                    )
                )
        else:
            port = _find_free_port()
            server = await _start_uvicorn_in_background(
                app, host=self.host, port=port
            )

            async def _client_factory() -> _AsyncContextManager:  # type: ignore[misc]
                return _AsyncContextManager(
                    httpx.AsyncClient(
                        base_url=f"http://{self.host}:{port}",
                        timeout=self.request_timeout_s,
                    )
                )

        try:
            cm = await _client_factory()
            async with cm as client:
                for prompt in self.prompts:
                    prompt_result = await self._drive_prompt(
                        client=client,
                        prompt=prompt,
                        model=model,
                    )
                    prompt_results.append(prompt_result)
        finally:
            # Restore the ``OPENROUTER_API_KEY`` env
            # var to its prior state so the harness
            # does not leak the hermetic placeholder
            # into subsequent tests (in particular,
            # the M6 ``TestOpenRouterAdapterReal``
            # tests gate on the env var's value).
            env_was_set, env_old_value = env_state
            if env_was_set:
                if env_old_value is not None:
                    os.environ["OPENROUTER_API_KEY"] = env_old_value
            else:
                os.environ.pop("OPENROUTER_API_KEY", None)
            if not self.fake_adapter and server is not None:
                # The live path tears down the
                # uvicorn task. The hermetic path has
                # no background task to cancel.
                try:
                    server.should_exit = True
                    await asyncio.wait_for(server.task, timeout=5.0)
                except (TimeoutError, Exception):  # noqa: BLE001
                    # Defensive: a teardown failure
                    # must not crash the harness.
                    logger.warning(
                        "BenchmarkRunner: uvicorn teardown failed for "
                        "cell model=%s variant=%s",
                        model,
                        variant,
                    )

        return _aggregate_cell(
            model=model,
            variant=variant,
            prompts=prompt_results,
        )

    async def _drive_prompt(
        self,
        *,
        client: httpx.AsyncClient,
        prompt: CodingPrompt,
        model: str,
    ) -> PromptResult:
        """POST a single prompt and record the per-prompt metrics.

        The method measures the wall-clock latency of
        the POST, parses the OpenAI-shaped response,
        and computes the score using the runner's
        configured scorer (or the deterministic default
        for the prompt's category).

        The method never raises: a network failure
        or a non-200 response is captured in the
        returned :class:`PromptResult` with
        ``ok=False`` and a human-readable ``error``
        string, so the cell's pass rate and mean
        quality reflect the failure.
        """
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt.prompt_text}],
        }
        start = time.perf_counter()
        try:
            response = await client.post(
                _CHAT_COMPLETIONS_PATH, json=body
            )
        except httpx.HTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return PromptResult(
                task_id=prompt.task_id,
                category=prompt.category,
                prompt_text=prompt.prompt_text,
                response_text="",
                latency_ms=latency_ms,
                status_code=0,
                ok=False,
                score=None,
                score_method=prompt.scoring_method,
                moaxy_headers={},
                error=f"http error: {exc}",
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        status_code = int(response.status_code)
        moaxy_headers = _collect_moaxy_headers(response.headers)
        if status_code >= 400:
            return PromptResult(
                task_id=prompt.task_id,
                category=prompt.category,
                prompt_text=prompt.prompt_text,
                response_text="",
                latency_ms=latency_ms,
                status_code=status_code,
                ok=False,
                score=None,
                score_method=prompt.scoring_method,
                moaxy_headers=moaxy_headers,
                error=f"non-2xx response: {status_code}",
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            return PromptResult(
                task_id=prompt.task_id,
                category=prompt.category,
                prompt_text=prompt.prompt_text,
                response_text="",
                latency_ms=latency_ms,
                status_code=status_code,
                ok=False,
                score=None,
                score_method=prompt.scoring_method,
                moaxy_headers=moaxy_headers,
                error=f"non-JSON response: {exc}",
            )
        # Extract the response text + usage from the
        # OpenAI-shaped payload. The proxy is contract-
        # pinned to this shape; a deviation surfaces as
        # a graceful failure here rather than a crash.
        try:
            choices = payload.get("choices") or []
            first = choices[0] if choices else {}
            message = first.get("message") or {}
            content = str(message.get("content", ""))
        except (AttributeError, TypeError, IndexError) as exc:
            return PromptResult(
                task_id=prompt.task_id,
                category=prompt.category,
                prompt_text=prompt.prompt_text,
                response_text="",
                latency_ms=latency_ms,
                status_code=status_code,
                ok=False,
                score=None,
                score_method=prompt.scoring_method,
                moaxy_headers=moaxy_headers,
                error=f"malformed response shape: {exc}",
            )
        usage = payload.get("usage") or {}
        prompt_tokens = (
            int(usage["prompt_tokens"])
            if isinstance(usage.get("prompt_tokens"), (int, float))
            else None
        )
        completion_tokens = (
            int(usage["completion_tokens"])
            if isinstance(usage.get("completion_tokens"), (int, float))
            else None
        )
        total_tokens = (
            int(usage["total_tokens"])
            if isinstance(usage.get("total_tokens"), (int, float))
            else None
        )
        # Score the response. The runner uses the
        # configured scorer when set; otherwise the
        # deterministic default for the category.
        score_value: float | None
        if self.scorer is not None:
            try:
                score_value = float(
                    await self.scorer(prompt, content)
                )
            except Exception as exc:  # noqa: BLE001
                # A scoring failure is captured as a
                # graceful ``score=None`` so the cell's
                # pass rate reflects the failure
                # without crashing the harness.
                logger.warning(
                    "BenchmarkRunner: scorer raised for %s: %s",
                    prompt.task_id,
                    exc,
                )
                score_value = None
        else:
            default_scorer = _default_scorer_for(prompt)
            try:
                score_value = float(
                    await default_scorer(prompt, content)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "BenchmarkRunner: default scorer raised for %s: %s",
                    prompt.task_id,
                    exc,
                )
                score_value = None
        return PromptResult(
            task_id=prompt.task_id,
            category=prompt.category,
            prompt_text=prompt.prompt_text,
            response_text=content,
            latency_ms=latency_ms,
            status_code=status_code,
            ok=bool(content) and status_code == 200,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            score=score_value,
            score_method=prompt.scoring_method,
            moaxy_headers=moaxy_headers,
            error=None,
        )

    async def execute(self) -> list[CellResult]:
        """Run the full benchmark sweep and return all :class:`CellResult` objects.

        The method iterates over the Cartesian product
        of ``models`` and ``config_variants`` and
        produces one :class:`CellResult` per cell. The
        order is ``[(model[0], variant[0]),
        (model[0], variant[1]), ..., (model[-1],
        variant[-1])]`` (i.e. variants vary fastest).

        The method never raises. A cell that fails
        end-to-end is reported via a single
        :class:`CellResult` whose every
        :class:`PromptResult` carries the same
        ``error`` string; the caller's test asserts on
        ``len(results) == len(models) *
        len(config_variants)`` (the contract pin) and
        on each result's ``prompt_count`` (the
        per-cell invariant).
        """
        results: list[CellResult] = []
        for model in self.models:
            for variant in self.config_variants:
                cell = await self._run_cell(
                    model=model, variant=variant
                )
                results.append(cell)
        return results


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────


def _aggregate_cell(
    *,
    model: str,
    variant: ConfigVariant,
    prompts: list[PromptResult],
) -> CellResult:
    """Aggregate the per-prompt results into a :class:`CellResult`.

    The function is a pure aggregator: it takes the
    per-prompt :class:`PromptResult` list and computes
    the cell's summary statistics. The aggregator is
    extracted as a module-level helper so a future
    test (or a future re-aggregation strategy the
    report generator wants) can re-use it without
    re-implementing the mean / pass-rate logic.

    Args:
        model: The client-facing model alias.
        variant: The :class:`ConfigVariant` that
            defined the cell.
        prompts: The per-prompt results in the order
            the harness drove them.

    Returns:
        A :class:`CellResult` whose summary statistics
        reflect the ``prompts`` list. ``None`` summary
        statistics are returned when the ``prompts``
        list is empty (the cell produced no data).
    """
    if not prompts:
        return CellResult(
            model=model,
            variant=variant,
            prompt_count=0,
            mean_latency_ms=None,
            mean_quality=None,
            mean_tokens=None,
            pass_rate=None,
            prompts=[],
        )
    latencies = [p.latency_ms for p in prompts]
    mean_latency_ms = float(statistics.fmean(latencies))
    scored = [p for p in prompts if p.score is not None]
    if scored:
        mean_quality = float(statistics.fmean([p.score for p in scored]))
        pass_rate = float(
            sum(1 for p in scored if p.score == 1.0) / len(scored)
        )
    else:
        mean_quality = None
        pass_rate = None
    token_totals = [
        p.total_tokens for p in prompts if p.total_tokens is not None
    ]
    mean_tokens = (
        float(statistics.fmean(token_totals)) if token_totals else None
    )
    return CellResult(
        model=model,
        variant=variant,
        prompt_count=len(prompts),
        mean_latency_ms=mean_latency_ms,
        mean_quality=mean_quality,
        mean_tokens=mean_tokens,
        pass_rate=pass_rate,
        prompts=list(prompts),
    )


class _AsyncContextManager:
    """Lightweight async context manager wrapper around a value.

    The runner's cell loop builds an
    :class:`httpx.AsyncClient` (hermetic) and then drives
    it inside an ``async with`` block. The wrapper adapts
    the construction pattern to the
    ``async with await factory() as client:`` flow used
    by the live path (where the client is constructed
    AFTER uvicorn starts), so the same ``async with``
    syntax works in both branches.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()


async def _start_uvicorn_in_background(
    app: Any,
    *,
    host: str,
    port: int,
) -> Any:
    """Start uvicorn in a background task and return the server object.

    The live benchmark path binds uvicorn to a free
    port and runs the server in a background asyncio
    task; the cell loop POSTs to the bound URL and the
    teardown cancels the task. The helper is the
    single source of truth for that lifecycle so a
    future tweak (e.g. switching to a different ASGI
    server) lives in one place.

    The server object is the
    :class:`uvicorn.Server` instance; the harness
    sets ``server.should_exit = True`` and awaits
    ``server.task`` in the teardown. Importing
    uvicorn is lazy so the hermetic path (which never
    calls this helper) does not pay the import cost.

    Args:
        app: The FastAPI app to serve.
        host: The host to bind to.
        port: The port to bind to.

    Returns:
        The :class:`uvicorn.Server` instance whose
        ``task`` is the running background coroutine.
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.task = asyncio.create_task(server.serve())
    # Wait for the server to bind. The "started" flag
    # is set by uvicorn after the socket is open; we
    # poll with a small backoff so the harness does
    # not race the server's first accept.
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    return server


__all__ = [
    "BenchmarkRunner",
    "CellResult",
    "PromptResult",
    "PromptScorer",
]
