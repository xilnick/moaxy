"""M7 benchmark config variants.

The :mod:`moaxy.benchmark.configs` module owns the four
:class:`ConfigVariant` values that the M7 benchmark harness sweeps
over for every (model, configuration) cell:

* :attr:`ConfigVariant.BASELINE` — no reflection, no advisor; one
  outbound LLM call per request. The control cell.
* :attr:`ConfigVariant.REFLECTION_ONLY` — one self-reflection turn,
  no advisor. The cell that measures the value of self-critique in
  isolation.
* :attr:`ConfigVariant.ADVISOR_ONLY` — no self-reflection, one
  advisor turn using the OTHER comparison model as the advisor
  (cross-advise). The cell that measures the value of an external
  model critic in isolation.
* :attr:`ConfigVariant.BOTH` — one self-reflection turn followed by
  one advisor turn. The cell that measures the combined effect.

For every model in :data:`COMPARISON_MODELS`, the :func:`make_config`
factory returns a fully-validated :class:`moaxy.models.config.MoaxyConfig`
that routes the model through the canonical OpenRouter backend and
applies the variant's reflection/advisor configuration.

The contract (VAL-BENCH-002) asserts the following:

* Exactly four variants exist on the :class:`ConfigVariant` enum.
* For every variant and every model in :data:`COMPARISON_MODELS`,
  the result of :func:`make_config` validates cleanly through
  :meth:`moaxy.models.config.MoaxyConfig.model_validate` (i.e. the
  Pydantic v2 schema accepts the constructed config without raising).
* The :attr:`~moaxy.models.config.AdvisorConfig.model` field of
  :attr:`ConfigVariant.ADVISOR_ONLY` and :attr:`ConfigVariant.BOTH`
  is the OTHER comparison model's full OpenRouter id (cross-advise).

The module deliberately re-exports :data:`COMPARISON_MODELS`,
:data:`MODEL_ALIASES`, and :func:`make_config` so callers (the
:class:`~moaxy.benchmark.harness.BenchmarkRunner`, the live
benchmark CLI, and the test suite) can import the full set of
public names from a single import path. Editing a model name?
Update the alias table, :data:`COMPARISON_MODELS`, and the
contract pins in :mod:`tests.test_benchmark` in lockstep.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict

from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    MoaxyConfig,
    ReflectionConfig,
    RouteConfig,
    RouteMatch,
)

# ────────────────────────────────────────────────────────────────────
# Model alias table
# ────────────────────────────────────────────────────────────────────
#
# The benchmark aliases (the model name the *client* sends) map to
# the full OpenRouter model id (the name the upstream API expects).
# The alias is short and human-friendly ("minimax-m3", "mimo-v2.5-pro");
# the OpenRouter id is provider-prefixed ("minimax/minimax-m3",
# "xiaomi/mimo-v2.5-pro"). The alias is what the harness uses when
# it issues the POST; the OpenRouter id is what the adapter sends to
# upstream. The mapping is captured as a module-level constant so
# callers do not have to maintain a parallel table.
#
# The two models in :data:`COMPARISON_MODELS` are the cells the live
# benchmark runs against. Both aliases are pinned by the contract
# (VAL-BENCH-002) and are matched verbatim by the
# :class:`TestConfigVariantsContract` tests.
MODEL_ALIASES: Final[dict[str, str]] = {
    "minimax-m3": "minimax/minimax-m3",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
}
"""Alias → OpenRouter model id mapping for the benchmark sweep.

The keys are the short, human-friendly aliases the harness uses as
the ``model`` field of its POST requests. The values are the full
provider-prefixed model ids the OpenRouter API expects. The harness
issues one request per ``(model, variant)`` cell where ``model``
ranges over :data:`COMPARISON_MODELS` (the keys of this table).
"""


COMPARISON_MODELS: Final[tuple[str, ...]] = (
    "minimax-m3",
    "mimo-v2.5-pro",
)
"""The two model aliases the benchmark sweeps over.

The contract (VAL-BENCH-002) fixes this tuple at length 2; the live
benchmark runs ``len(COMPARISON_MODELS) * len(ConfigVariant) = 8``
cells. Cross-advise for a model is defined as ``the other model in
this tuple`` — so when testing ``minimax-m3`` with
:attr:`ConfigVariant.ADVISOR_ONLY`, the advisor is
``mimo-v2.5-pro``'s OpenRouter id, and vice versa.

Editing this tuple? Update the test in
:mod:`tests.test_benchmark` that pins the length-2 invariant.
"""


# The single backend used by every benchmark cell. The harness does
# not mix backends; all cells route through one OpenRouter account
# to keep the per-run token accounting uniform.
_BACKEND_NAME: Final[str] = "openrouter-main"
"""The single backend name shared by every benchmark config variant.

Every :func:`make_config` call registers one
:class:`moaxy.models.config.AdapterConfig` with this name; the
generated :class:`moaxy.models.config.RouteConfig` references it.
The harness's :class:`AdapterRegistry` looks up adapters by this
name at request time.
"""


# The alias of the route that wraps every benchmark cell. The route
# name is informational (visible in logs and in
# :attr:`moaxy.models.config.RouteConfig.name`); the matcher keys
# off the route's ``match.model`` and ``match.path`` glob patterns
# rather than the route name.
_ROUTE_NAME: Final[str] = "bench"
"""The canonical route name shared by every benchmark cell."""


class ConfigVariant(Enum):
    """The four configuration variants the M7 benchmark sweeps.

    Each variant fixes the values of
    :attr:`~moaxy.models.config.ReflectionConfig.turns` and
    :attr:`~moaxy.models.config.AdvisorConfig.turns` (and, for
    advisor-enabled variants, the advisor's
    :attr:`~moaxy.models.config.AdvisorConfig.model`). The
    :func:`make_config` factory consults this enum to assemble a
    fully-validated :class:`moaxy.models.config.MoaxyConfig`.

    Values:

    * ``BASELINE`` — ``reflection.turns = 0``, ``advisor.turns = 0``.
      One LLM call per request. The control cell.
    * ``REFLECTION_ONLY`` — ``reflection.turns = 1``,
      ``advisor.turns = 0``. The cell that isolates the value of
      self-critique.
    * ``ADVISOR_ONLY`` — ``reflection.turns = 0``,
      ``advisor.turns = 1``, ``advisor.model = the OTHER comparison
      model`` (cross-advise). The cell that isolates the value of
      an external model critic.
    * ``BOTH`` — ``reflection.turns = 1``, ``advisor.turns = 1``,
      ``advisor.model = the OTHER comparison model`` (cross-advise).
      The cell that measures the combined effect.

    The enum has exactly four members. The contract
    (VAL-BENCH-002) pins ``len(ConfigVariant) == 4`` and the
    explicit member names; the
    :class:`TestConfigVariantsContract` test class enforces both
    invariants.
    """

    BASELINE = "baseline"
    REFLECTION_ONLY = "reflection_only"
    ADVISOR_ONLY = "advisor_only"
    BOTH = "both"


# Pin the canonical cross-advise model for a given (model, variant)
# pair. The function is a pure lookup with no side effects; it is
# defined as a module-level helper so the cross-advise rule is
# expressed in exactly one place. The harness and the live benchmark
# CLI both call :func:`make_config` (which consults this helper
# internally) so the cross-advise rule is enforced uniformly across
# all consumers.
def _cross_advise_model(model_alias: str) -> str:
    """Return the OpenRouter id of the OTHER comparison model.

    The function is the single source of truth for the cross-advise
    rule: when a benchmark cell uses ``ADVISOR_ONLY`` or ``BOTH``,
    the advisor is the comparison model the cell is NOT testing. For
    example, when the harness tests ``minimax-m3`` with
    :attr:`ConfigVariant.ADVISOR_ONLY`, the advisor is
    ``xiaomi/mimo-v2.5-pro`` (the OpenRouter id of
    ``mimo-v2.5-pro``), so the two models are not advising
    themselves.

    Args:
        model_alias: The client-facing alias of the model the cell
            is testing. Must be a key of :data:`MODEL_ALIASES` (and
            a member of :data:`COMPARISON_MODELS`).

    Returns:
        The full OpenRouter id of the OTHER model in
        :data:`COMPARISON_MODELS`. For example, when
        ``model_alias == "minimax-m3"``, the return value is
        ``"xiaomi/mimo-v2.5-pro"``.

    Raises:
        ValueError: ``model_alias`` is not a member of
            :data:`COMPARISON_MODELS` (i.e. the caller passed a
            model that the benchmark does not sweep over).
    """
    if model_alias not in COMPARISON_MODELS:
        raise ValueError(
            f"unknown comparison model {model_alias!r}; "
            f"the benchmark sweeps over {COMPARISON_MODELS!r}"
        )
    # The cross-advise model is the OTHER member of the
    # two-element tuple. The list comprehension is the simplest
    # way to express "the element that is not me".
    other = [m for m in COMPARISON_MODELS if m != model_alias]
    if len(other) != 1:
        raise ValueError(
            f"cross-advise lookup failed for {model_alias!r}: "
            f"expected exactly one OTHER model in {COMPARISON_MODELS!r}, "
            f"got {other!r}"
        )
    other_alias = other[0]
    return MODEL_ALIASES[other_alias]


class _VariantParams(BaseModel):
    """Internal typed view of the per-variant reflection/advisor settings.

    The :func:`make_config` factory uses an instance of this model
    to communicate the variant's reflection and advisor settings
    to the Pydantic model constructors without scattering literal
    dicts across the codebase. Defining a Pydantic model for the
    intermediate shape keeps the field types pinned and surfaces
    typos at validation time rather than at runtime.

    The class has ``extra="forbid"`` so a future edit that adds a
    field to the variant spec without updating the factory is
    caught immediately.
    """

    model_config = ConfigDict(extra="forbid")

    reflection_turns: int
    advisor_turns: int
    advisor_model: str | None


def _params_for(variant: ConfigVariant, model_alias: str) -> _VariantParams:
    """Return the per-variant reflection/advisor settings.

    The function is the single source of truth for the
    ``(variant, model) → (reflection.turns, advisor.turns,
    advisor.model)`` mapping. The four expected cases are pinned
    by the contract (VAL-BENCH-002) and matched by the
    :class:`TestConfigVariantsContract` test class.

    Args:
        variant: One of the four :class:`ConfigVariant` values.
        model_alias: The client-facing alias of the model the cell
            is testing. Used only for ``ADVISOR_ONLY`` and ``BOTH``,
            where the advisor is the OTHER comparison model.

    Returns:
        A :class:`_VariantParams` with the variant's
        ``reflection.turns``, ``advisor.turns``, and
        ``advisor.model`` (which is ``None`` for the two advisor-
        disabled variants).
    """
    if variant is ConfigVariant.BASELINE:
        return _VariantParams(
            reflection_turns=0,
            advisor_turns=0,
            advisor_model=None,
        )
    if variant is ConfigVariant.REFLECTION_ONLY:
        return _VariantParams(
            reflection_turns=1,
            advisor_turns=0,
            advisor_model=None,
        )
    if variant is ConfigVariant.ADVISOR_ONLY:
        return _VariantParams(
            reflection_turns=0,
            advisor_turns=1,
            advisor_model=_cross_advise_model(model_alias),
        )
    if variant is ConfigVariant.BOTH:
        return _VariantParams(
            reflection_turns=1,
            advisor_turns=1,
            advisor_model=_cross_advise_model(model_alias),
        )
    # Defensive: Enum members are exhaustive but the explicit
    # ``raise`` documents the invariant for static analysers and
    # surfaces a future enum addition here rather than silently
    # returning the wrong config.
    raise ValueError(f"unhandled ConfigVariant: {variant!r}")


def make_config(model_alias: str, variant: ConfigVariant) -> MoaxyConfig:
    """Build a :class:`MoaxyConfig` for one benchmark cell.

    The returned config:

    * declares one :class:`moaxy.models.config.AdapterConfig`
      named :data:`_BACKEND_NAME` (``"openrouter-main"``) with
      ``adapter == "openrouter"`` and ``base_url`` set to the
      OpenRouter canonical default (``https://openrouter.ai/api/v1``).
      The ``api_key`` field is left ``None`` because the live
      benchmark reads the key from the ``OPENROUTER_API_KEY`` env
      var at adapter construction time; the harness relies on the
      same env-var contract when it instantiates the adapter.
    * declares one :class:`moaxy.models.config.RouteConfig` named
      :data:`_ROUTE_NAME` (``"bench"``) whose ``match.model``
      glob is the model alias (so the route matches the harness's
      POSTs verbatim) and whose ``match.path`` is
      ``"/v1/chat/completions"``. The route references the
      backend by name.
    * aliases the model alias to the full OpenRouter id (e.g.
      ``"minimax-m3"`` → ``"minimax/minimax-m3"``). The harness's
      client sends ``model=<alias>``; the route matcher rewrites
      the alias to the OpenRouter id before the adapter is called.
    * configures the route's
      :class:`~moaxy.models.config.ReflectionConfig` and
      :class:`~moaxy.models.config.AdvisorConfig` to the values
      :func:`_params_for` returns for the requested variant.

    The result is returned as a fully-validated
    :class:`moaxy.models.config.MoaxyConfig`. The contract
    (VAL-BENCH-002) requires the result to round-trip through
    :meth:`moaxy.models.config.MoaxyConfig.model_validate` without
    raising; the factory does not call ``model_validate`` itself
    (the constructor's Pydantic v2 validation already does), so
    the test asserts the round-trip explicitly.

    Args:
        model_alias: The client-facing alias of the model the cell
            is testing. Must be a key of :data:`MODEL_ALIASES` (and
            a member of :data:`COMPARISON_MODELS`).
        variant: One of the four :class:`ConfigVariant` values.

    Returns:
        A :class:`moaxy.models.config.MoaxyConfig` ready to be
        passed to :class:`moaxy.config.loader.load_config`'s
        downstream consumers (the harness, the live CLI).

    Raises:
        ValueError: ``model_alias`` is not a member of
            :data:`COMPARISON_MODELS`.
    """
    if model_alias not in MODEL_ALIASES:
        raise ValueError(
            f"unknown model alias {model_alias!r}; "
            f"known aliases: {sorted(MODEL_ALIASES)!r}"
        )
    if model_alias not in COMPARISON_MODELS:
        raise ValueError(
            f"model alias {model_alias!r} is not in "
            f"COMPARISON_MODELS={COMPARISON_MODELS!r}; the benchmark "
            "sweep is restricted to the canonical comparison set"
        )

    model_id = MODEL_ALIASES[model_alias]
    params = _params_for(variant, model_alias)

    backend = AdapterConfig(
        name=_BACKEND_NAME,
        adapter="openrouter",
        base_url="https://openrouter.ai/api/v1",
        # The ``api_key`` is intentionally ``None``: the live
        # benchmark reads ``OPENROUTER_API_KEY`` from the
        # environment at adapter construction time. Leaving
        # ``api_key=None`` here keeps the config file
        # ``OPENROUTER_API_KEY``-free (the env var is the
        # canonical source) and matches the M6 spec.
    )
    route = RouteConfig(
        name=_ROUTE_NAME,
        match=RouteMatch(
            model=model_alias,
            path="/v1/chat/completions",
        ),
        strategy="single",
        backend=_BACKEND_NAME,
        aliases={model_alias: model_id},
        fallbacks=[],
        retry=0,
        reflection=ReflectionConfig(turns=params.reflection_turns),
        advisor=AdvisorConfig(
            turns=params.advisor_turns,
            model=params.advisor_model,
        ),
    )
    return MoaxyConfig(backends=[backend], routes=[route])


__all__ = [
    "COMPARISON_MODELS",
    "ConfigVariant",
    "MODEL_ALIASES",
    "make_config",
]
