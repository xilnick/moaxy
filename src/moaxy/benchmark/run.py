"""M7 benchmark CLI entry point.

The :mod:`moaxy.benchmark.run` module owns the ``moaxy.benchmark.run``
command-line interface. The CLI is a thin wrapper around the
:class:`~moaxy.benchmark.harness.BenchmarkRunner` and
:class:`~moaxy.benchmark.report.MarkdownReportGenerator`; it parses
the user-supplied arguments, instantiates the runner, executes the
full M7 sweep, writes the JSON results to ``<output>/results.json``,
and renders a markdown report to ``<output>/report.md``.

Usage::

    python -m moaxy.benchmark.run \
        --models minimax-m3,mimo-v2.5-pro \
        --configs baseline,reflection,advisor,both \
        --output .benchmarks/results/m7-run-20260612-024000/ \
        [--fake-adapter]

The ``--fake-adapter`` flag is the hermetic path: the runner wires
an in-process fake transport for the OpenRouter adapter, so no real
``OPENROUTER_API_KEY`` is required. The hermetic path is the
contract-pinned test surface (VAL-BENCH-008).

Argument semantics
------------------

* ``--models``: a comma-separated list of client-facing model
  aliases. Each entry must be a key of
  :data:`~moaxy.benchmark.configs.MODEL_ALIASES` (i.e. one of
  the :data:`~moaxy.benchmark.configs.COMPARISON_MODELS`).
* ``--configs``: a comma-separated list of
  :class:`~moaxy.benchmark.configs.ConfigVariant` values. The CLI
  accepts the four short aliases the live benchmark uses
  (``baseline``, ``reflection``, ``advisor``, ``both``) and
  maps them to the canonical :class:`ConfigVariant` members
  (``BASELINE``, ``REFLECTION_ONLY``, ``ADVISOR_ONLY``,
  ``BOTH``). The full enum ``.value`` strings
  (``reflection_only``, ``advisor_only``) are also accepted for
  symmetry.
* ``--output``: the directory the CLI writes the results to.
  The directory is created if it does not exist. The CLI writes
  ``<output>/results.json`` and ``<output>/report.md``.
* ``--fake-adapter``: when set, the runner uses the hermetic
  transport. When unset, the runner reads
  ``OPENROUTER_API_KEY`` from the environment and POSTs to the
  real OpenRouter. A missing key causes the CLI to exit non-zero
  with a clear error message.

Exit codes
----------

* ``0`` — the sweep completed and the output files were written.
* ``1`` — argument error (unknown model, unknown config, missing
  ``OPENROUTER_API_KEY`` when ``--fake-adapter`` is unset, etc.).
* ``2`` — runtime error (the runner raised an unhandled exception;
  the live benchmark failed a cell).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Final

from moaxy.benchmark.configs import (
    COMPARISON_MODELS,
    MODEL_ALIASES,
    ConfigVariant,
)
from moaxy.benchmark.harness import BenchmarkRunner, CellResult
from moaxy.benchmark.prompts import PROMPT_SET
from moaxy.benchmark.report import MarkdownReportGenerator

logger = logging.getLogger(__name__)


# The mapping from the user-friendly short config name to the
# canonical :class:`ConfigVariant` member. The live benchmark CLI
# accepts the short aliases (``baseline``, ``reflection``,
# ``advisor``, ``both``, ``reflection_fresh``) so the
# on-the-wire command line is short and readable; the table also
# accepts the canonical enum values (``baseline``,
# ``reflection_only``, ``advisor_only``, ``both``,
# ``reflection_fresh``) for symmetry with the
# ``ConfigVariant.value`` strings.
_CONFIG_NAME_TO_VARIANT: Final[dict[str, ConfigVariant]] = {
    "baseline": ConfigVariant.BASELINE,
    "reflection": ConfigVariant.REFLECTION_ONLY,
    "reflection_only": ConfigVariant.REFLECTION_ONLY,
    "advisor": ConfigVariant.ADVISOR_ONLY,
    "advisor_only": ConfigVariant.ADVISOR_ONLY,
    "both": ConfigVariant.BOTH,
    "reflection_fresh": ConfigVariant.REFLECTION_FRESH,
    "fresh": ConfigVariant.REFLECTION_FRESH,
}
"""User-friendly config name → :class:`ConfigVariant` mapping.

The short aliases (``baseline``, ``reflection``, ``advisor``,
``both``, ``reflection_fresh`` / ``fresh``) are the
contract-pinned CLI surface; the canonical
``ConfigVariant.value`` strings are accepted for symmetry with
internal code that already uses the canonical names.
"""


# The default path the CLI uses when ``--output`` is unset. The
# default lands in the canonical ``.benchmarks/results/`` tree so
# the live benchmark and the hermetic test can co-exist without
# overwriting each other.
_DEFAULT_OUTPUT: Final[str] = ".benchmarks/results/m7-run"
"""The default ``--output`` directory when the user does not set one."""


# The canonical on-disk path of the M7 live benchmark JSON. The
# M8 report generator consults this file (when present) to
# populate the ``Δ vs M7`` column and the ``M8 vs M7 Delta``
# section. The path is a module-level constant so a future
# relocation of the M7 baseline is caught by a single edit.
_M7_LIVE_BASELINE_PATH: Final[str] = ".benchmarks/results/m7-live-run.json"
"""The canonical on-disk path of the M7 live benchmark JSON.

The M8 live run reads this file (when present) to populate
the ``Δ vs M7`` column in the per-cell table and the
``M8 vs M7 Delta`` section in the report. When the file is
absent (the hermetic path, or an M8 sweep that runs before
the M7 baseline is committed), the report is rendered
without the delta surface — the M7-only contract
(VAL-BENCH-009) is unaffected.
"""


# The exit codes the CLI uses. The constants are module-level
# so a future test (or a wrapper script) can reference them
# without re-implementing the magic numbers.
_EXIT_OK: Final[int] = 0
"""The CLI's success exit code (the sweep completed)."""
_EXIT_ARG_ERROR: Final[int] = 1
"""The CLI's argument-error exit code (bad model, bad config, etc.)."""
_EXIT_RUNTIME_ERROR: Final[int] = 2
"""The CLI's runtime-error exit code (the runner raised)."""


def _load_m7_baseline(path: str = _M7_LIVE_BASELINE_PATH) -> dict[str, float] | None:
    """Load the M7 live baseline dict from ``path`` (if present).

    The function is a pure helper that reads the M7 live
    benchmark JSON and converts it into the composite-keyed
    ``dict[str, float]`` the
    :class:`~moaxy.benchmark.report.MarkdownReportGenerator`
    consumes. The composite key is the same one the
    generator builds (:func:`_baseline_key` in
    :mod:`moaxy.benchmark.report`): ``"<model>:<variant.value>"``.

    The function is deliberately defensive: a missing file
    returns ``None`` (the report renders without the delta
    surface); a malformed file returns ``None`` and logs a
    warning (a worker running the M8 sweep before the M7
    baseline is regenerated sees a clean, no-delta report
    rather than a crash). The function is the single
    integration point between the M7 baseline JSON and the
    M8 report's delta surface.

    Args:
        path: The on-disk path of the M7 baseline JSON.
            Defaults to the canonical
            :data:`_M7_LIVE_BASELINE_PATH`.

    Returns:
        A ``dict[str, float]`` mapping the composite
        ``"<model>:<variant.value>"`` key to the M7
        baseline mean quality, or ``None`` when the file
        is absent or malformed. The contract pins the
        returned dict's keys to the four M7 variants;
        the M8 ``REFLECTION_FRESH`` variant has no M7
        counterpart and is therefore absent from the
        dict (the report renders ``"N/A (new variant)"``
        for those rows).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "m7 baseline at %s could not be loaded: %s; "
            "the M8 report will render without the delta surface",
            path,
            exc,
        )
        return None
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list):
        return None
    baseline: dict[str, float] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        model = cell.get("model")
        variant = cell.get("variant")
        quality = cell.get("mean_quality")
        if not (isinstance(model, str) and isinstance(variant, str)):
            continue
        if not isinstance(quality, (int, float)):
            continue
        baseline[f"{model}:{variant}"] = float(quality)
    if not baseline:
        return None
    return baseline


def _parse_models(raw: str) -> list[str]:
    """Parse the ``--models`` argument into a list of aliases.

    The helper is a pure function: it splits the comma-separated
    string, trims whitespace from each entry, deduplicates the
    list (preserving first-occurrence order), and raises
    :class:`argparse.ArgumentTypeError` when an entry is not a
    key of :data:`MODEL_ALIASES`.

    Args:
        raw: The verbatim ``--models`` argument value (a single
            comma-separated string).

    Returns:
        The list of model aliases, in the order the user
        supplied them, with duplicates removed.

    Raises:
        argparse.ArgumentTypeError: an entry is empty, or is
            not a key of :data:`MODEL_ALIASES`. The error
            message names the offending entry and lists the
            known aliases.
    """
    if not raw or not raw.strip():
        raise argparse.ArgumentTypeError(
            "--models must be a non-empty comma-separated list"
        )
    aliases = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not aliases:
        raise argparse.ArgumentTypeError(
            "--models must contain at least one non-empty model alias"
        )
    unknown = [a for a in aliases if a not in MODEL_ALIASES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model alias(es) {unknown!r}; "
            f"known aliases: {sorted(MODEL_ALIASES)!r}"
        )
    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return deduped


def _parse_configs(raw: str) -> list[ConfigVariant]:
    """Parse the ``--configs`` argument into a list of variants.

    The helper splits the comma-separated string, maps each
    entry through :data:`_CONFIG_NAME_TO_VARIANT`, deduplicates
    the list (preserving first-occurrence order), and raises
    :class:`argparse.ArgumentTypeError` when an entry is not
    a known config name.

    Args:
        raw: The verbatim ``--configs`` argument value.

    Returns:
        The list of :class:`ConfigVariant` values, in the
        order the user supplied them, with duplicates removed.

    Raises:
        argparse.ArgumentTypeError: an entry is empty or
            not a key of :data:`_CONFIG_NAME_TO_VARIANT`.
    """
    if not raw or not raw.strip():
        raise argparse.ArgumentTypeError(
            "--configs must be a non-empty comma-separated list"
        )
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        raise argparse.ArgumentTypeError(
            "--configs must contain at least one non-empty config name"
        )
    variants: list[ConfigVariant] = []
    for entry in entries:
        if entry not in _CONFIG_NAME_TO_VARIANT:
            raise argparse.ArgumentTypeError(
                f"unknown config name {entry!r}; "
                f"known names: {sorted(_CONFIG_NAME_TO_VARIANT)!r}"
            )
        variants.append(_CONFIG_NAME_TO_VARIANT[entry])
    seen: set[ConfigVariant] = set()
    deduped: list[ConfigVariant] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


def _build_parser() -> argparse.ArgumentParser:
    """Return the :class:`argparse.ArgumentParser` for the CLI.

    The helper is the single source of truth for the CLI's
    argument surface; ``main`` and the test class both call
    it. Pinning the parser in a single function keeps the
    ``--help`` output and the runtime argument parser in
    lockstep, and lets a future test assert on the parser
    directly without re-implementing the flag list.

    Returns:
        A configured :class:`argparse.ArgumentParser`. The
        caller is responsible for ``parse_args`` and the
        downstream error handling.
    """
    parser = argparse.ArgumentParser(
        prog="python -m moaxy.benchmark.run",
        description=(
            "Run the M7 benchmark harness against a configurable "
            "set of models and configuration variants. The CLI "
            "writes a JSON results file and a markdown report "
            "to the --output directory."
        ),
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        required=True,
        help=(
            "Comma-separated list of model aliases to sweep over. "
            f"Allowed values: {','.join(COMPARISON_MODELS)} "
            "(and any other key of MODEL_ALIASES)."
        ),
    )
    parser.add_argument(
        "--configs",
        type=_parse_configs,
        required=True,
        help=(
            "Comma-separated list of config variants. Allowed "
            "values: baseline,reflection,advisor,both (and the "
            "canonical ConfigVariant values baseline,reflection_only,"
            "advisor_only,both)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=(
            "Directory the CLI writes results.json and report.md to. "
            "Created if it does not exist. Defaults to "
            f"{_DEFAULT_OUTPUT}/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--fake-adapter",
        action="store_true",
        help=(
            "Use the hermetic in-process fake transport. No real "
            "OpenRouter network call; no OPENROUTER_API_KEY required."
        ),
    )
    return parser


def _append_timestamp_to_output(path: str) -> str:
    """Return ``path`` with a ``YYYYMMDD-HHMMSS`` suffix appended.

    The CLI's contract surface (``--output <dir>``) specifies a
    *path* the user supplies; the live benchmark in
    ``services.yaml`` always passes a path that already ends
    with the timestamp. The helper appends a timestamp only
    when the path does not already end with one, so a caller
    that pre-supplies a timestamped path (the canonical live
    command) is unchanged, and a caller that supplies a bare
    directory (``--output .benchmarks/results``) gets a
    timestamped subdirectory.

    Args:
        path: The user-supplied ``--output`` value.

    Returns:
        The path the CLI will use. When ``path`` already ends
        in a ``YYYYMMDD-HHMMSS/``-style suffix, the path is
        returned unchanged. Otherwise a timestamped subdirectory
        is appended.
    """
    import datetime as _dt

    # Detect an existing trailing timestamp segment so the
    # canonical live command (which already includes the
    # timestamp) is not double-suffixed.
    trailing = os.path.basename(path.rstrip("/"))
    has_timestamp = (
        len(trailing) == 15
        and trailing[8] == "-"
        and trailing[:8].isdigit()
        and trailing[9:].isdigit()
    )
    if has_timestamp:
        return path
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(path, f"m7-run-{ts}")


def _cell_result_to_dict(cell: CellResult) -> dict[str, object]:
    """Serialise a :class:`CellResult` to a JSON-friendly dict.

    The CLI writes the per-cell results to ``results.json`` as
    a JSON array of dicts (one per cell). The serialiser
    converts the dataclass to a plain ``dict`` so the JSON
    encoder can render it without a custom encoder; the
    :class:`ConfigVariant` enum is rendered via ``.value`` so
    the on-disk format is the human-friendly string
    (``"baseline"``) rather than the Python repr
    (``"<ConfigVariant.BASELINE: 'baseline'>"``).

    Args:
        cell: The cell result to serialise.

    Returns:
        A JSON-friendly dict with one key per
        :class:`CellResult` field. The ``prompts`` list is
        serialised via :func:`dataclasses.asdict` so the
        per-prompt details are also JSON-friendly.
    """
    data = asdict(cell)
    # ``asdict`` does not unwrap the ``ConfigVariant`` enum;
    # we replace it with the canonical string value.
    data["variant"] = cell.variant.value
    return data


def _write_results_json(
    cells: list[CellResult], path: str
) -> None:
    """Write the cell results to ``path`` as a JSON file.

    The file format is a single JSON object with two keys:

    * ``cells`` — the list of serialised :class:`CellResult`
      dicts (one per ``(model, variant)`` cell).
    * ``metadata`` — a small dict with the ``model_count``,
      ``variant_count``, and ``prompt_count`` the runner was
      configured with. The metadata lets a downstream
      consumer sanity-check the JSON without re-deriving the
      counts from the cells.

    The function is the single source of truth for the JSON
    schema; a future test that pins the schema (e.g. the
    :class:`TestCLIEntryPoint` test) reads the file through
    this function's output.

    Args:
        cells: The cell results to serialise.
        path: The on-disk path of the JSON file.
    """
    payload = {
        "cells": [_cell_result_to_dict(c) for c in cells],
        "metadata": {
            "model_count": len({c.model for c in cells}),
            "variant_count": len({c.variant for c in cells}),
            "prompt_count": (
                cells[0].prompt_count if cells else 0
            ),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


async def _run_sweep(
    *,
    models: list[str],
    variants: list[ConfigVariant],
    fake_adapter: bool,
) -> list[CellResult]:
    """Execute the full M7 sweep and return the cell results.

    The helper wraps the :class:`BenchmarkRunner` in a single
    ``await runner.execute()`` call. Splitting the wrap into
    its own coroutine lets the CLI's ``main`` function stay
    purely synchronous (the ``asyncio.run`` boundary lives
    here, not in ``main``).

    Args:
        models: The model aliases to sweep over. Each entry
            is a key of :data:`MODEL_ALIASES`.
        variants: The :class:`ConfigVariant` values to sweep
            over.
        fake_adapter: When ``True``, the runner uses the
            hermetic in-process transport. When ``False``,
            the runner reads ``OPENROUTER_API_KEY`` from the
            environment and POSTs to the live OpenRouter.

    Returns:
        The list of :class:`CellResult` objects the runner
        produced, in the runner's canonical ordering
        (variants vary fastest).
    """
    runner = BenchmarkRunner(
        models=models,
        config_variants=variants,
        prompts=PROMPT_SET,
        fake_adapter=fake_adapter,
    )
    return await runner.execute()


def _validate_runtime(fake_adapter: bool) -> None:
    """Validate the runtime prerequisites the live path needs.

    The hermetic path (``fake_adapter=True``) has no
    prerequisites beyond the in-process transport. The live
    path requires ``OPENROUTER_API_KEY`` to be set in the
    environment; without it the OpenRouter adapter raises at
    construction time with a less-than-helpful error. The
    helper surfaces a clear error early so the CLI's exit
    code is a clean 1 (argument error) rather than a 2
    (runtime error) when the user forgot to export the key.

    Args:
        fake_adapter: Whether the CLI is in hermetic mode.

    Raises:
        SystemExit: the live path was requested and
            ``OPENROUTER_API_KEY`` is not set. The error
            message is printed to stderr and the CLI exits
            with :data:`_EXIT_ARG_ERROR`.
    """
    if fake_adapter:
        return
    if "OPENROUTER_API_KEY" not in os.environ:
        print(
            "error: --fake-adapter is not set and "
            "OPENROUTER_API_KEY is not exported; either pass "
            "--fake-adapter for a hermetic run or export the "
            "OpenRouter API key in the shell environment",
            file=sys.stderr,
        )
        raise SystemExit(_EXIT_ARG_ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI's main entry point.

    The function parses the command-line arguments, validates
    the runtime prerequisites, instantiates the
    :class:`BenchmarkRunner`, executes the full M7 sweep,
    writes ``<output>/results.json`` and ``<output>/report.md``,
    and returns the exit code.

    The function is intentionally synchronous from the
    caller's perspective: it uses :func:`asyncio.run` to drive
    the runner's coroutine, so a wrapper script (or the
    contract-pinned :class:`TestCLIEntryPoint` test) can
    invoke ``main`` directly from a synchronous context.

    Args:
        argv: The argument vector to parse. When ``None``,
            :mod:`argparse` defaults to ``sys.argv[1:]``
            (the standard CLI behaviour). Tests pass an
            explicit list to drive the function in-process.

    Returns:
        An integer exit code: :data:`_EXIT_OK` on success,
        :data:`_EXIT_ARG_ERROR` on argument / environment
        error, or :data:`_EXIT_RUNTIME_ERROR` on an
        unhandled exception in the runner.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # ``argparse`` calls ``sys.exit`` on argument
        # errors; we re-raise so the test harness can
        # observe the exit code, and we return the
        # underlying code so a direct ``main(...)`` call
        # also surfaces the right value.
        return int(exc.code) if exc.code is not None else _EXIT_ARG_ERROR
    try:
        _validate_runtime(fake_adapter=bool(args.fake_adapter))
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else _EXIT_ARG_ERROR
    output_dir = _append_timestamp_to_output(str(args.output))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(
            f"error: failed to create output directory {output_dir!r}: "
            f"{exc}",
            file=sys.stderr,
        )
        return _EXIT_ARG_ERROR
    results_json_path = os.path.join(output_dir, "results.json")
    report_md_path = os.path.join(output_dir, "report.md")
    try:
        cells = asyncio.run(
            _run_sweep(
                models=list(args.models),
                variants=list(args.configs),
                fake_adapter=bool(args.fake_adapter),
            )
        )
    except Exception as exc:  # noqa: BLE001
        # The runner is contract-pinned to never raise, so
        # the only way to reach this branch is an unexpected
        # exception in a third-party dependency (httpx,
        # uvicorn, the OpenRouter adapter, etc.). The CLI
        # surfaces a clean error and a non-zero exit code
        # so a wrapper script can detect the failure.
        print(
            f"error: benchmark sweep failed: {exc}",
            file=sys.stderr,
        )
        logger.exception("benchmark CLI: unhandled exception in sweep")
        return _EXIT_RUNTIME_ERROR
    try:
        _write_results_json(cells, results_json_path)
        m7_baseline = _load_m7_baseline()
        if m7_baseline is not None:
            logger.info(
                "loaded M7 baseline (%d cells) from %s; "
                "the M8 report will include the Δ vs M7 column "
                "and the M8 vs M7 Delta section",
                len(m7_baseline),
                _M7_LIVE_BASELINE_PATH,
            )
        report_text = MarkdownReportGenerator(
            cells, m7_baseline=m7_baseline
        ).render()
        with open(report_md_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)
    except OSError as exc:
        print(
            f"error: failed to write output files: {exc}",
            file=sys.stderr,
        )
        return _EXIT_RUNTIME_ERROR
    print(
        f"wrote {len(cells)} cells to {results_json_path}\n"
        f"wrote markdown report to {report_md_path}"
    )
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
