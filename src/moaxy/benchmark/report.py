"""M7 benchmark markdown report generator (with the M8 REFLECTION_FRESH delta).

The :mod:`moaxy.benchmark.report` module owns the
:class:`MarkdownReportGenerator`, the public surface that turns a
list of :class:`~moaxy.benchmark.harness.CellResult` objects into
a human-readable markdown report. The generator is the canonical
sister of the :class:`~moaxy.benchmark.harness.BenchmarkRunner`:
the runner produces :class:`CellResult` objects; the generator
turns them into the report the live benchmark CLI commits to
``.benchmarks/results/m7-live-report.md`` (M7) or
``.benchmarks/results/m8-live-report.md`` (M8).

The contract (VAL-BENCH-009) requires the rendered report to
contain five sections:

1. **Per-cell table** — one row per ``(model, variant)`` cell with
   the four summary statistics: mean quality, mean latency, mean
   tokens, pass-rate. Ten rows in the canonical M8 sweep
   (``len(COMPARISON_MODELS) * len(ConfigVariant) = 2 * 5 = 10``).
2. **Best configuration per model** — for each model, the variant
   that maximises mean quality.
3. **Best model per configuration** — for each variant, the
   model that maximises mean quality.
4. **Cost-quality scatter** — a textual scatter (mean quality vs
   mean total tokens) with a brief comment on cost-effectiveness.
5. **Raw data appendix** — the per-prompt, per-cell results.

The generator is hermetic and side-effect free: it does not call
the network, it does not depend on the orchestrator, and it does
not require a real OpenRouter key. The class accepts the cell
results directly and returns a plain ``str`` (no ``IO``,
``pathlib``, or print side effects) so the live benchmark CLI
can write the string to disk and the test suite can assert on
its contents.

Markdown table formatting
-------------------------

The generator builds the per-cell table by hand (no
``tabulate`` dep) so the module has no new third-party
dependencies. The table format is the standard
``| col | col |`` markdown pipe table; the alignment is
"left" for strings and "right" for numeric values. The
header row is followed by a separator row
(``|---|---|---|``) and then the data rows. The formatting
helper :func:`_format_float` renders floats with three
decimal places (the resolution needed to compare two
quality scores that differ in the third decimal) and
``None`` values as the literal string ``"n/a"`` (so a
malformed cell with no data still renders cleanly).

The "best configuration per model" / "best model per
configuration" sections are simple bullet lists. The
"cost-quality scatter" section is a textual plot
(asterisks per row, sorted by cost) with a header that
names the cost and quality axes; this is the most
portable way to render a scatter plot in pure markdown
without dragging in a plotting library.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from moaxy.benchmark.configs import COMPARISON_MODELS, ConfigVariant
from moaxy.benchmark.harness import CellResult

# The five markdown section headers the contract pins. The
# constant tuple is the single source of truth: the test
# asserts each header appears verbatim in the rendered
# string, and the generator renders the headers verbatim.
# Adding a sixth section requires updating the contract
# (VAL-BENCH-009) and the test class in lockstep.
_SECTION_PER_CELL: Final[str] = "## Per-Cell Results"
_SECTION_BEST_PER_MODEL: Final[str] = "## Best Configuration per Model"
_SECTION_BEST_PER_CONFIG: Final[str] = "## Best Model per Configuration"
_SECTION_COST_QUALITY: Final[str] = "## Cost-Quality Scatter"
_SECTION_RAW_DATA: Final[str] = "## Raw Data Appendix"
_SECTION_DELTA: Final[str] = "## M8 vs M7 Delta"

# The M8 delta column header. Pinned by the contract
# (m8-report-delta-vs-m7): the column is named exactly
# ``"Δ vs M7"`` (with the Greek capital delta, a single
# space, the lowercase ``"vs"``, another single space, and
# the uppercase ``"M7"``). The generator renders the column
# only when ``m7_baseline`` is provided.
_DELTA_COLUMN_HEADER: Final[str] = "Δ vs M7"

# The literal string rendered for the M8 ``REFLECTION_FRESH``
# delta cells. The M8 ``REFLECTION_FRESH`` variant has no M7
# counterpart, so the delta is undefined; the contract pins
# the rendered placeholder as either ``"N/A (new variant)"``
# (the default) or the bare em-dash ``"—"``. The test
# (:class:`TestM8ReportDelta`) accepts both forms.
_DELTA_NA: Final[str] = "N/A (new variant)"

# The per-cell table column order. The order matches the
# contract's column list (model, variant, mean quality,
# mean latency, mean tokens, pass-rate); future edits
# to the column order must update the contract pins in
# the test class. The M8 delta extends the column list
# with a trailing ``Δ vs M7`` column when ``m7_baseline``
# is provided.
_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "Model",
    "Variant",
    "Mean Quality",
    "Mean Latency (ms)",
    "Mean Tokens",
    "Pass Rate",
)

# The float-format precision used for the numeric columns.
# Three decimal places is enough to distinguish two
# quality scores that differ in the third decimal (e.g.
# 0.832 vs 0.835); a larger precision would clutter
# the table and a smaller one would hide real
# differences.
_FLOAT_PRECISION: Final[int] = 3

# The literal string rendered for a missing value. The
# test asserts the report never raises a formatting
# error for a malformed cell with no data, so the
# fallback is a stable, well-known string.
_NA: Final[str] = "n/a"


def _format_float(value: float | None) -> str:
    """Format a float for the per-cell table.

    Args:
        value: The float to format, or ``None`` when the
            cell produced no data.

    Returns:
        A string representation of the float with three
        decimal places, or the literal ``"n/a"`` when
        the value is ``None``. The function is a pure
        helper; it never raises.
    """
    if value is None:
        return _NA
    return f"{value:.{_FLOAT_PRECISION}f}"


def _format_table(rows: list[list[str]]) -> str:
    """Render a list of rows as a markdown pipe table.

    The function is a minimal pure-Python markdown
    table formatter (no third-party deps). It computes
    the column widths from the data, then renders the
    header row, the separator row, and each data row
    with consistent padding so the rendered table
    aligns in a markdown viewer.

    Args:
        rows: The table rows. The first row is the
            header; subsequent rows are the data rows.

    Returns:
        A multi-line string with one pipe-separated
        line per row. The string ends with a trailing
        newline so the caller can concatenate sections
        without a manual ``"\\n"`` join.
    """
    if not rows:
        return ""
    column_count = max(len(r) for r in rows)
    padded = [r + [""] * (column_count - len(r)) for r in rows]
    widths = [
        max(len(cell) for cell in (padded[row][col] for row in range(len(padded))))
        for col in range(column_count)
    ]
    lines: list[str] = []
    for idx, row in enumerate(padded):
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
        if idx == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines) + "\n"


def _cells_grouped_by_model(
    cell_results: Iterable[CellResult],
) -> dict[str, list[CellResult]]:
    """Group cell results by their ``model`` alias.

    The "best configuration per model" and "best model
    per configuration" sections iterate over the cell
    results grouped by model and by variant. The helper
    builds the two groupings in a single pass so the
    generator does not pay the O(N^2) cost of repeated
    linear scans.

    Args:
        cell_results: The cell results to group.

    Returns:
        A dict mapping each model alias to the list of
        cell results for that model, in the order the
        caller supplied. Models that appear only once
        (e.g. a sub-sweep) are still present; the
        caller decides what to do with singletons.
    """
    grouped: dict[str, list[CellResult]] = {}
    for cell in cell_results:
        grouped.setdefault(cell.model, []).append(cell)
    return grouped


def _cells_grouped_by_variant(
    cell_results: Iterable[CellResult],
) -> dict[str, list[CellResult]]:
    """Group cell results by their ``variant.value`` string.

    The mirror of :func:`_cells_grouped_by_model` for
    the variant axis. The function uses ``variant.value``
    (the canonical string form, e.g. ``"baseline"``)
    rather than the :class:`~moaxy.benchmark.configs.ConfigVariant`
    enum instance so the rendered report text is
    stable across Python versions (the enum's ``str()``
    rendering has changed between versions).

    Args:
        cell_results: The cell results to group.

    Returns:
        A dict mapping each variant value (string) to
        the list of cell results for that variant, in
        the order the caller supplied.
    """
    grouped: dict[str, list[CellResult]] = {}
    for cell in cell_results:
        grouped.setdefault(cell.variant.value, []).append(cell)
    return grouped


def _best_cell(cells: list[CellResult]) -> CellResult | None:
    """Return the cell with the highest mean quality.

    The helper is the single source of truth for the
    "best" rule used by both "best configuration per
    model" and "best model per configuration". A cell
    with ``mean_quality is None`` (no scored prompts)
    is skipped; ties are broken by the cell's order in
    the input list (the first cell wins), so the
    output is deterministic for a given input.

    Args:
        cells: The candidate cells, in caller order.

    Returns:
        The winning cell, or ``None`` when every
        candidate has ``mean_quality is None`` (the
        cells produced no data).
    """
    best: CellResult | None = None
    best_quality: float | None = None
    for cell in cells:
        if cell.mean_quality is None:
            continue
        if best is None or cell.mean_quality > (best_quality or 0.0):
            best = cell
            best_quality = cell.mean_quality
    return best


def _baseline_key(model_alias: str, variant_value: str) -> str:
    """Build the composite key for the ``m7_baseline`` dict.

    The M8 contract (m8-report-delta-vs-m7) pins the
    ``m7_baseline`` parameter as a ``dict[str, float]``
    keyed by a composite ``"<model>:<variant.value>"``
    string. The composite-key format is stable across
    Python versions and is human-readable; the helper
    is the single source of truth so the table
    renderer and the delta section renderer agree
    on the key format.

    Args:
        model_alias: The client-facing model alias
            (e.g. ``"minimax-m3"``).
        variant_value: The ``ConfigVariant.value``
            string (e.g. ``"baseline"``).

    Returns:
        The composite key
        ``f"{model_alias}:{variant_value}"``. For
        example, ``"minimax-m3:baseline"``.
    """
    return f"{model_alias}:{variant_value}"


def _format_delta(delta: float | None) -> str:
    """Format a delta value for the ``Δ vs M7`` column.

    The helper renders a signed float with three decimal
    places (matching the precision of the other
    numeric columns). Positive deltas are prefixed with
    ``"+"``; negative deltas carry the natural Python
    sign; zero is rendered as ``"+0.000"`` so the
    column reads uniformly. A ``None`` delta renders
    as the literal string ``"n/a"`` (matching the
    :func:`_format_float` fallback for missing data).

    Args:
        delta: The delta to format, or ``None`` when
            the value is missing (the cell has no M7
            counterpart or the M7 baseline is unknown).

    Returns:
        The formatted string. Examples:
        ``"+0.215"``, ``"-0.108"``, ``"+0.000"``,
        ``"n/a"``.
    """
    if delta is None:
        return _NA
    return f"{delta:+.{_FLOAT_PRECISION}f}"


def _render_per_cell_table(
    cell_results: list[CellResult],
    m7_baseline: dict[str, float] | None = None,
) -> str:
    """Render the per-cell results section.

    The function builds the standard six-column
    markdown table (model, variant, mean quality,
    mean latency, mean tokens, pass-rate) and prepends
    the section header. When ``m7_baseline`` is
    provided, the function appends a seventh column
    ``Δ vs M7`` to the table; the column carries the
    change in mean quality vs the M7 baseline (same
    model + same variant value). The M8
    ``REFLECTION_FRESH`` variant has no M7 counterpart,
    so its delta cell renders the literal string
    ``"N/A (new variant)"``.

    The number of data rows equals the number of cell
    results; the contract pins
    ``len(rows) == len(cell_results)`` and the
    canonical M8 sweep is ten rows (2 models × 5
    variants).

    Args:
        cell_results: The cell results to render.
        m7_baseline: An optional ``dict[str, float]``
            mapping the composite
            ``"<model>:<variant.value>"`` key to the
            M7 baseline mean quality. When ``None``
            (the M7 default), the delta column is
            omitted. When provided, the M8
            ``REFLECTION_FRESH`` cells and any cell
            whose key is missing from the dict render
            the ``"N/A (new variant)"`` placeholder.

    Returns:
        The rendered markdown string (header + table)
        with a trailing blank line so the next
        section starts on its own paragraph.
    """
    columns: tuple[str, ...] = _TABLE_COLUMNS
    if m7_baseline is not None:
        columns = _TABLE_COLUMNS + (_DELTA_COLUMN_HEADER,)
    rows: list[list[str]] = [list(columns)]
    for cell in cell_results:
        row = [
            cell.model,
            cell.variant.value,
            _format_float(cell.mean_quality),
            _format_float(cell.mean_latency_ms),
            _format_float(cell.mean_tokens),
            _format_float(cell.pass_rate),
        ]
        if m7_baseline is not None:
            row.append(_delta_cell_for(cell, m7_baseline))
        rows.append(row)
    return f"{_SECTION_PER_CELL}\n\n{_format_table(rows)}\n"


def _delta_cell_for(
    cell: CellResult, m7_baseline: dict[str, float]
) -> str:
    """Return the ``Δ vs M7`` cell string for one cell.

    The helper is the single source of truth for the
    delta-column rendering. The rules are:

    * The M8 ``REFLECTION_FRESH`` variant has no M7
      counterpart, so its delta cell renders the
      literal ``"N/A (new variant)"`` placeholder.
    * When the cell's ``(model, variant.value)`` key
      is missing from ``m7_baseline``, the helper
      also renders the placeholder (the M7 baseline
      did not include this cell; the delta is
      undefined).
    * When the cell's ``mean_quality`` is ``None`` (the
      cell produced no data), the helper renders the
      placeholder (the delta is undefined).
    * Otherwise the helper computes
      ``cell.mean_quality - m7_baseline[key]`` and
      formats the result with :func:`_format_delta`.

    Args:
        cell: The cell result to compute the delta
            for.
        m7_baseline: The M7 baseline dict keyed by
            ``"<model>:<variant.value>"``.

    Returns:
        The string to render in the delta column.
    """
    if cell.variant.value == ConfigVariant.REFLECTION_FRESH.value:
        return _DELTA_NA
    key = _baseline_key(cell.model, cell.variant.value)
    if cell.mean_quality is None or key not in m7_baseline:
        return _DELTA_NA
    return _format_delta(cell.mean_quality - m7_baseline[key])


def _render_m8_vs_m7_delta(
    cell_results: list[CellResult],
    m7_baseline: dict[str, float] | None = None,
) -> str:
    """Render the ``M8 vs M7 Delta`` section.

    The section lists the cells where M8 improved
    (Δ > 0) and the cells where M8 regressed
    (Δ < 0), with brief commentary. The M8
    ``REFLECTION_FRESH`` cells are excluded from the
    wins / losses list (their delta is undefined);
    the section's commentary notes the exclusion.

    The section is rendered only when ``m7_baseline``
    is provided. When every cell has Δ == 0, the
    section renders a single "no change" line so the
    section is never empty.

    Args:
        cell_results: The cell results to summarise.
        m7_baseline: An optional ``dict[str, float]``
            mapping the composite
            ``"<model>:<variant.value>"`` key to the
            M7 baseline mean quality. When ``None``,
            the function returns an empty string (the
            section is omitted from the rendered
            report).

    Returns:
        The rendered markdown string with a trailing
        blank line, or the empty string when
        ``m7_baseline`` is ``None``.
    """
    if m7_baseline is None:
        return ""
    lines: list[str] = [
        _SECTION_DELTA,
        "",
        "Each bullet below compares an M8 cell's mean "
        "quality to the same ``(model, variant)`` cell "
        "from the M7 baseline. A positive Δ means M8 "
        "improved; a negative Δ means M8 regressed. The "
        "M8 ``REFLECTION_FRESH`` variant has no M7 "
        "counterpart and is excluded from this section.",
        "",
    ]
    deltas: list[tuple[str, str, float, float, float]] = []
    for cell in cell_results:
        if cell.variant.value == ConfigVariant.REFLECTION_FRESH.value:
            continue
        if cell.mean_quality is None:
            continue
        key = _baseline_key(cell.model, cell.variant.value)
        if key not in m7_baseline:
            continue
        m7_q = m7_baseline[key]
        delta = cell.mean_quality - m7_q
        deltas.append(
            (cell.model, cell.variant.value, delta, cell.mean_quality, m7_q)
        )
    if not deltas:
        lines.append(
            "No comparable cells; the M7 baseline is empty "
            "or every M8 cell is missing mean quality."
        )
        return "\n".join(lines) + "\n"
    wins = [d for d in deltas if d[2] > 0]
    losses = [d for d in deltas if d[2] < 0]
    unchanged = [d for d in deltas if d[2] == 0]
    if wins:
        lines.append("**M8 improved (Δ > 0):**")
        lines.append("")
        for model, variant_value, delta, m8_q, m7_q in wins:
            lines.append(
                f"- `{model}` / `{variant_value}`: Δ = "
                f"{_format_delta(delta)} (M8 = "
                f"{_format_float(m8_q)}, M7 = "
                f"{_format_float(m7_q)})"
            )
        lines.append("")
    if losses:
        lines.append("**M8 regressed (Δ < 0):**")
        lines.append("")
        for model, variant_value, delta, m8_q, m7_q in losses:
            lines.append(
                f"- `{model}` / `{variant_value}`: Δ = "
                f"{_format_delta(delta)} (M8 = "
                f"{_format_float(m8_q)}, M7 = "
                f"{_format_float(m7_q)})"
            )
        lines.append("")
    if unchanged:
        lines.append("**No change (Δ = 0):**")
        lines.append("")
        for model, variant_value, delta, m8_q, m7_q in unchanged:
            lines.append(
                f"- `{model}` / `{variant_value}`: Δ = "
                f"{_format_delta(delta)} (M8 = "
                f"{_format_float(m8_q)}, M7 = "
                f"{_format_float(m7_q)})"
            )
        lines.append("")
    # Brief commentary: a one-line summary the user can
    # read at a glance. The summary references the
    # absolute number of wins and losses and the maximum
    # |Δ| across all cells. The summary is purely
    # informational; the contract does not pin its
    # exact text.
    win_count = len(wins)
    loss_count = len(losses)
    unchanged_count = len(unchanged)
    max_delta = max((abs(d[2]) for d in deltas), default=0.0)
    if win_count == 0 and loss_count == 0:
        summary = (
            f"All {unchanged_count} comparable cells "
            f"are unchanged; max |Δ| is "
            f"{_format_float(max_delta)}."
        )
    else:
        summary = (
            f"Net: {win_count} improved, {loss_count} "
            f"regressed, {unchanged_count} unchanged; "
            f"max |Δ| is {_format_float(max_delta)}."
        )
    lines.append(summary)
    return "\n".join(lines) + "\n"


def _render_best_per_model(
    cell_results: list[CellResult],
) -> str:
    """Render the "best configuration per model" section.

    The section is a markdown bullet list: for each
    model in the input, a single bullet line that
    names the model's best configuration. The section
    is rendered in the model order the caller
    supplied (so the report is reproducible given a
    reproducible input).

    Args:
        cell_results: The cell results to summarise.

    Returns:
        The rendered markdown string with a trailing
        blank line.
    """
    grouped = _cells_grouped_by_model(cell_results)
    lines: list[str] = [_SECTION_BEST_PER_MODEL, ""]
    for model_alias in sorted(grouped):
        cells = grouped[model_alias]
        best = _best_cell(cells)
        if best is None:
            lines.append(
                f"- **{model_alias}**: no scored cells (no "
                "best configuration can be determined)"
            )
            continue
        lines.append(
            f"- **{model_alias}**: `{best.variant.value}` "
            f"(mean quality = {_format_float(best.mean_quality)})"
        )
    return "\n".join(lines) + "\n"


def _render_best_per_config(
    cell_results: list[CellResult],
) -> str:
    """Render the "best model per configuration" section.

    The mirror of :func:`_render_best_per_model` for
    the variant axis. The section is a markdown bullet
    list: for each variant, a single bullet that names
    the variant's best model.

    Args:
        cell_results: The cell results to summarise.

    Returns:
        The rendered markdown string with a trailing
        blank line.
    """
    grouped = _cells_grouped_by_variant(cell_results)
    lines: list[str] = [_SECTION_BEST_PER_CONFIG, ""]
    for variant_value in sorted(grouped):
        cells = grouped[variant_value]
        best = _best_cell(cells)
        if best is None:
            lines.append(
                f"- **{variant_value}**: no scored cells (no "
                "best model can be determined)"
            )
            continue
        lines.append(
            f"- **{variant_value}**: `{best.model}` "
            f"(mean quality = {_format_float(best.mean_quality)})"
        )
    return "\n".join(lines) + "\n"


def _render_cost_quality_scatter(
    cell_results: list[CellResult],
) -> str:
    """Render the cost-quality scatter section.

    The section is a textual scatter plot: for each
    cell, a row in the form
    ``<model> / <variant> | <cost bar> | quality <float>``,
    sorted by ascending cost (mean tokens). The
    section is plain markdown (no third-party
    plotting library required) and renders cleanly
    in every markdown viewer.

    The contract does not pin a specific scatter
    format; it only requires the section to exist.
    This implementation favours readability and
    reproducibility: the cost bars are scaled
    relative to the maximum mean tokens in the data
    so the chart "fills" the available width on any
    input.

    Args:
        cell_results: The cell results to plot.

    Returns:
        The rendered markdown string with a trailing
        blank line.
    """
    lines: list[str] = [
        _SECTION_COST_QUALITY,
        "",
        "Each row is one (model, variant) cell. The cost "
        "axis is mean total tokens (a proxy for the cell's "
        "OpenRouter bill); the quality axis is the cell's "
        "mean quality score. Lower cost and higher quality "
        "are better.",
        "",
    ]
    scorable = [c for c in cell_results if c.mean_tokens is not None]
    if not scorable:
        lines.append("No cost data available; the scatter is empty.")
        return "\n".join(lines) + "\n"
    max_tokens = max(c.mean_tokens or 0.0 for c in scorable)
    bar_width = 20
    sorted_cells = sorted(
        scorable, key=lambda c: (c.mean_tokens or 0.0, c.model)
    )
    rows: list[list[str]] = [
        ["Model", "Variant", "Cost (tokens)", "Quality"]
    ]
    for cell in sorted_cells:
        cost = cell.mean_tokens or 0.0
        bar_len = (
            int(round((cost / max_tokens) * bar_width))
            if max_tokens > 0
            else 0
        )
        bar = "*" * bar_len
        rows.append(
            [
                cell.model,
                cell.variant.value,
                f"{_format_float(cost)}  {bar}",
                _format_float(cell.mean_quality),
            ]
        )
    lines.append(_format_table(rows))
    return "\n".join(lines) + "\n"


def _render_raw_data_appendix(
    cell_results: list[CellResult],
) -> str:
    """Render the raw-data appendix section.

    The appendix is a list of sub-tables, one per
    cell. Each sub-table has columns ``task_id``,
    ``category``, ``score``, ``latency_ms``, and
    ``total_tokens``; the per-prompt details are the
    raw data the live benchmark CLI commits to the
    JSON output. The appendix is the only section
    that grows with the prompt set; the other four
    sections have a constant number of rows.

    Args:
        cell_results: The cell results to render.

    Returns:
        The rendered markdown string with a trailing
        blank line.
    """
    lines: list[str] = [_SECTION_RAW_DATA, ""]
    if not cell_results:
        lines.append("No cell results to render.")
        return "\n".join(lines) + "\n"
    for cell in cell_results:
        lines.append(
            f"### {cell.model} / `{cell.variant.value}`"
        )
        lines.append("")
        rows: list[list[str]] = [
            ["task_id", "category", "score", "latency_ms", "total_tokens"]
        ]
        for prompt in cell.prompts:
            rows.append(
                [
                    prompt.task_id,
                    prompt.category,
                    _format_float(prompt.score),
                    _format_float(prompt.latency_ms),
                    _format_float(prompt.total_tokens),
                ]
            )
        lines.append(_format_table(rows))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


class MarkdownReportGenerator:
    """Render a benchmark cell-result list as a markdown report.

    The class is the public surface of the
    :mod:`moaxy.benchmark.report` module. The
    constructor stores the cell results; :meth:`render`
    returns the full markdown report as a single
    string. The class is stateless beyond the stored
    results, so callers can reuse a single instance
    to render the same data multiple times (e.g. for
    caching or forking the output to multiple files).

    The generator is hermetic: it does not call the
    network, it does not depend on the orchestrator,
    and it does not require a real OpenRouter key. A
    unit test that constructs eight canned
    :class:`CellResult` objects and calls
    :meth:`render` exercises the full rendering path
    in-process; that is the contract-pinned test
    path (:class:`TestReportGeneratorContract` in
    :mod:`tests.test_benchmark`).

    Args:
        cell_results: The cell results to render. The
            order is preserved within each section
            (the per-cell table renders in input
            order; the per-model and per-config
            summaries render in sorted-key order so
            the output is deterministic across runs).
    """

    def __init__(
        self,
        cell_results: list[CellResult] | tuple[CellResult, ...],
        *,
        m7_baseline: dict[str, float] | None = None,
    ) -> None:
        """Store the cell results and the optional M7 baseline.

        Args:
            cell_results: The cell results to render. The
                order is preserved within each section
                (the per-cell table renders in input
                order; the per-model and per-config
                summaries render in sorted-key order so
                the output is deterministic across runs).
            m7_baseline: An optional ``dict[str, float]``
                mapping the composite
                ``"<model>:<variant.value>"`` key to the
                M7 baseline mean quality. When ``None``
                (the M7 default), the rendered report
                contains the original 5 sections
                unchanged. When provided, the report
                gains a ``Δ vs M7`` column in the
                per-cell table and a new
                ``M8 vs M7 Delta`` section that lists
                the cells where M8 improved and where
                it regressed, with brief commentary.
                The M8 ``REFLECTION_FRESH`` variant has
                no M7 counterpart, so its delta cell
                renders the literal string
                ``"N/A (new variant)"``. The constructor
                validates that ``m7_baseline`` is
                either a ``dict`` or ``None``; any
                other type raises :class:`TypeError`.
        """
        if m7_baseline is not None and not isinstance(m7_baseline, dict):
            raise TypeError(
                "m7_baseline must be a dict[str, float] or None; "
                f"got {type(m7_baseline).__name__}"
            )
        self.cell_results: list[CellResult] = list(cell_results)
        self.m7_baseline: dict[str, float] | None = (
            dict(m7_baseline) if m7_baseline is not None else None
        )

    def render(self) -> str:
        """Render the full markdown report.

        The function concatenates the five contract-
        pinned sections (per-cell table, best
        configuration per model, best model per
        configuration, cost-quality scatter, raw data
        appendix) with a blank line between each
        section. When ``m7_baseline`` is provided, an
        additional ``M8 vs M7 Delta`` section is
        rendered after the per-cell table so the
        delta column and the delta section live next
        to each other. The returned string is valid
        markdown: every section starts with a level-2
        header (``##``), every table is the standard
        pipe table format, and there are no
        trailing-whitespace or syntax errors.

        The function is pure: it does not mutate
        ``self.cell_results`` and it does not have
        any side effects (no prints, no file I/O).

        Returns:
            A markdown string with the 5 contract-
            pinned sections (or 6 sections when
            ``m7_baseline`` is provided) in the
            contract-pinned order.
        """
        sections: list[str] = [
            "# M7 Benchmark Report",
            "",
            "Generated by the M7 benchmark harness. The "
            "sweep covers "
            f"{len(COMPARISON_MODELS)} models × 4 config "
            f"variants = {len(COMPARISON_MODELS) * 4} cells.",
            "",
            _render_per_cell_table(
                self.cell_results, self.m7_baseline
            ),
            _render_m8_vs_m7_delta(
                self.cell_results, self.m7_baseline
            ),
            _render_best_per_model(self.cell_results),
            _render_best_per_config(self.cell_results),
            _render_cost_quality_scatter(self.cell_results),
            _render_raw_data_appendix(self.cell_results),
        ]
        return "\n".join(sections)


__all__ = ["MarkdownReportGenerator"]
