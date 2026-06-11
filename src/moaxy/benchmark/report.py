"""M7 benchmark markdown report generator.

The :mod:`moaxy.benchmark.report` module owns the
:class:`MarkdownReportGenerator`, the public surface that turns a
list of :class:`~moaxy.benchmark.harness.CellResult` objects into
a human-readable markdown report. The generator is the canonical
sister of the :class:`~moaxy.benchmark.harness.BenchmarkRunner`:
the runner produces :class:`CellResult` objects; the generator
turns them into the report the live benchmark CLI commits to
``.benchmarks/results/m7-live-report.md``.

The contract (VAL-BENCH-009) requires the rendered report to
contain five sections:

1. **Per-cell table** — one row per ``(model, variant)`` cell with
   the four summary statistics: mean quality, mean latency, mean
   tokens, pass-rate. Eight rows in the canonical M7 sweep
   (``len(COMPARISON_MODELS) * len(ConfigVariant) = 2 * 4 = 8``).
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

from moaxy.benchmark.configs import COMPARISON_MODELS
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

# The per-cell table column order. The order matches the
# contract's column list (model, variant, mean quality,
# mean latency, mean tokens, pass-rate); future edits
# to the column order must update the contract pins in
# the test class.
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


def _render_per_cell_table(
    cell_results: list[CellResult],
) -> str:
    """Render the per-cell results section.

    The function builds the standard six-column
    markdown table (model, variant, mean quality,
    mean latency, mean tokens, pass-rate) and prepends
    the section header. The number of data rows
    equals the number of cell results; the contract
    pins ``len(rows) == len(cell_results)`` and the
    canonical M7 sweep is eight rows.

    Args:
        cell_results: The cell results to render.

    Returns:
        The rendered markdown string (header + table)
        with a trailing blank line so the next
        section starts on its own paragraph.
    """
    rows: list[list[str]] = [list(_TABLE_COLUMNS)]
    for cell in cell_results:
        rows.append(
            [
                cell.model,
                cell.variant.value,
                _format_float(cell.mean_quality),
                _format_float(cell.mean_latency_ms),
                _format_float(cell.mean_tokens),
                _format_float(cell.pass_rate),
            ]
        )
    return f"{_SECTION_PER_CELL}\n\n{_format_table(rows)}\n"


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
        self, cell_results: list[CellResult] | tuple[CellResult, ...]
    ) -> None:
        self.cell_results: list[CellResult] = list(cell_results)

    def render(self) -> str:
        """Render the full markdown report.

        The function concatenates the five contract-
        pinned sections (per-cell table, best
        configuration per model, best model per
        configuration, cost-quality scatter, raw data
        appendix) with a blank line between each
        section. The returned string is valid
        markdown: every section starts with a level-2
        header (``##``), every table is the standard
        pipe table format, and there are no
        trailing-whitespace or syntax errors.

        The function is pure: it does not mutate
        ``self.cell_results`` and it does not have
        any side effects (no prints, no file I/O).

        Returns:
            A markdown string with all five
            contract-pinned sections in the
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
            _render_per_cell_table(self.cell_results),
            _render_best_per_model(self.cell_results),
            _render_best_per_config(self.cell_results),
            _render_cost_quality_scatter(self.cell_results),
            _render_raw_data_appendix(self.cell_results),
        ]
        return "\n".join(sections)


__all__ = ["MarkdownReportGenerator"]
