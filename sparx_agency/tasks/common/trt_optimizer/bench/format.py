"""Small formatting primitives shared by the report and its Markdown renderer.

Three-significant-figure rounding, human parameter counts, a fixed-width table
and the duck-typed accessors that let any object with ``mean_ms``/``hz`` stand in
for a :class:`..bench.latency.LatencyStats`.

They live in their own module because both the data module and the renderer need
them, and importing a private name across modules to share it is how two
modules end up quietly disagreeing about how a number is rounded.
"""
from __future__ import annotations

#: Bytes per MiB, for the memory section.
MIB = 1 << 20


def _sig3(value):
    """Format a number to three significant figures, never in exponent form.

    Args:
        value: a float, an int, or None.

    Returns:
        A short string, or ``"-"`` for None. Magnitudes at or above 100 are
        printed as whole numbers so a wide table stays narrow.
    """
    if value is None:
        return "-"
    v = float(value)
    a = abs(v)
    if a == 0.0:
        return "0"
    if a >= 100.0:
        return "%.0f" % v
    if a >= 10.0:
        return "%.1f" % v
    if a >= 1.0:
        return "%.2f" % v
    return "%.3g" % v


def _fmt_params(count):
    """Format a parameter count compactly (``1.23B`` / ``86.4M`` / ``512``)."""
    if count is None:
        return "-"
    n = float(count)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return _sig3(n / scale) + suffix
    return "%d" % int(n)


def _fmt_cell(value):
    """Render any scalar for a table cell without lying about its type."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _sig3(value)
    return str(value)


def _table(headers, rows, right=()):
    """Build a padded Markdown table.

    Args:
        headers: column titles.
        rows: sequence of already-stringified cell sequences.
        right: indices of columns to right-align (numbers).

    Returns:
        A list of Markdown lines. Columns are widened to their content so the
        raw text stays readable without a renderer.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    rule = []
    for i, width in enumerate(widths):
        rule.append("-" * (width + 1) + ":" if i in right else "-" * (width + 2))
    lines.append("|" + "|".join(rule) + "|")
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _stats_value(stats, attr):
    """Read one duck-typed field off a LatencyStats-like object, or None."""
    if stats is None:
        return None
    value = getattr(stats, attr, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _hz(stats):
    """Throughput of a LatencyStats-like object, derived from ms if absent."""
    hz = _stats_value(stats, "hz")
    if hz is not None:
        return hz
    mean_ms = _stats_value(stats, "mean_ms")
    if mean_ms is None or mean_ms <= 0.0:
        return None
    return 1000.0 / mean_ms
