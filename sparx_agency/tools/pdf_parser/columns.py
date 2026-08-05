"""Finding columns by looking for the empty vertical stripes between them.

Used twice, at two scales. On a whole page it separates the left and right
columns of a two-column paper, which is what tells the region finder how wide a
figure may be. Inside a table region it separates the table's columns, which is
what turns a set of positioned words into cells.

The method is the same at both scales and needs no machine learning: mark every
horizontal position covered by any word, then look for runs of positions covered
by none. A run wide enough to be deliberate is a gutter; what lies between the
gutters is a column. Table gutters are narrower than page gutters, which is the
only difference between the two uses and is why the threshold is an argument.

This works because it measures the *typesetter's* whitespace rather than
guessing from the text, and it fails in exactly one predictable way: a table
whose columns are packed close enough that some row bridges the gutter reads as
one wide column. That shows up immediately when the parsed table is compared
with the rendered crop, which is why both are always produced.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from sparx_agency.tools.pdf_parser.geometry import BBox

BIN_PT = 0.5
"""Resolution of the occupancy scan, in points — about a tenth of a character."""

PAGE_GUTTER_PT = 18.0
"""Minimum empty width that counts as the gutter between page columns.

Two-column layouts in this literature leave 20-30 pt between columns, while the
widest inter-word space inside a justified line is under 15 pt. Eighteen sits in
the gap and does not have to be tuned per template.
"""

TABLE_GUTTER_PT = 8.0
"""Minimum empty width that counts as the gutter between table columns.

Lower than a page gutter because table columns are set tighter, and still above
the ordinary inter-word space at the font sizes used in tables.
"""

MIN_COLUMN_PT = 12.0
"""Narrowest span kept as a column. Below this it is a stray mark, not a column."""

MAX_BRIDGE_FRACTION = 0.2
"""Share of rows allowed to cross a gutter and have it still count as a gutter.

Without this, a table with a heading centred over two sub-columns — "BLEU" above
an EN-DE and an EN-FR column — has no stripe that is empty on *every* row, and
the two sub-columns merge into one. The spanning headings are always a small
minority of the rows, so allowing a fifth of them to bridge recovers the real
columns without inventing any.
"""


def _row_counts(
    rows: Sequence[Sequence[BBox]], x_min: float, x_max: float, bin_pt: float
) -> List[int]:
    """Count, per horizontal bin, how many rows have any text over it.

    Counting rows rather than boxes is what makes the bridging tolerance
    meaningful: one row with three words over a bin still counts once.
    """
    count = max(1, int((x_max - x_min) / bin_pt) + 1)
    totals = [0] * count
    for row in rows:
        touched = set()
        for box in row:
            start = max(0, int((box.x_min - x_min) / bin_pt))
            end = min(count - 1, int((box.x_max - x_min) / bin_pt))
            touched.update(range(start, end + 1))
        for index in touched:
            totals[index] += 1
    return totals


def detect_column_spans(
    boxes: Iterable[BBox],
    min_gap_pt: float = TABLE_GUTTER_PT,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    min_column_pt: float = MIN_COLUMN_PT,
    bin_pt: float = BIN_PT,
) -> List[Tuple[float, float]]:
    """Split a set of word boxes into columns along their empty vertical stripes.

    Every box is treated as its own row, so a gutter has to be empty of
    everything. This is the right test for page columns; for table columns use
    :func:`detect_column_spans_from_rows`, which tolerates spanning headings.

    Args:
        boxes: Word rectangles to scan.
        min_gap_pt: Empty width that counts as a gutter.
        x_min: Left bound of the scan; defaults to the leftmost box.
        x_max: Right bound; defaults to the rightmost box.
        min_column_pt: Narrowest span kept.
        bin_pt: Scan resolution.

    Returns:
        ``(left, right)`` spans in points, left to right. A single span means
        no gutter wide enough was found, which is the correct answer for a
        one-column page or a single-column list.
    """
    return detect_column_spans_from_rows(
        [[box] for box in boxes],
        min_gap_pt=min_gap_pt,
        x_min=x_min,
        x_max=x_max,
        min_column_pt=min_column_pt,
        bin_pt=bin_pt,
        max_bridge_fraction=0.0,
    )


def detect_column_spans_from_rows(
    rows: Sequence[Sequence[BBox]],
    min_gap_pt: float = TABLE_GUTTER_PT,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    min_column_pt: float = MIN_COLUMN_PT,
    bin_pt: float = BIN_PT,
    max_bridge_fraction: float = MAX_BRIDGE_FRACTION,
) -> List[Tuple[float, float]]:
    """Split rows of word boxes into columns, allowing a few rows to span gutters.

    Args:
        rows: One sequence of word rectangles per row.
        min_gap_pt: Empty width that counts as a gutter.
        x_min: Left bound of the scan; defaults to the leftmost box.
        x_max: Right bound; defaults to the rightmost box.
        min_column_pt: Narrowest span kept.
        bin_pt: Scan resolution.
        max_bridge_fraction: Share of rows permitted to cross a gutter. Zero
            demands a stripe empty on every row.

    Returns:
        ``(left, right)`` spans in points, left to right.
    """
    boxes = [box for row in rows for box in row]
    if not boxes:
        return []

    left = min(box.x_min for box in boxes) if x_min is None else x_min
    right = max(box.x_max for box in boxes) if x_max is None else x_max
    if right <= left:
        return []

    counts = _row_counts(rows, left, right, bin_pt)
    # Rounded, not truncated: a four-row table with one spanning heading is a
    # real shape, and truncation would allow zero bridging rows for any table
    # with fewer than five.
    allowed = int(round(max_bridge_fraction * len(rows)))
    covered = [count > allowed for count in counts]
    min_gap_bins = max(1, int(min_gap_pt / bin_pt))

    spans: List[Tuple[float, float]] = []
    run_start: Optional[int] = None
    empty_run = 0

    for index, is_covered in enumerate(covered):
        if is_covered:
            if empty_run >= min_gap_bins and run_start is not None:
                spans.append((left + run_start * bin_pt, left + (index - empty_run) * bin_pt))
                run_start = None
            empty_run = 0
            if run_start is None:
                run_start = index
        else:
            empty_run += 1

    if run_start is not None:
        spans.append((left + run_start * bin_pt, right))

    return [span for span in spans if span[1] - span[0] >= min_column_pt]


def column_index(spans: Sequence[Tuple[float, float]], box: BBox) -> int:
    """Return the index of the column a box belongs to.

    Assignment is by horizontal overlap, falling back to nearest centre for a
    word that straddles a gutter — a spanning header cell, usually. Such a cell
    lands in one column rather than being duplicated or dropped.

    Args:
        spans: Column spans from :func:`detect_column_spans`.
        box: The rectangle to place.

    Returns:
        The 0-based column index, or -1 if there are no spans.
    """
    if not spans:
        return -1

    best_index, best_overlap = -1, 0.0
    for index, (left, right) in enumerate(spans):
        overlap = min(box.x_max, right) - max(box.x_min, left)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    if best_index >= 0:
        return best_index

    centre = box.center_x
    return min(
        range(len(spans)),
        key=lambda index: abs(centre - 0.5 * (spans[index][0] + spans[index][1])),
    )


def spans_covering(
    spans: Sequence[Tuple[float, float]], box: BBox, min_overlap_pt: float = 1.0
) -> Tuple[float, float]:
    """Return the total extent of every column span the box reaches into.

    This is how a full-width figure gets a full-width crop while a
    single-column figure gets a single-column crop, without either being
    special-cased.

    Args:
        spans: Column spans from :func:`detect_column_spans`.
        box: The rectangle to test, usually a caption.
        min_overlap_pt: Overlap below which a span is not considered touched.

    Returns:
        ``(left, right)``, falling back to the box's own extent when it touches
        no span.
    """
    touched = [
        span
        for span in spans
        if min(box.x_max, span[1]) - max(box.x_min, span[0]) >= min_overlap_pt
    ]
    if not touched:
        return box.x_min, box.x_max
    return min(span[0] for span in touched), max(span[1] for span in touched)
