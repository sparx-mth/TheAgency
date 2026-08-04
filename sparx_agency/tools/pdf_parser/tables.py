"""Turning a table on a page back into cells.

A PDF has no idea it contains a table. There are no cells, no rows and no
column headers in the file — only glyphs at coordinates that a reader's eye
assembles into a grid. Copying a table out of a paper by hand is where numbers
get transposed, so it is worth doing mechanically.

The reconstruction is two independent steps, and keeping them independent is
what makes it robust. Rows come from vertical position: everything sitting at
the same height is one row. Columns come from the whitespace: the vertical
stripes no word ever crosses are the gutters (see :mod:`columns`). Every word is
then dropped into the cell where its row meets its column.

Rows are clustered rather than taken from poppler's lines, and that distinction
is the whole difference between a table that parses and one that does not.
Poppler groups glyphs into lines by *flow*, and in a table it usually decides
each cell is its own line, so a five-column row arrives as five separate lines
that happen to share a baseline. Treating each of those as a row produces a
table with one populated cell per row and everything else blank — which looks
like a parser bug and is really a wrong assumption about what a line is.

What this does not do is decide which row is the header — it takes the first —
or merge a cell that wrapped onto two lines, which arrives as two rows. Both are
visible at a glance in the crop that is rendered beside every parsed table, and
that pairing is the point: the markdown is for quoting, the image is for
checking it.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sparx_agency.tools.pdf_parser import columns as columns_mod
from sparx_agency.tools.pdf_parser import layout as layout_mod
from sparx_agency.tools.pdf_parser import regions as regions_mod
from sparx_agency.tools.pdf_parser.captions import Caption
from sparx_agency.tools.pdf_parser.layout import Line, PageLayout
from sparx_agency.tools.pdf_parser.regions import Region

MIN_COLUMNS = 2
"""Fewest columns a region must resolve into to be called a table."""

MIN_ROWS = 2
"""Fewest non-empty rows a region must resolve into to be called a table."""

group_rows = layout_mod.group_rows
"""Cluster lines sharing a baseline into rows. Defined in :mod:`layout`, which the
region finder also uses for the same purpose."""


@dataclass(frozen=True)
class Table:
    """A parsed table.

    Attributes:
        caption: The caption the table was found from.
        header: The first row, taken as column headings.
        rows: Every row after the first.
        column_spans: Horizontal extent of each column, in points, for tracing a
            misparse back to the page.
    """

    caption: Caption
    header: List[str]
    rows: List[List[str]]
    column_spans: List[Tuple[float, float]]

    @property
    def column_count(self) -> int:
        """Number of columns detected."""
        return len(self.column_spans)

    @property
    def row_count(self) -> int:
        """Number of rows below the header."""
        return len(self.rows)

    def to_markdown(self) -> str:
        """Render as a GitHub-flavoured markdown table, caption first."""
        lines = ["**{}** — {}".format(self.caption.label, self.caption.text).rstrip(" —"), ""]
        lines.append("| " + " | ".join(_escape(cell) for cell in self.header) + " |")
        lines.append("|" + "|".join([" --- "] * self.column_count) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
        lines.append("")
        lines.append(
            "_Parsed from page {}. Check against `figures/{}.png` before quoting._".format(
                self.caption.page, self.caption.slug
            )
        )
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        """Render as CSV, header row first."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.header)
        writer.writerows(self.rows)
        return buffer.getvalue()


def _escape(cell: str) -> str:
    """Escape the one character that would break a markdown table."""
    return cell.replace("|", "\\|").strip()


def _row_cells(row: Sequence[Line], spans: Sequence[Tuple[float, float]]) -> List[str]:
    """Place one row's words into one cell per column."""
    cells: List[List[str]] = [[] for _ in spans]
    words = [word for line in row for word in line.words]
    for word in sorted(words, key=lambda item: item.bbox.x_min):
        index = columns_mod.column_index(spans, word.bbox)
        if index >= 0:
            cells[index].append(word.text)
    return [" ".join(cell).strip() for cell in cells]


def _pad(rows: List[List[str]], width: int) -> List[List[str]]:
    """Make every row the same width, so markdown and CSV stay rectangular."""
    return [row + [""] * (width - len(row)) for row in rows]


def parse_table(page: PageLayout, region: Region) -> Optional[Table]:
    """Reconstruct the cells of a table from its region.

    Args:
        page: The page the table is on.
        region: The region resolved from the table's caption.

    Returns:
        The parsed :class:`Table`, or None when the region does not resolve into
        a grid — too few columns or too few rows. None is the honest answer for
        a "table" that is really a boxed equation or a single-column list.
    """
    lines = regions_mod.lines_in(page, region)
    grouped = group_rows(lines)
    if len(grouped) < MIN_ROWS:
        return None

    spans = columns_mod.detect_column_spans_from_rows(
        [[word.bbox for line in row for word in line.words] for row in grouped],
        min_gap_pt=columns_mod.TABLE_GUTTER_PT,
        x_min=region.content.x_min,
        x_max=region.content.x_max,
    )
    if len(spans) < MIN_COLUMNS:
        return None

    rows = [_row_cells(row, spans) for row in grouped]
    rows = [row for row in rows if any(cell for cell in row)]
    if len(rows) < MIN_ROWS:
        return None

    rows = _pad(rows, len(spans))
    return Table(
        caption=region.caption,
        header=rows[0],
        rows=rows[1:],
        column_spans=spans,
    )


def parse_tables(page: PageLayout, page_regions: Sequence[Region]) -> List[Table]:
    """Parse every table region on a page, skipping those that do not resolve.

    Args:
        page: The parsed page.
        page_regions: Regions found on it; non-table regions are ignored.

    Returns:
        One :class:`Table` per region that produced a grid.
    """
    parsed = (
        parse_table(page, region)
        for region in page_regions
        if region.caption.kind == "table"
    )
    return [table for table in parsed if table is not None]
