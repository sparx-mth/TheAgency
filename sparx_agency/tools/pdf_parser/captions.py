"""Finding the labels that tell you what is a figure, a table or an algorithm.

Captions are the index into a paper's exhibits, and they are the only reliable
one: a paper is free to lay its figures out however it likes, but it will always
label them, and it will always label them the same way throughout. So every
figure crop, every parsed table and every extracted algorithm in this package
starts from a caption and works outward geometrically.

Two rules keep false positives down, and both matter more than they look:

**A caption starts a block.** ``pdftotext -bbox-layout`` groups lines into
paragraph-sized blocks, and a caption is its own block. A sentence in running
prose that happens to begin "Figure 2 shows..." is not, because it continues the
paragraph it lives in.

**A caption separates its number from its text.** "Figure 2: The architecture"
and "Fig. 1. Overview" are captions; "Figure 2 shows the architecture" is not.
Requiring the colon or full stop after the number is what distinguishes a label
from a cross-reference, and it costs almost nothing — every template in this
literature uses one.

The cost of these rules is that an unlabelled figure is invisible to this
module. That is the right trade: a missed figure is still on the page render,
while a cropped paragraph is noise that has to be read to be dismissed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sparx_agency.tools.pdf_parser import layout as layout_mod
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.layout import Block, PageLayout

FIGURE = "figure"
TABLE = "table"
ALGORITHM = "algorithm"

_KIND_BY_WORD: Dict[str, str] = {
    "figure": FIGURE, "fig": FIGURE, "figs": FIGURE,
    "table": TABLE, "tab": TABLE,
    "algorithm": ALGORITHM, "alg": ALGORITHM, "listing": ALGORITHM,
    "procedure": ALGORITHM,
}
"""Caption-opening words, and the exhibit kind each one introduces."""

_CAPTION = re.compile(
    r"^(?P<word>figure|figs?|table|tab|algorithm|alg|listing|procedure)\s*\.?\s*"
    r"(?P<number>[0-9]+(?:\.[0-9]+)*|[A-Z][0-9]*|[IVXLC]+)\s*"
    r"(?P<separator>[:.—–)])\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)
"""One caption line. The separator group is what makes this not match prose."""


@dataclass(frozen=True)
class Caption:
    """One labelled exhibit, located on the page.

    Attributes:
        kind: One of ``figure``, ``table``, ``algorithm``.
        number: The label as printed — ``"2"``, ``"A1"``, ``"3.1"``.
        text: The caption text, with the label stripped, lines joined.
        page: 1-based page number.
        bbox: The caption block's rectangle.
        block_index: Index of the caption's block within the page, so the region
            finder can exclude the caption from the content it measures.
    """

    kind: str
    number: str
    text: str
    page: int
    bbox: BBox
    block_index: int

    @property
    def label(self) -> str:
        """The exhibit's name as printed, e.g. ``"Figure 2"``."""
        return "{} {}".format(self.kind.capitalize(), self.number)

    @property
    def slug(self) -> str:
        """A filename stem for this exhibit, e.g. ``"figure-2"``."""
        safe = re.sub(r"[^A-Za-z0-9.]+", "-", self.number).strip("-").lower()
        return "{}-{}".format(self.kind, safe or "x")

    def to_dict(self) -> Dict[str, object]:
        """A JSON-serialisable view, for ``captions.json``."""
        return {
            "kind": self.kind,
            "number": self.number,
            "label": self.label,
            "text": self.text,
            "page": self.page,
            "bbox": [self.bbox.x_min, self.bbox.y_min, self.bbox.x_max, self.bbox.y_max],
        }


def caption_line_count(block: Block) -> int:
    """How many of a block's leading lines are the caption sentence itself.

    Usually all of them — a caption is its own block. But poppler sometimes
    groups a caption together with the table underneath it, and then the block
    holds the caption *and* the exhibit. Without this the caption text would
    swallow the whole table, and the region finder would have nothing left to
    look for below it.

    The caption is the run of single, full-width lines at the top. It ends at
    the first sign of an exhibit: text sitting side by side at one height,
    gutters inside a single line, or a vertical jump. The side-by-side test is
    the one that carries a table — a table's cells are separate lines at the
    same height, and a caption's lines never are.

    Args:
        block: The caption's block.

    Returns:
        The number of leading lines belonging to the caption, at least one.
    """
    rows = layout_mod.group_rows(block.lines)
    if len(rows) < 2:
        return len(block.lines)

    heights = sorted(line.bbox.height for line in block.lines)
    pitch = heights[len(heights) // 2] or 1.0

    count = 0
    previous = None
    for row in rows:
        if count and len(row) > 1:
            break
        line = row[0]
        if count and line.max_word_gap >= layout_mod.WIDE_GAP_PT:
            break
        if previous is not None and line.bbox.y_min - previous.bbox.y_max > 1.8 * pitch:
            break
        count += len(row)
        previous = line
    return max(count, 1)


def _caption_from_block(block: Block, page: int, block_index: int) -> Optional[Caption]:
    """Read a caption off a block, or return None if the block is not one."""
    if not block.lines:
        return None

    # Topmost line, not first-emitted: a caption always opens its block visually,
    # and poppler's emission order is only usually the same thing.
    ordered = sorted(block.lines, key=lambda line: (line.bbox.y_min, line.bbox.x_min))
    match = _CAPTION.match(ordered[0].text.strip())
    if match is None:
        return None

    kind = _KIND_BY_WORD.get(match.group("word").lower().rstrip("."))
    if kind is None:
        return None

    own_lines = ordered[: caption_line_count(block)]
    remainder = [match.group("text").strip()]
    remainder += [line.text.strip() for line in own_lines[1:]]
    text = " ".join(part for part in remainder if part)

    return Caption(
        kind=kind,
        number=match.group("number"),
        text=text,
        page=page,
        bbox=block.bbox,
        block_index=block_index,
    )


def find_captions(page: PageLayout) -> List[Caption]:
    """Find every caption on one page.

    Args:
        page: The parsed page.

    Returns:
        The captions, in the order their blocks appear.
    """
    found: List[Caption] = []
    for index, block in enumerate(page.blocks):
        caption = _caption_from_block(block, page.number, index)
        if caption is not None:
            found.append(caption)
    return found


def find_all_captions(pages: List[PageLayout]) -> List[Caption]:
    """Find every caption in a document.

    Args:
        pages: Parsed pages, in order.

    Returns:
        Every caption found, in document order.
    """
    return [caption for page in pages for caption in find_captions(page)]
