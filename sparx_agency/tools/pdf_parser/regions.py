"""Working out how much of the page each caption is talking about.

A caption says "Figure 2" but not where Figure 2 ends, and the PDF does not
record it either — there is no object in the file meaning "this drawing is the
figure". The extent has to be inferred, and this module infers it the way a
reader does: a figure is the material between its caption and the last piece of
running prose on the same side of it.

The direction differs by kind, because the templates differ. Figure captions sit
*below* their artwork, so a figure grows upward. Table and algorithm captions sit
*above* their body, so those grow downward. Both directions stop at the same
three things: a block of running prose, another caption, or a vertical gap too
large to be inside one exhibit.

Two supports make this reliable rather than lucky. Horizontally the region takes
the full width of whichever page columns the caption touches, so a full-width
figure gets a full-width crop and a single-column one does not (see
:mod:`columns`). And "running prose" is decided by how the words sit rather than
by how many there are (see :mod:`prose`), which is what keeps a table body or a
diagram's labels from being mistaken for the paragraph that should stop the
search.

Where the inference is wrong it is visibly wrong, because every region is
rendered as an image next to whatever was parsed out of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from sparx_agency.tools.pdf_parser import captions as captions_mod
from sparx_agency.tools.pdf_parser import columns as columns_mod
from sparx_agency.tools.pdf_parser import prose
from sparx_agency.tools.pdf_parser.captions import ALGORITHM, Caption, FIGURE, TABLE
from sparx_agency.tools.pdf_parser.geometry import BBox, union_all
from sparx_agency.tools.pdf_parser.layout import Block, Line, PageLayout

ABOVE = "above"
BELOW = "below"
INSIDE = "inside"

MAX_INTERNAL_GAP_PT = 30.0
"""Vertical gap *within* an exhibit that ends it.

Wide enough to cross the rule under a table header, narrow enough not to
swallow the next paragraph.
"""

MAX_CAPTION_GAP_PT = 60.0
"""Vertical gap between a caption and the start of its exhibit.

Deliberately larger than the internal gap, because these are two different
distances. A caption is separated from its table by a deliberate typographic
space, often with a rule in it; the rows of the table are separated only by
line pitch. Holding both to the tighter number loses tables that are merely
set generously.
"""

TOP_MARGIN_PT = 12.0
"""Extra headroom above the topmost text when a figure runs to the top of the page."""

PAGE_TOP_MARGIN_PT = 54.0
"""Where the top of the text area is assumed to be when the page cannot say.

Reached on the page that is nothing but one large figure: the artwork is vector,
so it contributes no text, and the only text on the page is the caption itself.
Measuring the top of the figure from the topmost text then measures it from the
caption and yields nothing at all. Three quarters of an inch is the standard
margin in every template in this literature, and the crop is padded anyway.
"""

MIN_CONTENT_PT = 18.0
"""Content shorter than this is not an exhibit — the caption search went the wrong way."""

MAX_REGION_FRACTION = 0.92
"""Cap on region height as a fraction of the page, so a bad inference cannot run away."""

GROW_DIRECTION = {FIGURE: ABOVE, TABLE: BELOW, ALGORITHM: BELOW}
"""Which way each kind of exhibit extends from its caption, by template convention."""

TEXTUAL_KINDS = (TABLE, ALGORITHM)
"""Exhibits made of text, whose region must therefore contain some."""


@dataclass(frozen=True)
class Region:
    """The area of a page one caption refers to.

    Attributes:
        caption: The caption this region belongs to.
        content: The exhibit itself, without the caption.
        bbox: Content and caption together — what gets rendered.
        side: ``above`` or ``below``, which way the search grew.
        block_indices: Blocks of the page that fell inside the content.
    """

    caption: Caption
    content: BBox
    bbox: BBox
    side: str
    block_indices: List[int]

    @property
    def page(self) -> int:
        """1-based page number."""
        return self.caption.page


def page_column_spans(page: PageLayout) -> List[Tuple[float, float]]:
    """Detect the page's text columns, one span per column."""
    word_boxes = [word.bbox for block in page.blocks for word in block.words()]
    return columns_mod.detect_column_spans(
        word_boxes, min_gap_pt=columns_mod.PAGE_GUTTER_PT
    )


def _column_box(page: PageLayout, caption: Caption) -> BBox:
    """The full-height box of whichever page columns the caption touches."""
    left, right = columns_mod.spans_covering(page_column_spans(page), caption.bbox)
    return BBox(left, 0.0, right, page.height)


def _in_column(block: Block, column: BBox) -> bool:
    """True when a block belongs to the given column band."""
    return block.bbox.x_overlap_ratio(column) >= 0.5


def _grow_up(
    page: PageLayout, caption: Caption, column: BBox, stop_blocks: Set[int]
) -> Tuple[Optional[float], List[int]]:
    """Find the top edge of an exhibit whose caption sits beneath it."""
    taken: List[int] = []
    bound: Optional[float] = None

    candidates = [
        (index, block)
        for index, block in enumerate(page.blocks)
        if index != caption.block_index
        and _in_column(block, column)
        and block.bbox.y_max <= caption.bbox.y_min + 2.0
    ]
    for index, block in sorted(candidates, key=lambda pair: pair[1].bbox.y_max, reverse=True):
        if index in stop_blocks or prose.is_body_text(block, column.width):
            bound = block.bbox.y_max + 1.0
            break
        taken.append(index)

    if bound is None:
        # Nothing above bounded the figure, so it runs to the top of the text
        # area — measured from the highest text if there is any above the
        # caption, and from the standard margin if the page has none.
        tops = [block.bbox.y_min for _, block in candidates]
        bound = (min(tops) if tops else PAGE_TOP_MARGIN_PT) - TOP_MARGIN_PT
    return max(bound, 0.0), taken


def _grow_down(
    page: PageLayout, caption: Caption, column: BBox, stop_blocks: Set[int]
) -> Tuple[Optional[float], List[int]]:
    """Find the bottom edge of an exhibit whose caption sits above it."""
    bottom = caption.bbox.y_max
    taken: List[int] = []

    candidates = [
        (index, block)
        for index, block in enumerate(page.blocks)
        if index != caption.block_index
        and _in_column(block, column)
        and block.bbox.y_min >= caption.bbox.y_max - 2.0
    ]
    for index, block in sorted(candidates, key=lambda pair: pair[1].bbox.y_min):
        if index in stop_blocks or prose.is_body_text(block, column.width):
            break
        allowed = MAX_CAPTION_GAP_PT if not taken else MAX_INTERNAL_GAP_PT
        if block.bbox.y_min - bottom > allowed:
            break
        bottom = max(bottom, block.bbox.y_max)
        taken.append(index)
    return min(bottom, page.height), taken


def _content_inside_caption(page: PageLayout, caption: Caption) -> Optional[BBox]:
    """Return the exhibit's box when it shares a block with its caption.

    Args:
        page: The page the caption is on.
        caption: The caption to inspect.

    Returns:
        The box around everything after the caption sentence, or None when the
        block holds nothing but the caption — which is the usual case.
    """
    block = page.blocks[caption.block_index]
    lines = sorted(block.lines, key=lambda line: (line.bbox.y_min, line.bbox.x_min))
    remainder = lines[captions_mod.caption_line_count(block):]
    if not remainder:
        return None

    box = union_all(line.bbox for line in remainder)
    if box is None or box.height < MIN_CONTENT_PT:
        return None
    return box


def _has_text(page: PageLayout, box: BBox) -> bool:
    """True when any of the page's lines sits inside the box."""
    return any(
        box.y_min <= line.bbox.center_y <= box.y_max
        and line.bbox.x_overlap_ratio(box) >= 0.3
        for line in page.lines
    )


def _region_for_side(
    page: PageLayout, caption: Caption, column: BBox, stop_blocks: Set[int], side: str
) -> Optional[Region]:
    """Build the region on one side of a caption, or None if it is degenerate."""
    if side == ABOVE:
        top, taken = _grow_up(page, caption, column, stop_blocks)
        content = BBox(column.x_min, top, column.x_max, caption.bbox.y_min)
    else:
        bottom, taken = _grow_down(page, caption, column, stop_blocks)
        content = BBox(column.x_min, caption.bbox.y_max, column.x_max, bottom)

    if content.height < MIN_CONTENT_PT:
        return None
    if content.height > MAX_REGION_FRACTION * page.height:
        return None
    if caption.kind in TEXTUAL_KINDS and not _has_text(page, content):
        # A table or an algorithm is made of text by definition, so an empty
        # region is the search having wandered into the page margin. A figure
        # is not: its region is legitimately empty when the artwork is vector.
        return None

    return Region(
        caption=caption,
        content=content,
        bbox=content.union(caption.bbox),
        side=side,
        block_indices=taken,
    )


def find_region(
    page: PageLayout, caption: Caption, captions: Sequence[Caption]
) -> Optional[Region]:
    """Locate the exhibit one caption refers to.

    Three possibilities are tried in order: the exhibit shares a block with its
    caption, the exhibit is on the caption's conventional side, or it is on the
    other side — the last covering templates that caption tables below rather
    than above.

    Args:
        page: The page the caption is on.
        caption: The caption to resolve.
        captions: Every caption on the page, so the search stops at its
            neighbours instead of swallowing them.

    Returns:
        The region, or None if nothing exhibit-shaped is anywhere near it.
    """
    column = _column_box(page, caption)
    stop_blocks = {other.block_index for other in captions if other is not caption}

    inside = _content_inside_caption(page, caption)
    if inside is not None:
        return Region(
            caption=caption,
            content=inside,
            bbox=inside.union(caption.bbox),
            side=INSIDE,
            block_indices=[caption.block_index],
        )

    preferred = GROW_DIRECTION.get(caption.kind, ABOVE)
    for side in (preferred, BELOW if preferred == ABOVE else ABOVE):
        region = _region_for_side(page, caption, column, stop_blocks, side)
        if region is not None:
            return region
    return None


def find_regions(page: PageLayout, captions: Sequence[Caption]) -> List[Region]:
    """Resolve every caption on a page to a region, dropping those that fail.

    Args:
        page: The parsed page.
        captions: The captions found on it.

    Returns:
        One region per resolvable caption, in caption order.
    """
    resolved = (find_region(page, caption, captions) for caption in captions)
    return [region for region in resolved if region is not None]


def lines_in(page: PageLayout, region: Region) -> List[Line]:
    """Return the page's lines that fall inside a region's content, top to bottom.

    Used by the table and algorithm parsers, which work on the text inside a
    region rather than on the region's picture.
    """
    inside = [
        line
        for line in page.lines
        if line.bbox.center_y >= region.content.y_min
        and line.bbox.center_y <= region.content.y_max
        and line.bbox.x_overlap_ratio(region.content) >= 0.3
    ]
    return sorted(inside, key=lambda line: (line.bbox.y_min, line.bbox.x_min))
