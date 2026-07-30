"""Where every word on the page is, not just what it says.

``pdftotext -layout`` gives readable text and nothing else — no coordinates, so
no way to tell a caption from a paragraph, a table cell from the sentence next
to it, or where a figure ends. ``pdftotext -bbox-layout`` gives the same text as
XHTML with a rectangle on every word, line and block, and that is what makes the
rest of this package possible: captions are found by their text but *located* by
their box, and everything downstream — figure crops, table columns, algorithm
blocks — is geometry from there.

The hierarchy poppler emits is ``page > flow > block > line > word``. Flows are
poppler's guess at reading order and are not useful to us, so they are flattened
away; blocks are roughly paragraphs and are the unit the region finder works in.

Coordinates are PDF points with the origin at the top-left of the page, which is
the same convention :mod:`geometry` and ``pdftoppm`` use, so nothing is flipped
anywhere in this package.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence
from xml.etree import ElementTree

from sparx_agency.tools.pdf_parser import poppler
from sparx_agency.tools.pdf_parser.geometry import BBox, union_all

_TAG_NAMESPACE = re.compile(r"^\{[^}]*\}")


@dataclass(frozen=True)
class Word:
    """One whitespace-delimited token with its rectangle."""

    text: str
    bbox: BBox


WIDE_GAP_PT = 12.0
"""Gap between words above which a line is tabular rather than prose.

Justified body text stretches its spaces, but not this far at the font sizes in
use; a table gutter, or the space between a figure's scattered labels, always
exceeds it. Several decisions in this package turn on this one number: where a
caption stops, whether a block is running prose, and whether a region is a grid.
"""


@dataclass(frozen=True)
class Line:
    """One typeset line: the words on it, and the box around them."""

    words: List[Word]
    bbox: BBox

    @property
    def text(self) -> str:
        """The line as a single space-joined string."""
        return " ".join(word.text for word in self.words)

    @property
    def max_word_gap(self) -> float:
        """The widest gap between consecutive words on the line, in points.

        The measurement that separates a table row from a sentence.
        """
        if len(self.words) < 2:
            return 0.0
        ordered = sorted(self.words, key=lambda word: word.bbox.x_min)
        return max(
            later.bbox.x_min - earlier.bbox.x_max
            for earlier, later in zip(ordered, ordered[1:])
        )


@dataclass(frozen=True)
class Block:
    """A group of lines poppler considers one unit — usually a paragraph."""

    lines: List[Line]
    bbox: BBox

    @property
    def text(self) -> str:
        """The block as newline-joined lines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def word_count(self) -> int:
        """Total words in the block."""
        return sum(len(line.words) for line in self.lines)

    def words(self) -> Iterator[Word]:
        """Yield every word in the block, in reading order."""
        for line in self.lines:
            for word in line.words:
                yield word


@dataclass(frozen=True)
class PageLayout:
    """Every positioned element on one page.

    Attributes:
        number: 1-based page number.
        width: Page width in points.
        height: Page height in points.
        blocks: Paragraph-level groups, in the order poppler emitted them.
    """

    number: int
    width: float
    height: float
    blocks: List[Block]

    @property
    def bbox(self) -> BBox:
        """The whole page as a box, for clipping crops."""
        return BBox(0.0, 0.0, self.width, self.height)

    @property
    def lines(self) -> List[Line]:
        """Every line on the page, flattened out of its block."""
        return [line for block in self.blocks for line in block.lines]

    @property
    def text(self) -> str:
        """The page as text, blocks separated by a blank line."""
        return "\n\n".join(block.text for block in self.blocks)

    def text_extent(self) -> Optional[BBox]:
        """The box around all text on the page, or None if the page has none.

        This is the practical definition of the page margins, and the region
        finder uses it as the outer bound for a figure that runs off the top of
        the text area.
        """
        return union_all(block.bbox for block in self.blocks)


ROW_TOLERANCE_FRACTION = 0.6
"""Share of a line's height two pieces of text may differ by and still be one row.

Text on one row shares a baseline but not always an exact box — a superscript or
a taller glyph shifts one of them. Six tenths of a line is comfortably more than
that drift and comfortably less than the pitch between rows.
"""


def group_rows(lines: Sequence[Line]) -> List[List[Line]]:
    """Cluster lines that share a baseline into rows, top to bottom.

    Poppler groups glyphs into lines by flow, and inside a table it usually
    decides each *cell* is its own line — so one visual row of five columns
    arrives as five separate lines at the same height. Anything that reasons
    about rows has to put them back together first, which is what this does.

    Args:
        lines: Lines to cluster, in any order.

    Returns:
        One list of lines per row, ordered down the page, each row's lines
        ordered left to right.
    """
    ordered = sorted(lines, key=lambda line: line.bbox.center_y)
    if not ordered:
        return []

    heights = sorted(line.bbox.height for line in ordered)
    tolerance = ROW_TOLERANCE_FRACTION * (heights[len(heights) // 2] or 1.0)

    rows: List[List[Line]] = [[ordered[0]]]
    centre = ordered[0].bbox.center_y
    for line in ordered[1:]:
        if abs(line.bbox.center_y - centre) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
        centre = sum(item.bbox.center_y for item in rows[-1]) / len(rows[-1])
    return [sorted(row, key=lambda line: line.bbox.x_min) for row in rows]


def _local_name(tag: str) -> str:
    """Strip the XHTML namespace poppler wraps every element in."""
    return _TAG_NAMESPACE.sub("", tag)


def _read_bbox(element: ElementTree.Element) -> Optional[BBox]:
    """Read the four bbox attributes off an element, or None if incomplete."""
    try:
        return BBox(
            float(element.attrib["xMin"]),
            float(element.attrib["yMin"]),
            float(element.attrib["xMax"]),
            float(element.attrib["yMax"]),
        )
    except (KeyError, ValueError):
        return None


def _parse_line(element: ElementTree.Element) -> Optional[Line]:
    """Build a :class:`Line` from a ``<line>`` element, or None if it is empty."""
    words: List[Word] = []
    for child in element:
        if _local_name(child.tag) != "word":
            continue
        text = (child.text or "").strip()
        box = _read_bbox(child)
        if text and box is not None:
            words.append(Word(text, box))
    if not words:
        return None
    bbox = _read_bbox(element) or union_all(word.bbox for word in words)
    return Line(words, bbox)


def _parse_block(element: ElementTree.Element) -> Optional[Block]:
    """Build a :class:`Block` from a ``<block>`` element, or None if it is empty."""
    lines = [line for line in (_parse_line(child) for child in element
                               if _local_name(child.tag) == "line") if line is not None]
    if not lines:
        return None
    bbox = _read_bbox(element) or union_all(line.bbox for line in lines)
    return Block(lines, bbox)


def _parse_page(element: ElementTree.Element, number: int) -> PageLayout:
    """Build a :class:`PageLayout` from a ``<page>`` element."""
    blocks: List[Block] = []
    for descendant in element.iter():
        if _local_name(descendant.tag) != "block":
            continue
        block = _parse_block(descendant)
        if block is not None:
            blocks.append(block)
    return PageLayout(
        number=number,
        width=float(element.attrib.get("width", 0.0)),
        height=float(element.attrib.get("height", 0.0)),
        blocks=blocks,
    )


def parse_layout_xml(xml_text: str) -> List[PageLayout]:
    """Parse the XHTML from ``pdftotext -bbox-layout`` into pages.

    Args:
        xml_text: The complete document poppler wrote to stdout.

    Returns:
        One :class:`PageLayout` per page, in document order.

    Raises:
        ValueError: If the text is not parseable XML.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ValueError("pdftotext -bbox-layout did not emit valid XML: {}".format(exc))

    pages: List[PageLayout] = []
    for element in root.iter():
        if _local_name(element.tag) == "page":
            pages.append(_parse_page(element, len(pages) + 1))
    return pages


def load_layout(
    pdf_path: Path, first: Optional[int] = None, last: Optional[int] = None
) -> List[PageLayout]:
    """Run ``pdftotext -bbox-layout`` over a PDF and parse the result.

    Args:
        pdf_path: The PDF to read.
        first: First page, 1-based, or None for the beginning.
        last: Last page, 1-based inclusive, or None for the end.

    Returns:
        One :class:`PageLayout` per page in the range. Page ``number`` is
        renumbered from 1 within the range, so pass the offset back yourself if
        you asked for a slice.
    """
    poppler.check_pdf(pdf_path)
    args = poppler.page_range_args(first, last) + ["-bbox-layout", str(pdf_path), "-"]
    xml_text = poppler.run("pdftotext", args)
    pages = parse_layout_xml(xml_text)
    if first is not None:
        pages = [
            PageLayout(page.number + first - 1, page.width, page.height, page.blocks)
            for page in pages
        ]
    return pages
