"""Building small PDFs by hand, so the tests can exercise the real toolchain.

Every interesting behaviour in this package depends on where poppler says a word
is, so a test suite built on hand-written fixture strings would check the
heuristics against an imagined PDF rather than a real one. Committing a paper as
a binary fixture is not an option either — it is somebody's copyright, and the
repository does not carry blobs.

So the fixtures are written as PDFs, here, from nothing. A page is a list of
strings with positions, and the file that comes out is a valid, if very plain,
PDF that poppler reads exactly as it reads a typeset one.

Coordinates are given with the origin at the **top left** and ``y`` growing
downward — the convention the rest of this package uses — and are flipped to
PDF's bottom-left origin on the way out, so a test never has to think about it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0
DEFAULT_FONT_SIZE = 10.0


@dataclass(frozen=True)
class TextItem:
    """One string placed on a page.

    Attributes:
        x: Left edge in points from the left of the page.
        y: Baseline in points from the *top* of the page.
        text: The string, Latin-1 only.
        size: Font size in points.
    """

    x: float
    y: float
    text: str
    size: float = DEFAULT_FONT_SIZE


def _escape(text: str) -> bytes:
    """Escape the three characters that are special inside a PDF string."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return escaped.encode("latin-1", "replace")


def _content_stream(items: Sequence[TextItem], page_height: float) -> bytes:
    """Build the drawing commands for one page."""
    parts: List[bytes] = []
    for item in items:
        parts.append(
            b"BT /F1 %f Tf %f %f Td (%s) Tj ET\n"
            % (item.size, item.x, page_height - item.y, _escape(item.text))
        )
    return b"".join(parts)


def build_pdf(
    pages: Sequence[Sequence[TextItem]],
    width: float = PAGE_WIDTH_PT,
    height: float = PAGE_HEIGHT_PT,
) -> bytes:
    """Assemble a valid PDF containing the given text.

    Args:
        pages: One sequence of :class:`TextItem` per page.
        width: Page width in points.
        height: Page height in points.

    Returns:
        The complete PDF as bytes, ready to write to a file.

    Raises:
        ValueError: If no pages were given.
    """
    if not pages:
        raise ValueError("a PDF needs at least one page")

    page_ids = [4 + 2 * index for index in range(len(pages))]
    content_ids = [5 + 2 * index for index in range(len(pages))]

    bodies: List[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % pid for pid in page_ids), len(pages)),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    for index, items in enumerate(pages):
        stream = _content_stream(items, height)
        bodies.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %f %f] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>"
            % (width, height, content_ids[index])
        )
        bodies.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        )

    return _assemble(bodies)


def _assemble(bodies: Sequence[bytes]) -> bytes:
    """Serialise numbered objects with a cross-reference table and trailer."""
    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = []

    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(bodies) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset

    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(bodies) + 1,
        xref_offset,
    )
    return bytes(out)


def paragraph(
    y: float, text: str, x: float = 60.0, size: float = DEFAULT_FONT_SIZE
) -> TextItem:
    """A body-text line — long enough that the region finder treats it as prose."""
    return TextItem(x=x, y=y, text=text, size=size)


def table_row(y: float, cells: Sequence[str], starts: Sequence[float]) -> List[TextItem]:
    """A row of cells at fixed column positions, with real gutters between them.

    Args:
        y: Baseline from the top of the page.
        cells: Cell contents, left to right.
        starts: Left edge of each column in points; must match ``cells``.

    Returns:
        One :class:`TextItem` per non-empty cell.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(cells) != len(starts):
        raise ValueError("{} cells but {} column starts".format(len(cells), len(starts)))
    return [
        TextItem(x=start, y=y, text=cell)
        for cell, start in zip(cells, starts)
        if cell
    ]


def lorem(words: int, seed: str = "alpha") -> str:
    """A deterministic run of filler words, for making a block look like prose."""
    vocabulary = (
        "the model learns a policy over observations and produces actions which "
        "the controller then tracks along the planned route without further input"
    ).split()
    return " ".join(vocabulary[index % len(vocabulary)] for index in range(words)) + " " + seed


def page_size() -> Tuple[float, float]:
    """The default page size used by :func:`build_pdf`."""
    return PAGE_WIDTH_PT, PAGE_HEIGHT_PT
