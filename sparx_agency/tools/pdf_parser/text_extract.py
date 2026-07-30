"""The paper as text, with the columns still standing.

Plain ``pdftotext`` reads a two-column page across both columns, so consecutive
lines come from different halves of the page and the result is unreadable prose
that looks like a bad OCR. ``-layout`` keeps the physical arrangement, which is
what makes the output quotable — and, as a side effect, keeps table rows on one
line, which is what makes :mod:`tables` able to check its own work.

Text is written twice: whole-paper, and one file per page. The per-page copy is
the one to quote from, because the filename carries the page number and a quote
without a page number cannot be checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from sparx_agency.tools.pdf_parser import poppler

PAGE_SEPARATOR = "\f"
"""Form feed — what ``pdftotext`` writes between pages."""

SCANNED_TEXT_THRESHOLD = 200
"""Characters per page below which a PDF is treated as having no text layer.

A typeset page carries 1500-4000 characters. A scanned page carries whatever the
header and the page number amount to, often zero. The gap is wide enough that
one threshold covers everything in between without tuning.
"""


@dataclass(frozen=True)
class ExtractedText:
    """The text of a paper, whole and per page.

    Attributes:
        full: The entire document, pages separated by form feeds.
        pages: One string per page, in order.
        is_scanned: True when there is too little text for the page count,
            meaning the paper must be read from page renders instead.
    """

    full: str
    pages: List[str]

    @property
    def char_count(self) -> int:
        """Total characters of extracted text."""
        return len(self.full)

    @property
    def is_scanned(self) -> bool:
        """True when the PDF has no usable text layer."""
        if not self.pages:
            return True
        return self.char_count / len(self.pages) < SCANNED_TEXT_THRESHOLD

    def page(self, number: int) -> str:
        """Return the text of a 1-based page number.

        Raises:
            IndexError: If the page does not exist.
        """
        if not 1 <= number <= len(self.pages):
            raise IndexError(
                "page {} out of range 1..{}".format(number, len(self.pages))
            )
        return self.pages[number - 1]


def split_pages(full_text: str) -> List[str]:
    """Split ``pdftotext`` output on form feeds into per-page strings.

    The final form feed produces a trailing empty string, which is dropped so
    that the page count matches the document.
    """
    pages = full_text.split(PAGE_SEPARATOR)
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def extract_text(pdf_path: Path) -> ExtractedText:
    """Read a PDF's text with the page layout preserved.

    Args:
        pdf_path: The PDF to read.

    Returns:
        The populated :class:`ExtractedText`.
    """
    poppler.check_pdf(pdf_path)
    full = poppler.run("pdftotext", ["-layout", str(pdf_path), "-"])
    return ExtractedText(full=full, pages=split_pages(full))


def write_text(text: ExtractedText, full_path: Path, page_dir: Path) -> List[Path]:
    """Write the whole-paper and per-page text files.

    Args:
        text: What :func:`extract_text` returned.
        full_path: Where to write the whole document.
        page_dir: Directory to write ``pNNN.txt`` into.

    Returns:
        The per-page paths written, in page order.
    """
    full_path.parent.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    full_path.write_text(text.full, encoding="utf-8")

    written: List[Path] = []
    for index, page_text in enumerate(text.pages, start=1):
        path = page_dir / "p{:03d}.txt".format(index)
        path.write_text(page_text, encoding="utf-8")
        written.append(path)
    return written
