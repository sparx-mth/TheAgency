"""The PDF itself: how many pages, how big, and what it claims to be.

Thin on purpose. Everything expensive — text, layout, renders — is loaded by the
module that needs it, because a fifteen-page paper's layout XML is several
megabytes and most callers want the page count and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from sparx_agency.tools.pdf_parser import poppler


@dataclass(frozen=True)
class PdfDocument:
    """A PDF on disk and the metadata ``pdfinfo`` reports for it.

    Attributes:
        path: Absolute path to the file.
        page_count: Number of pages.
        width: Width of the first page in points.
        height: Height of the first page in points.
        info: Every ``key: value`` line ``pdfinfo`` printed, keys lowercased.
    """

    path: Path
    page_count: int
    width: float
    height: float
    info: Dict[str, str] = field(default_factory=dict)

    @property
    def title(self) -> Optional[str]:
        """The embedded title, or None.

        Usually absent or wrong on arXiv preprints — LaTeX writes it only when
        the author sets ``pdftitle``. Read the title off page 1 instead of
        trusting this.
        """
        title = self.info.get("title", "").strip()
        return title or None

    @property
    def producer(self) -> Optional[str]:
        """The producing toolchain, e.g. ``pdfTeX-1.40.25``, or None.

        Worth a glance: a producer naming a scanner rather than a typesetter
        predicts that there is no text layer and the paper must be read from
        page renders.
        """
        producer = self.info.get("producer", "").strip()
        return producer or None


def _parse_pdfinfo(output: str) -> Dict[str, str]:
    """Turn ``pdfinfo`` output into a dict, keys lowercased and stripped."""
    info: Dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            info[key.strip().lower()] = value.strip()
    return info


def _parse_page_size(info: Dict[str, str]) -> tuple:
    """Read ``Page size: 612 x 792 pts (letter)`` into ``(width, height)``.

    Returns US Letter if the line is missing or unparseable. That default only
    affects crops on a document poppler could not describe, and every crop is
    clipped to the real page by the renderer anyway.
    """
    raw = info.get("page size", "")
    parts = raw.replace("x", " ").split()
    numbers = []
    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            continue
        if len(numbers) == 2:
            break
    if len(numbers) < 2:
        return 612.0, 792.0
    return numbers[0], numbers[1]


def open_pdf(path: Path) -> PdfDocument:
    """Read a PDF's metadata.

    Args:
        path: Path to the PDF.

    Returns:
        The populated :class:`PdfDocument`.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a PDF, or reports no pages.
        PopplerNotInstalled: If poppler is not installed.
    """
    path = Path(path).expanduser().resolve()
    poppler.check_pdf(path)
    info = _parse_pdfinfo(poppler.run("pdfinfo", [str(path)]))

    try:
        page_count = int(info.get("pages", "0"))
    except ValueError:
        page_count = 0
    if page_count < 1:
        raise ValueError("{} reports {} pages — it is damaged".format(path, page_count))

    width, height = _parse_page_size(info)
    return PdfDocument(path=path, page_count=page_count, width=width, height=height, info=info)
