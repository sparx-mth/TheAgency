"""Turning pages into images, whole or in part.

This is how figures actually get read. Most plots in a LaTeX paper are vector
drawings — there is no image inside the PDF to lift out, only drawing commands —
so the only way to see a chart is to render the page it is on. Rendering is also
the honest way to read a *table*: the parsed cells in :mod:`tables` can be
checked against the picture, and where the two disagree the picture is right.

Two resolutions, for two jobs. Whole pages go at 150 DPI, which is legible for
body text and cheap enough to render every page of a paper. Crops go at 300 DPI,
because a crop is taken when something was too small to read at 150.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from sparx_agency.tools.pdf_parser import poppler
from sparx_agency.tools.pdf_parser.geometry import BBox

PAGE_DPI = 150.0
"""Resolution for whole-page renders. Body text is comfortably legible here."""

CROP_DPI = 300.0
"""Resolution for cropped regions. A crop exists because 150 was not enough."""

CROP_PADDING_PT = 6.0
"""Points of margin added around a measured region before cropping.

A figure's extent is measured from the text it contains — axis labels, node
names — so the drawing itself, the axes and the plot border all sit slightly
outside it. Without padding every crop shaves its own frame.
"""


def render_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: float = PAGE_DPI,
    first: Optional[int] = None,
    last: Optional[int] = None,
) -> List[Path]:
    """Render whole pages to PNG.

    Args:
        pdf_path: The PDF to render.
        out_dir: Directory to write into; created if absent.
        dpi: Resolution.
        first: First page, 1-based, or None for the beginning.
        last: Last page, 1-based inclusive, or None for the end.

    Returns:
        The PNG paths written, sorted by page number.
    """
    poppler.check_pdf(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = poppler.page_range_args(first, last) + [
        "-png", "-r", str(dpi), str(pdf_path), str(out_dir / "p"),
    ]
    poppler.run("pdftoppm", args)
    return sorted(out_dir.glob("p-*.png"))


def render_region(
    pdf_path: Path,
    page_number: int,
    region: BBox,
    out_path: Path,
    dpi: float = CROP_DPI,
    padding_pt: float = CROP_PADDING_PT,
    page_bounds: Optional[BBox] = None,
) -> Path:
    """Render one rectangle of one page to a PNG.

    Args:
        pdf_path: The PDF to render.
        page_number: 1-based page the region is on.
        region: The rectangle, in PDF points.
        out_path: Exact file to write. Its parent is created if absent.
        dpi: Resolution.
        padding_pt: Margin added on every side before cropping.
        page_bounds: The page rectangle, used to clip the padded region. Pass it
            whenever it is known; without it a region near an edge produces a
            crop with a blank strip.

    Returns:
        ``out_path``.

    Raises:
        PopplerFailed: If poppler produced no output for the requested crop,
            which in practice means the region fell outside the page.
    """
    padded = region.padded(padding_pt, page_bounds)
    x, y, width, height = padded.to_pixels(dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pdf_parser_crop_") as staging:
        prefix = Path(staging) / "crop"
        poppler.run(
            "pdftoppm",
            [
                "-png", "-r", str(dpi),
                "-f", str(page_number), "-l", str(page_number),
                "-x", str(x), "-y", str(y), "-W", str(width), "-H", str(height),
                str(pdf_path), str(prefix),
            ],
        )
        produced = sorted(Path(staging).glob("crop*.png"))
        if not produced:
            raise poppler.PopplerFailed(
                "no output cropping page {} at ({}, {}) {}x{} px — the region "
                "is outside the page".format(page_number, x, y, width, height)
            )
        shutil.move(str(produced[0]), str(out_path))
    return out_path
