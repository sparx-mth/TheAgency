"""Raster images lifted out of the PDF exactly as they are stored.

This is the *other* way to get a picture out of a paper, and it is the lesser
one. It only sees images that were embedded as pixels — photographs, screenshots,
qualitative result grids, rendered simulator frames. The plots and architecture
diagrams in a machine-learning paper are almost always vector drawings, so for
most papers this yields nothing at all and the crops in :mod:`page_render` are
the whole story. When it does yield something, though, it yields it at the
source resolution rather than at whatever the page was rendered to, which is why
it is worth running.

What comes out needs triage. A PDF's image list includes soft masks, one-pixel
rules and the occasional glyph stored as a bitmap, and dumped unsorted they
outnumber the real figures. Anything too small to be a figure is moved aside
rather than deleted — the judgement is a threshold, and a threshold is
sometimes wrong.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sparx_agency.tools.pdf_parser import poppler

MIN_DIMENSION_PX = 64
"""Smallest width or height, in pixels, an image may have and still be a figure.

Anything below this is a rule, a bullet, a logo or a bitmap glyph. Real figures
in this literature are hundreds of pixels on a side even when printed small.
"""

MASK_TYPES = ("smask", "stencil", "mask")
"""Image kinds that are transparency data for another image, not pictures."""


@dataclass(frozen=True)
class EmbeddedImage:
    """One raster lifted out of the PDF.

    Attributes:
        path: Where it was written.
        page: 1-based page it appears on.
        index: Poppler's index for it within the document.
        width: Stored width in pixels.
        height: Stored height in pixels.
        kind: Poppler's type string, e.g. ``image`` or ``smask``.
    """

    path: Path
    page: int
    index: int
    width: int
    height: int
    kind: str

    @property
    def is_figure_sized(self) -> bool:
        """True when the image is large enough, and not a mask, to be a figure."""
        if self.kind.lower() in MASK_TYPES:
            return False
        return self.width >= MIN_DIMENSION_PX and self.height >= MIN_DIMENSION_PX


@dataclass(frozen=True)
class ImageExtraction:
    """What came out of one PDF.

    Attributes:
        figures: Images large enough to be worth looking at.
        set_aside: Everything else, kept on disk under ``_small/``.
        note: A human-readable caveat, or None. Populated when the image list
            and the extracted files did not line up, so the manifest can say so
            instead of quietly presenting a guess.
    """

    figures: List[EmbeddedImage]
    set_aside: List[EmbeddedImage]
    note: Optional[str] = None


def _parse_list(output: str) -> List[dict]:
    """Parse ``pdfimages -list`` into one dict per row."""
    rows: List[dict] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "page": int(fields[0]),
                    "index": int(fields[1]),
                    "kind": fields[2],
                    "width": int(fields[3]),
                    "height": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _written_path(out_dir: Path, prefix: str, page: int, index: int) -> Optional[Path]:
    """Find the file ``pdfimages -p`` wrote for one image list row.

    The naming is ``<prefix>-<page>-<index>.png`` with both numbers zero-padded
    to three digits, so the path can be constructed rather than guessed. The
    glob is the fallback for a document with more than 999 pages, where poppler
    widens the field.
    """
    exact = out_dir / "{}-{:03d}-{:03d}.png".format(prefix, page, index)
    if exact.is_file():
        return exact
    matches = sorted(out_dir.glob("{}-*-{}.png".format(prefix, str(index).zfill(3))))
    return matches[0] if matches else None


def extract_images(pdf_path: Path, out_dir: Path, prefix: str = "img") -> ImageExtraction:
    """Extract every embedded raster and sort the figures from the noise.

    Args:
        pdf_path: The PDF to read.
        out_dir: Directory to write into; created if absent. Small images are
            moved into an ``_small`` subdirectory of it.
        prefix: Filename prefix for extracted images.

    Returns:
        The populated :class:`ImageExtraction`.
    """
    poppler.check_pdf(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    small_dir = out_dir / "_small"
    small_dir.mkdir(parents=True, exist_ok=True)

    listed = _parse_list(poppler.run("pdfimages", ["-list", str(pdf_path)]))
    poppler.run("pdfimages", ["-png", "-p", str(pdf_path), str(out_dir / prefix)])

    figures: List[EmbeddedImage] = []
    set_aside: List[EmbeddedImage] = []
    unmatched = 0

    for row in listed:
        path = _written_path(out_dir, prefix, row["page"], row["index"])
        if path is None:
            unmatched += 1
            continue
        image = EmbeddedImage(
            path=path,
            page=row["page"],
            index=row["index"],
            width=row["width"],
            height=row["height"],
            kind=row["kind"],
        )
        if image.is_figure_sized:
            figures.append(image)
        else:
            moved = small_dir / path.name
            shutil.move(str(path), str(moved))
            set_aside.append(
                EmbeddedImage(moved, image.page, image.index, image.width, image.height, image.kind)
            )

    note = None
    if unmatched:
        note = (
            "{} of {} listed images produced no file — they use an encoding "
            "pdfimages cannot write as PNG".format(unmatched, len(listed))
        )
    return ImageExtraction(figures=figures, set_aside=set_aside, note=note)
