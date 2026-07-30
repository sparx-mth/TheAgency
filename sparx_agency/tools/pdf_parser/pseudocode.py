"""Algorithm blocks, with the indentation that carries their meaning.

An algorithm listing is the one part of a paper where horizontal position is
semantic: the nesting of the loops and branches is the algorithm. Extracted with
ordinary text tools it arrives as a flat list of statements, every line starting
at column zero, and the control flow — the thing worth reading — is gone.

So listings are rebuilt on a character grid instead (see :mod:`textgrid`), which
restores both the indentation and the alignment of any line numbers down the
left margin. What comes out is close enough to the printed block to be read as
code, and it is written to its own file per algorithm rather than buried in the
page text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from sparx_agency.tools.pdf_parser import regions as regions_mod
from sparx_agency.tools.pdf_parser import textgrid
from sparx_agency.tools.pdf_parser.captions import ALGORITHM, Caption
from sparx_agency.tools.pdf_parser.layout import PageLayout
from sparx_agency.tools.pdf_parser.regions import Region

MIN_LISTING_LINES = 2
"""Fewest lines a region must hold to be worth writing out as a listing."""


@dataclass(frozen=True)
class Listing:
    """One algorithm or code block, as printed.

    Attributes:
        caption: The caption the listing was found from.
        body: The listing, indentation preserved, newline separated.
        line_count: Number of lines in the body.
    """

    caption: Caption
    body: str

    @property
    def line_count(self) -> int:
        """Number of lines in the listing."""
        return len(self.body.splitlines())

    def to_text(self) -> str:
        """Render for a standalone file: the caption, then the listing."""
        heading = "{} — {}".format(self.caption.label, self.caption.text).rstrip(" —")
        return "{}\n(page {})\n\n{}\n".format(heading, self.caption.page, self.body)


def extract_listing(page: PageLayout, region: Region) -> Optional[Listing]:
    """Rebuild the algorithm block a region covers.

    Args:
        page: The page the block is on.
        region: The region resolved from its caption.

    Returns:
        The :class:`Listing`, or None when the region holds too little to be one.
    """
    lines = regions_mod.lines_in(page, region)
    if len(lines) < MIN_LISTING_LINES:
        return None

    body = textgrid.render_lines(lines, left_pt=min(line.bbox.x_min for line in lines))
    if not body.strip():
        return None
    return Listing(caption=region.caption, body=body)


def extract_listings(page: PageLayout, page_regions: Sequence[Region]) -> List[Listing]:
    """Rebuild every algorithm block on a page.

    Args:
        page: The parsed page.
        page_regions: Regions found on it; non-algorithm regions are ignored.

    Returns:
        One :class:`Listing` per region that held a block.
    """
    extracted = (
        extract_listing(page, region)
        for region in page_regions
        if region.caption.kind == ALGORITHM
    )
    return [listing for listing in extracted if listing is not None]
