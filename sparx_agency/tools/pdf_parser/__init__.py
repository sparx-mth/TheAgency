"""Reading a research paper mechanically — text, figures, charts, tables, algorithms.

Built for the moment someone drops a paper into the team and asks what it means
for us. Reading it is the interesting part; getting at it is not, and doing that
by hand is where the errors come from — a number transposed out of a results
table, a figure skipped because the caption seemed to describe it, an algorithm
flattened into a list of statements with its loop structure gone.

So the mechanical half is done here, and it produces one directory per paper:
the text with its columns intact and split by page so a quote can carry a page
number, an image of every page, an enlarged image of every labelled figure,
every table as cells you can check against a picture of the same table, and
every algorithm with its indentation rebuilt.

Everything is built on the poppler command line tools, which are already
installed everywhere in this project, and on nothing else — no PDF library, no
model, no network beyond fetching the paper itself. It runs in any interpreter
in the repository, including the Noetic container's 3.8.

Typical use::

    from pathlib import Path
    from sparx_agency.tools.pdf_parser import extract_paper, PaperWorkspace

    workspace = PaperWorkspace.for_slug("navdp")
    result = extract_paper(Path("~/Downloads/navdp.pdf").expanduser(), workspace)
    print(result.workspace.manifest)

or from a shell::

    python -m sparx_agency.tools.pdf_parser extract 2505.08712

The one thing to keep in mind while reading what comes out: a PDF does not
record where a figure begins or that a table is a table. Those are inferred, by
the rules in :mod:`regions` and :mod:`columns`. So every inference is published
next to a picture of what it was drawn from, and where the two disagree the
picture wins.
"""
from __future__ import annotations

from sparx_agency.tools.pdf_parser.captions import (
    ALGORITHM,
    Caption,
    FIGURE,
    TABLE,
    find_all_captions,
    find_captions,
)
from sparx_agency.tools.pdf_parser.document import PdfDocument, open_pdf
from sparx_agency.tools.pdf_parser.extract import Exhibit, ExtractionResult, extract_paper
from sparx_agency.tools.pdf_parser.fetch import (
    DownloadFailed,
    PaperSource,
    arxiv_metadata,
    download_pdf,
    resolve_source,
)
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.image_extract import EmbeddedImage, extract_images
from sparx_agency.tools.pdf_parser.layout import Block, Line, PageLayout, Word, load_layout
from sparx_agency.tools.pdf_parser.links import PaperLinks, find_links
from sparx_agency.tools.pdf_parser.page_render import render_pages, render_region
from sparx_agency.tools.pdf_parser.prose import is_body_text, is_tabular
from sparx_agency.tools.pdf_parser.poppler import (
    PopplerFailed,
    PopplerNotInstalled,
    require_poppler,
)
from sparx_agency.tools.pdf_parser.pseudocode import Listing, extract_listing
from sparx_agency.tools.pdf_parser.regions import Region, find_region, find_regions
from sparx_agency.tools.pdf_parser.tables import Table, parse_table
from sparx_agency.tools.pdf_parser.text_extract import ExtractedText, extract_text
from sparx_agency.tools.pdf_parser.workspace import PaperWorkspace, slugify

__all__ = [
    "ALGORITHM",
    "BBox",
    "Block",
    "Caption",
    "DownloadFailed",
    "EmbeddedImage",
    "Exhibit",
    "ExtractedText",
    "ExtractionResult",
    "FIGURE",
    "Line",
    "Listing",
    "PageLayout",
    "PaperLinks",
    "PaperSource",
    "PaperWorkspace",
    "PdfDocument",
    "PopplerFailed",
    "PopplerNotInstalled",
    "Region",
    "TABLE",
    "Table",
    "Word",
    "arxiv_metadata",
    "download_pdf",
    "extract_images",
    "extract_listing",
    "extract_paper",
    "extract_text",
    "find_all_captions",
    "find_captions",
    "find_links",
    "find_region",
    "find_regions",
    "is_body_text",
    "is_tabular",
    "load_layout",
    "open_pdf",
    "parse_table",
    "render_pages",
    "render_region",
    "require_poppler",
    "resolve_source",
    "slugify",
]
