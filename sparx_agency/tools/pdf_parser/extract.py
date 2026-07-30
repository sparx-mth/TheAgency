"""One call that turns a PDF into a workspace.

The order here is not arbitrary. Text and page renders come first and never
fail on anything that is a PDF at all, so even a scanned paper or one whose
layout defeats every heuristic below still produces a workspace worth reading.
Everything after that — captions, regions, crops, tables, listings — is
inference, and each piece is allowed to fail on its own without taking the rest
with it. What failed is recorded and printed in the manifest rather than
swallowed, because a workspace that quietly contains three of a paper's nine
figures is worse than one that says so.

The result is a directory a person or an agent can work from directly: the text
to quote, an image of every page, an image of every labelled exhibit, the tables
as cells, and the algorithms with their indentation intact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from sparx_agency.tools.pdf_parser import (
    captions as captions_mod,
    document as document_mod,
    image_extract,
    layout as layout_mod,
    links as links_mod,
    manifest as manifest_mod,
    page_render,
    poppler,
    pseudocode,
    regions as regions_mod,
    tables as tables_mod,
    text_extract,
)
from sparx_agency.tools.pdf_parser.captions import ALGORITHM, Caption, TABLE
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.workspace import PaperWorkspace


@dataclass
class Exhibit:
    """One labelled figure, table or algorithm and everything produced from it.

    Attributes:
        caption: The caption it was found from.
        image: The rendered crop, or None if the region could not be resolved.
        markdown: Parsed table as markdown, or None.
        csv: Parsed table as CSV, or None.
        listing: Algorithm text, or None.
        note: Why a piece is missing, when it is.
    """

    caption: Caption
    image: Optional[Path] = None
    markdown: Optional[Path] = None
    csv: Optional[Path] = None
    listing: Optional[Path] = None
    note: Optional[str] = None


@dataclass
class ExtractionResult:
    """Everything one extraction produced.

    Attributes:
        document: The source PDF's metadata.
        workspace: Where it all went.
        text: The extracted text, whole and per page.
        page_images: One render per page, in page order.
        exhibits: One per caption found, in document order.
        images: Embedded rasters, sorted into figures and noise.
        links: Code, model and reference links found in the text.
        warnings: Anything that went wrong but did not stop the extraction.
    """

    document: document_mod.PdfDocument
    workspace: PaperWorkspace
    text: text_extract.ExtractedText
    page_images: List[Path] = field(default_factory=list)
    exhibits: List[Exhibit] = field(default_factory=list)
    images: Optional[image_extract.ImageExtraction] = None
    links: Optional[links_mod.PaperLinks] = None
    warnings: List[str] = field(default_factory=list)

    def of_kind(self, kind: str) -> List[Exhibit]:
        """Return the exhibits of one caption kind."""
        return [exhibit for exhibit in self.exhibits if exhibit.caption.kind == kind]


def _unique_path(directory: Path, stem: str, suffix: str, page: int) -> Path:
    """Build a path, disambiguating by page then by counter on collision.

    Papers restart their numbering in the appendix, so "Figure 1" can genuinely
    occur twice. Neither one may overwrite the other.
    """
    candidate = directory / "{}{}".format(stem, suffix)
    if not candidate.exists():
        return candidate
    candidate = directory / "{}-p{}{}".format(stem, page, suffix)
    counter = 2
    while candidate.exists():
        candidate = directory / "{}-p{}-{}{}".format(stem, page, counter, suffix)
        counter += 1
    return candidate


def _render_exhibit(
    pdf_path: Path, page: layout_mod.PageLayout, region: regions_mod.Region, workspace: PaperWorkspace
) -> Path:
    """Render one exhibit's region to its own PNG."""
    out_path = _unique_path(
        workspace.figure_dir, region.caption.slug, ".png", region.caption.page
    )
    return page_render.render_region(
        pdf_path=pdf_path,
        page_number=page.number,
        region=region.bbox,
        out_path=out_path,
        page_bounds=BBox(0.0, 0.0, page.width, page.height),
    )


def _write_table(table: tables_mod.Table, workspace: PaperWorkspace) -> tuple:
    """Write a parsed table as markdown and CSV, returning both paths."""
    markdown_path = _unique_path(
        workspace.table_dir, table.caption.slug, ".md", table.caption.page
    )
    markdown_path.write_text(table.to_markdown(), encoding="utf-8")
    csv_path = markdown_path.with_suffix(".csv")
    csv_path.write_text(table.to_csv(), encoding="utf-8")
    return markdown_path, csv_path


def _write_listing(listing: pseudocode.Listing, workspace: PaperWorkspace) -> Path:
    """Write an algorithm block to its own text file."""
    path = _unique_path(
        workspace.pseudocode_dir, listing.caption.slug, ".txt", listing.caption.page
    )
    path.write_text(listing.to_text(), encoding="utf-8")
    return path


def _exhibit_from_region(
    pdf_path: Path,
    page: layout_mod.PageLayout,
    region: regions_mod.Region,
    workspace: PaperWorkspace,
    warnings: List[str],
) -> Exhibit:
    """Produce the crop, and the cells or listing, for one resolved region."""
    exhibit = Exhibit(caption=region.caption)
    try:
        exhibit.image = _render_exhibit(pdf_path, page, region, workspace)
    except (poppler.PopplerFailed, ValueError) as exc:
        exhibit.note = "crop failed: {}".format(exc)
        warnings.append("{} on page {}: {}".format(region.caption.label, page.number, exc))

    if region.caption.kind == TABLE:
        table = tables_mod.parse_table(page, region)
        if table is None:
            exhibit.note = "no grid found — read the crop instead"
        else:
            exhibit.markdown, exhibit.csv = _write_table(table, workspace)
    elif region.caption.kind == ALGORITHM:
        listing = pseudocode.extract_listing(page, region)
        if listing is None:
            exhibit.note = "no listing text found — read the crop instead"
        else:
            exhibit.listing = _write_listing(listing, workspace)
    return exhibit


def _process_page(
    pdf_path: Path,
    page: layout_mod.PageLayout,
    workspace: PaperWorkspace,
    warnings: List[str],
) -> List[Exhibit]:
    """Find and extract every labelled exhibit on one page."""
    page_captions = captions_mod.find_captions(page)
    exhibits: List[Exhibit] = []
    for caption in page_captions:
        region = regions_mod.find_region(page, caption, page_captions)
        if region is None:
            exhibits.append(
                Exhibit(caption=caption, note="region not resolved — see the page render")
            )
            continue
        exhibits.append(_exhibit_from_region(pdf_path, page, region, workspace, warnings))
    return exhibits


def _write_index_files(result: ExtractionResult) -> None:
    """Write ``captions.json`` and ``links.json``."""
    workspace = result.workspace
    workspace.captions_json.write_text(
        json.dumps(
            [exhibit.caption.to_dict() for exhibit in result.exhibits], indent=2
        ),
        encoding="utf-8",
    )
    if result.links is not None:
        workspace.links_json.write_text(
            json.dumps(result.links.to_dict(), indent=2), encoding="utf-8"
        )


def extract_paper(
    pdf_path: Path,
    workspace: PaperWorkspace,
    render_pages: bool = True,
    extract_embedded: bool = True,
    clear: bool = True,
) -> ExtractionResult:
    """Extract a paper into a workspace.

    Args:
        pdf_path: The PDF to read. Copied to ``workspace.pdf`` if it is not
            already there, so the workspace stays self-contained.
        workspace: Where to write everything.
        render_pages: Render every page to PNG. Leave on; this is what makes
            figures readable at all.
        extract_embedded: Also lift out raster images stored in the PDF.
        clear: Delete output from a previous run first.

    Returns:
        The populated :class:`ExtractionResult`. ``MANIFEST.md`` is written as a
        side effect.

    Raises:
        PopplerNotInstalled: If poppler is not on ``PATH``.
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF.
    """
    poppler.require_poppler()
    workspace.create()
    if clear:
        workspace.clear_outputs()

    source = Path(pdf_path).expanduser().resolve()
    poppler.check_pdf(source)
    if source != workspace.pdf.resolve():
        workspace.pdf.write_bytes(source.read_bytes())

    document = document_mod.open_pdf(workspace.pdf)
    text = text_extract.extract_text(workspace.pdf)
    text_extract.write_text(text, workspace.full_text, workspace.page_text_dir)

    result = ExtractionResult(document=document, workspace=workspace, text=text)
    result.links = links_mod.find_links(text.full)

    if text.is_scanned:
        result.warnings.append(
            "no usable text layer ({} chars over {} pages) — this paper is scanned; "
            "read pages/*.png, and expect no tables or listings".format(
                text.char_count, document.page_count
            )
        )

    if render_pages:
        result.page_images = page_render.render_pages(
            workspace.pdf, workspace.page_image_dir
        )

    try:
        pages = layout_mod.load_layout(workspace.pdf)
    except (ValueError, poppler.PopplerFailed) as exc:
        pages = []
        result.warnings.append("layout unavailable, no exhibits extracted: {}".format(exc))

    for page in pages:
        result.exhibits += _process_page(workspace.pdf, page, workspace, result.warnings)

    if extract_embedded:
        result.images = image_extract.extract_images(workspace.pdf, workspace.embedded_dir)
        if result.images.note:
            result.warnings.append(result.images.note)

    _write_index_files(result)
    workspace.manifest.write_text(manifest_mod.render_manifest(result), encoding="utf-8")
    return result
