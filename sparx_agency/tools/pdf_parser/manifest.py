"""The note the extractor leaves for whoever opens the workspace next.

Its job is to say what is here, and — more usefully — what is *not*. Every
figure this package failed to locate, every table that did not resolve into a
grid, every page with no text on it is listed, because the alternative is a
reader who assumes the workspace is complete and quietly works from three
quarters of a paper.

The words-per-page table earns its place too: a page with almost no words is a
full-page figure or a results table, and knowing which pages those are is the
fastest way into a paper you have not read.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from sparx_agency.tools.pdf_parser.extract import ExtractionResult

LOW_TEXT_PAGE_WORDS = 200
"""Words below which a page is flagged as mostly figure or table.

A full page of body text in this literature runs 400-600 words, so this is
roughly "less than half a page of prose". Set lower it flags nothing: a paper's
qualitative-results pages still carry a caption and a paragraph, and come in
around 150.

This matters most for the figures nobody labelled, which the caption index
cannot see at all. A page that is nearly all picture is the only signal that one
is there.
"""


def _header(result: "ExtractionResult") -> List[str]:
    """The identifying block at the top of the manifest."""
    document = result.document
    title = document.title or "(no title embedded — read it off page 1)"
    return [
        "# Extraction manifest",
        "",
        "- **title (from PDF metadata):** {}".format(title),
        "- **source:** `{}`".format(document.path),
        "- **pages:** {}".format(document.page_count),
        "- **page size:** {:.0f} x {:.0f} pt".format(document.width, document.height),
        "- **producer:** {}".format(document.producer or "unknown"),
        "- **text extracted:** {} characters".format(result.text.char_count),
        "",
    ]


def _warnings(result: "ExtractionResult") -> List[str]:
    """Everything that went wrong, or a line saying nothing did."""
    if not result.warnings:
        return []
    lines = ["## Warnings", ""]
    lines += ["- {}".format(warning) for warning in result.warnings]
    lines.append("")
    return lines


def _orientation() -> List[str]:
    """A short guide to the directory, for a reader who has not seen one before."""
    return [
        "## How to read this workspace",
        "",
        "| you want | open |",
        "|---|---|",
        "| exact wording to quote | `text/pages/pNNN.txt` — the filename is the page number |",
        "| to see a page as printed | `pages/p-NN.png` |",
        "| one figure or chart, enlarged | `figures/figure-N.png` |",
        "| a table's numbers | `tables/table-N.md`, checked against `figures/table-N.png` |",
        "| an algorithm, indentation intact | `pseudocode/algorithm-N.txt` |",
        "| the paper's own code link | `links.json` |",
        "",
        "Figures are cropped by inference, not by anything recorded in the PDF.",
        "Where a crop looks wrong, the page render beside it is the ground truth.",
        "",
    ]


def _pages(result: "ExtractionResult") -> List[str]:
    """Per-page word counts, with the sparse pages called out."""
    lines = ["## Pages", "", "| page | words | note |", "|---|---|---|"]
    for number, page_text in enumerate(result.text.pages, start=1):
        words = len(page_text.split())
        note = "mostly figure or table — look at the render" if words < LOW_TEXT_PAGE_WORDS else ""
        lines.append("| {} | {} | {} |".format(number, words, note))
    lines.append("")
    return lines


def _exhibit_row(result: "ExtractionResult", exhibit) -> str:
    """One row of the exhibits table."""
    workspace = result.workspace
    produced: List[str] = []
    if exhibit.image:
        produced.append("`{}`".format(workspace.relative(exhibit.image)))
    if exhibit.markdown:
        produced.append("`{}`".format(workspace.relative(exhibit.markdown)))
    if exhibit.listing:
        produced.append("`{}`".format(workspace.relative(exhibit.listing)))
    caption = exhibit.caption.text
    if len(caption) > 90:
        caption = caption[:87] + "..."
    return "| {} | {} | {} | {} | {} |".format(
        exhibit.caption.label,
        exhibit.caption.page,
        caption.replace("|", "\\|"),
        ", ".join(produced) or "—",
        exhibit.note or "",
    )


def _exhibits(result: "ExtractionResult") -> List[str]:
    """The figure/table/algorithm index."""
    if not result.exhibits:
        return [
            "## Figures, tables and algorithms",
            "",
            "None found. Either the paper labels its exhibits in a way this parser",
            "does not recognise, or it has no text layer. Read `pages/*.png`.",
            "",
        ]

    counts = {}
    for exhibit in result.exhibits:
        counts[exhibit.caption.kind] = counts.get(exhibit.caption.kind, 0) + 1
    summary = ", ".join("{} {}s".format(count, kind) for kind, count in sorted(counts.items()))

    lines = [
        "## Figures, tables and algorithms",
        "",
        "Found {}.".format(summary),
        "",
        "| exhibit | page | caption | extracted | note |",
        "|---|---|---|---|---|",
    ]
    lines += [_exhibit_row(result, exhibit) for exhibit in result.exhibits]
    lines.append("")
    return lines


def _embedded(result: "ExtractionResult") -> List[str]:
    """What came out of the PDF as stored rasters."""
    lines = ["## Embedded images", ""]
    if result.images is None:
        lines += ["Not extracted.", ""]
        return lines
    if not result.images.figures:
        lines += [
            "None large enough to be figures. This is normal and expected: plots and",
            "diagrams in this literature are vector drawings, so there is no image",
            "inside the PDF to lift out. Use the crops in `figures/` instead.",
            "",
        ]
        return lines

    lines += [
        "{} figure-sized, {} set aside as rules and glyphs in `figures/embedded/_small/`.".format(
            len(result.images.figures), len(result.images.set_aside)
        ),
        "",
        "| file | page | pixels |",
        "|---|---|---|",
    ]
    for image in result.images.figures:
        lines.append(
            "| `{}` | {} | {}x{} |".format(
                result.workspace.relative(image.path), image.page, image.width, image.height
            )
        )
    lines.append("")
    return lines


def _links(result: "ExtractionResult") -> List[str]:
    """Code, model and reference links printed in the paper."""
    lines = ["## Links printed in the paper", ""]
    found = result.links
    if found is None or found.is_empty:
        lines += [
            "None. That does not mean there is no code — search the web for the title",
            "before concluding it was never released.",
            "",
        ]
        return lines

    for label, values in (
        ("Repositories", found.repositories),
        ("arXiv", found.arxiv_ids),
        ("DOI", found.dois),
        ("Project pages", found.project_pages),
    ):
        if values:
            lines.append("**{}**".format(label))
            lines.append("")
            lines += ["- {}".format(value) for value in values]
            lines.append("")
    lines += [
        "Links are read out of the text, including a pass that repairs addresses split",
        "across lines, so one may be slightly wrong. Open before trusting.",
        "",
    ]
    return lines


def render_manifest(result: "ExtractionResult") -> str:
    """Render the whole manifest as markdown.

    Args:
        result: What :func:`sparx_agency.tools.pdf_parser.extract.extract_paper`
            returned.

    Returns:
        The manifest text, ready to write to ``MANIFEST.md``.
    """
    sections = (
        _header(result)
        + _warnings(result)
        + _orientation()
        + _exhibits(result)
        + _pages(result)
        + _embedded(result)
        + _links(result)
    )
    return "\n".join(sections)
