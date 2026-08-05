"""End-to-end tests against real PDFs, built for the purpose.

Everything here runs the actual poppler toolchain over an actual file, because
the heuristics under test are all answers to "where did poppler say that word
is". The PDFs are written by :mod:`minimal_pdf` rather than committed, so the
suite carries no binary fixtures and no third party's copyright.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from sparx_agency.tools.pdf_parser import (
    captions as captions_mod,
    extract as extract_mod,
    layout as layout_mod,
    poppler,
    pseudocode,
    regions as regions_mod,
    tables as tables_mod,
    text_extract,
)
from sparx_agency.tools.pdf_parser.tests.minimal_pdf import TextItem, build_pdf, lorem
from sparx_agency.tools.pdf_parser.workspace import PaperWorkspace

pytestmark = pytest.mark.skipif(
    bool(poppler.missing_binaries()),
    reason="poppler-utils not installed: {}".format(", ".join(poppler.missing_binaries())),
)

TABLE_COLUMN_X = (60.0, 250.0, 430.0)


def figure_page() -> List[TextItem]:
    """A page shaped like a figure page: prose, artwork labels, caption, prose."""
    return [
        TextItem(60.0, 80.0, lorem(14)),
        TextItem(120.0, 170.0, "Encoder"),
        TextItem(280.0, 170.0, "Decoder"),
        TextItem(200.0, 210.0, "Softmax"),
        TextItem(60.0, 320.0, "Figure 1: The architecture of the proposed system."),
        TextItem(60.0, 400.0, lorem(14, seed="beta")),
    ]


def table_page() -> List[TextItem]:
    """A page shaped like a results page: caption, a three-column table, prose."""
    items = [TextItem(60.0, 80.0, "Table 1: Results on the benchmark.")]
    rows = (
        ("Method", "Success", "Latency"),
        ("Baseline", "71.2", "18 ms"),
        ("Ours", "88.4", "21 ms"),
        ("Ours large", "91.0", "34 ms"),
    )
    for index, row in enumerate(rows):
        y = 130.0 + 20.0 * index
        items += [TextItem(x, y, cell) for x, cell in zip(TABLE_COLUMN_X, row)]
    items.append(TextItem(60.0, 300.0, lorem(14, seed="gamma")))
    return items


def algorithm_page() -> List[TextItem]:
    """A page shaped like an algorithm page: caption, an indented listing, prose."""
    return [
        TextItem(60.0, 80.0, "Algorithm 1: Frontier selection."),
        TextItem(60.0, 110.0, "1: while frontiers remain do"),
        TextItem(90.0, 130.0, "2: pick the nearest frontier"),
        TextItem(120.0, 150.0, "3: plan a path to it"),
        TextItem(60.0, 170.0, "4: end while"),
        TextItem(60.0, 260.0, lorem(14, seed="delta")),
    ]


@pytest.fixture(scope="module")
def paper(tmp_path_factory) -> Path:
    """A three-page PDF holding a figure, a table and an algorithm."""
    path = tmp_path_factory.mktemp("pdf_parser") / "paper.pdf"
    path.write_bytes(build_pdf([figure_page(), table_page(), algorithm_page()]))
    return path


@pytest.fixture(scope="module")
def pages(paper: Path) -> List[layout_mod.PageLayout]:
    """The parsed layout of every page."""
    return layout_mod.load_layout(paper)


def region_on(page: layout_mod.PageLayout) -> regions_mod.Region:
    """Resolve the single caption on a page to its region."""
    found = captions_mod.find_captions(page)
    assert len(found) == 1, "expected exactly one caption, got {}".format(
        [caption.label for caption in found]
    )
    region = regions_mod.find_region(page, found[0], found)
    assert region is not None, "region did not resolve"
    return region


def test_layout_reports_one_entry_per_page(pages):
    assert len(pages) == 3
    assert [page.number for page in pages] == [1, 2, 3]


def test_text_is_extracted_per_page(paper: Path):
    text = text_extract.extract_text(paper)
    assert len(text.pages) == 3
    assert "Figure 1" in text.page(1)
    assert not text.is_scanned


def test_captions_are_found_on_the_right_pages(pages):
    found = captions_mod.find_all_captions(pages)
    assert [(caption.label, caption.page) for caption in found] == [
        ("Figure 1", 1),
        ("Table 1", 2),
        ("Algorithm 1", 3),
    ]


def test_a_figure_region_grows_upward_from_its_caption(pages):
    region = region_on(pages[0])
    assert region.side == regions_mod.ABOVE
    assert region.content.y_max <= region.caption.bbox.y_min + 1.0


def test_a_figure_region_holds_the_artwork_and_stops_at_the_prose_above(pages):
    """The whole job of the region finder, in one assertion."""
    region = region_on(pages[0])
    assert region.content.y_min > 82.0, "swallowed the paragraph above the figure"
    assert region.content.y_min < 165.0, "cut off the topmost artwork label"
    assert region.content.y_max > 215.0, "cut off the lowest artwork label"


def test_a_table_region_grows_downward_from_its_caption(pages):
    region = region_on(pages[1])
    assert region.side in (regions_mod.BELOW, regions_mod.INSIDE)
    assert region.content.y_max > 180.0


def test_a_table_parses_into_the_right_grid(pages):
    table = tables_mod.parse_table(pages[1], region_on(pages[1]))
    assert table is not None
    assert table.column_count == 3
    assert table.header == ["Method", "Success", "Latency"]
    assert table.rows == [
        ["Baseline", "71.2", "18 ms"],
        ["Ours", "88.4", "21 ms"],
        ["Ours large", "91.0", "34 ms"],
    ]


def test_a_table_renders_to_markdown_and_csv(pages):
    table = tables_mod.parse_table(pages[1], region_on(pages[1]))
    markdown = table.to_markdown()
    assert "| Method | Success | Latency |" in markdown
    assert "Table 1" in markdown
    assert table.to_csv().splitlines()[0] == "Method,Success,Latency"


def test_only_table_captions_are_parsed_as_grids(pages):
    """A figure's scattered labels are geometrically a grid; they are not a table.

    What keeps them out of ``tables/`` is the caption kind, not the geometry, so
    that is what is asserted here.
    """
    figure_page_regions = regions_mod.find_regions(
        pages[0], captions_mod.find_captions(pages[0])
    )
    assert tables_mod.parse_tables(pages[0], figure_page_regions) == []
    assert pseudocode.extract_listings(pages[0], figure_page_regions) == []


def test_an_algorithm_keeps_its_indentation(pages):
    listing = pseudocode.extract_listing(pages[2], region_on(pages[2]))
    assert listing is not None
    body = listing.body.splitlines()
    assert body[0].startswith("1: while")
    assert body[1].startswith(" "), "second step lost its indent"
    assert len(body[2]) - len(body[2].lstrip()) > len(body[1]) - len(body[1].lstrip())


def test_extract_paper_produces_the_whole_workspace(tmp_path: Path, paper: Path):
    workspace = PaperWorkspace(tmp_path / "ws")
    result = extract_mod.extract_paper(paper, workspace)

    assert result.document.page_count == 3
    assert len(result.page_images) == 3
    assert workspace.full_text.exists()
    assert len(list(workspace.page_text_dir.glob("p*.txt"))) == 3
    assert workspace.manifest.exists()
    assert workspace.captions_json.exists()


def test_extract_paper_writes_one_crop_per_exhibit(tmp_path: Path, paper: Path):
    result = extract_mod.extract_paper(paper, PaperWorkspace(tmp_path / "ws"))
    assert len(result.exhibits) == 3
    assert all(exhibit.image is not None for exhibit in result.exhibits)
    assert all(exhibit.image.stat().st_size > 0 for exhibit in result.exhibits)


def test_extract_paper_writes_the_table_and_the_listing(tmp_path: Path, paper: Path):
    result = extract_mod.extract_paper(paper, PaperWorkspace(tmp_path / "ws"))
    table = result.of_kind(captions_mod.TABLE)[0]
    listing = result.of_kind(captions_mod.ALGORITHM)[0]
    assert table.markdown is not None and table.csv is not None
    assert "88.4" in table.markdown.read_text(encoding="utf-8")
    assert listing.listing is not None
    assert "while frontiers remain" in listing.listing.read_text(encoding="utf-8")


def test_the_manifest_names_every_exhibit(tmp_path: Path, paper: Path):
    result = extract_mod.extract_paper(paper, PaperWorkspace(tmp_path / "ws"))
    manifest = result.workspace.manifest.read_text(encoding="utf-8")
    for label in ("Figure 1", "Table 1", "Algorithm 1"):
        assert label in manifest


def test_re_running_does_not_leave_stale_output(tmp_path: Path, paper: Path):
    workspace = PaperWorkspace(tmp_path / "ws").create()
    extract_mod.extract_paper(paper, workspace)
    stale = workspace.figure_dir / "figure-99.png"
    stale.write_bytes(b"stale")

    extract_mod.extract_paper(paper, workspace)
    assert not stale.exists()


def test_a_pdf_with_no_text_is_reported_as_scanned(tmp_path: Path):
    """A scanned paper must say so rather than look like an empty one."""
    blank = tmp_path / "scan.pdf"
    blank.write_bytes(build_pdf([[TextItem(60.0, 80.0, "1")]]))
    result = extract_mod.extract_paper(blank, PaperWorkspace(tmp_path / "ws"))
    assert result.text.is_scanned
    assert any("scanned" in warning for warning in result.warnings)


def test_an_html_error_page_saved_as_a_pdf_is_refused(tmp_path: Path):
    fake = tmp_path / "paper.pdf"
    fake.write_text("<html><body>403 Forbidden</body></html>", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_mod.extract_paper(fake, PaperWorkspace(tmp_path / "ws"))
