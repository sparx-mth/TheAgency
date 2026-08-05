"""Telling a caption from a sentence that mentions a figure."""
from __future__ import annotations

from typing import List

from sparx_agency.tools.pdf_parser import captions
from sparx_agency.tools.pdf_parser.geometry import BBox, union_all
from sparx_agency.tools.pdf_parser.layout import Block, Line, PageLayout, Word

CHAR_PT = 5.0
LINE_HEIGHT_PT = 10.0


def make_line(text: str, x: float = 60.0, y: float = 100.0, gap: float = 3.0) -> Line:
    """Lay a string out as words with ordinary prose spacing."""
    words: List[Word] = []
    cursor = x
    for token in text.split():
        width = len(token) * CHAR_PT
        words.append(Word(token, BBox(cursor, y, cursor + width, y + LINE_HEIGHT_PT)))
        cursor += width + gap
    return Line(words, BBox(x, y, max(cursor - gap, x + 1.0), y + LINE_HEIGHT_PT))


def make_block(*lines: Line) -> Block:
    """A block around the given lines."""
    return Block(list(lines), union_all(line.bbox for line in lines))


def page_of(*blocks: Block) -> PageLayout:
    """A single page holding the given blocks."""
    return PageLayout(1, 612.0, 792.0, list(blocks))


def test_a_figure_caption_is_found():
    page = page_of(make_block(make_line("Figure 2: The architecture of the model.")))
    found = captions.find_captions(page)
    assert len(found) == 1
    assert found[0].kind == captions.FIGURE
    assert found[0].number == "2"
    assert found[0].text == "The architecture of the model."


def test_a_cross_reference_in_prose_is_not_a_caption():
    """The rule that keeps the parser from cropping random paragraphs."""
    page = page_of(make_block(make_line("Figure 2 shows the architecture of the model.")))
    assert captions.find_captions(page) == []


def test_abbreviated_and_full_stop_styles():
    page = page_of(
        make_block(make_line("Fig. 1. Overview of the system.", y=100.0)),
        make_block(make_line("Table III: Results on the benchmark.", y=200.0)),
        make_block(make_line("Algorithm 1: Frontier selection.", y=300.0)),
    )
    found = captions.find_captions(page)
    assert [caption.kind for caption in found] == [
        captions.FIGURE,
        captions.TABLE,
        captions.ALGORITHM,
    ]
    assert found[1].number == "III"


def test_appendix_style_numbers():
    page = page_of(make_block(make_line("Figure A1: Extra results.")))
    assert captions.find_captions(page)[0].number == "A1"


def test_labels_and_slugs():
    page = page_of(make_block(make_line("Table 3.1: Ablations.")))
    caption = captions.find_captions(page)[0]
    assert caption.label == "Table 3.1"
    assert caption.slug == "table-3.1"


def test_a_wrapped_caption_keeps_both_lines():
    block = make_block(
        make_line("Figure 4: Two attention heads, also in layer five,", y=100.0),
        make_line("apparently involved in anaphora resolution.", y=112.0),
    )
    caption = captions.find_captions(page_of(block))[0]
    assert caption.text.endswith("anaphora resolution.")
    assert "layer five" in caption.text


def test_a_caption_merged_with_its_table_stops_at_the_table():
    """Poppler sometimes puts a caption and its table in one block."""
    caption_line = make_line("Table 4: Results on the benchmark.", y=100.0)
    left = make_line("Parser", x=60.0, y=130.0)
    right = make_line("91.3", x=400.0, y=130.0)
    block = make_block(caption_line, left, right)

    assert captions.caption_line_count(block) == 1
    caption = captions.find_captions(page_of(block))[0]
    assert caption.text == "Results on the benchmark."
    assert "Parser" not in caption.text


def test_caption_index_records_the_page():
    page = PageLayout(7, 612.0, 792.0, [make_block(make_line("Figure 9: A plot."))])
    caption = captions.find_all_captions([page])[0]
    assert caption.page == 7
    assert caption.to_dict()["page"] == 7
