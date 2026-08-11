"""Parsing poppler's bbox XML, and putting split table rows back together."""
from __future__ import annotations

import pytest

from sparx_agency.tools.pdf_parser import layout
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.layout import Line, Word

XML = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>
  <page width="612.000000" height="792.000000">
    <flow><block xMin="60" yMin="100" xMax="300" yMax="112">
      <line xMin="60" yMin="100" xMax="300" yMax="112">
        <word xMin="60" yMin="100" xMax="120" yMax="112">Scaled</word>
        <word xMin="126" yMin="100" xMax="300" yMax="112">Attention</word>
      </line>
    </block></flow>
    <flow><block xMin="60" yMin="200" xMax="300" yMax="212">
      <line xMin="60" yMin="200" xMax="300" yMax="212">
        <word xMin="60" yMin="200" xMax="300" yMax="212">Second</word>
      </line>
    </block></flow>
  </page>
</doc></body></html>
"""


def test_pages_and_dimensions_are_read():
    pages = layout.parse_layout_xml(XML)
    assert len(pages) == 1
    assert pages[0].number == 1
    assert pages[0].width == 612.0
    assert pages[0].height == 792.0


def test_words_lines_and_blocks_are_nested():
    page = layout.parse_layout_xml(XML)[0]
    assert len(page.blocks) == 2
    assert page.blocks[0].lines[0].text == "Scaled Attention"
    assert page.blocks[0].word_count == 2
    assert [word.text for word in page.blocks[0].words()] == ["Scaled", "Attention"]


def test_page_text_joins_blocks():
    page = layout.parse_layout_xml(XML)[0]
    assert page.text == "Scaled Attention\n\nSecond"


def test_text_extent_spans_every_block():
    page = layout.parse_layout_xml(XML)[0]
    assert page.text_extent() == BBox(60.0, 100.0, 300.0, 212.0)


def test_empty_page_has_no_text_extent():
    page = layout.parse_layout_xml(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>'
        '<page width="612" height="792"></page></doc></body></html>'
    )[0]
    assert page.text_extent() is None


def test_malformed_xml_raises():
    with pytest.raises(ValueError):
        layout.parse_layout_xml("this is not xml")


def test_control_characters_from_a_symbol_font_do_not_kill_the_page():
    """One unmapped maths glyph used to cost the whole document its exhibits."""
    page = layout.parse_layout_xml(XML.replace("Second", "\x14\x15"))[0]
    assert len(page.blocks) == 2
    assert page.blocks[1].lines[0].text == "��"


def test_sanitising_keeps_tabs_newlines_and_ordinary_text():
    text = "a\tb\nc\r\nd é ∑"
    assert layout.sanitise_control_characters(text) == text


def _line(x: float, y: float, text: str = "cell", width: float = 40.0) -> Line:
    """One line at a position, as a single word."""
    box = BBox(x, y, x + width, y + 10.0)
    return Line([Word(text, box)], box)


def test_group_rows_reunites_cells_on_one_baseline():
    """Poppler emits each table cell as its own line; a row must survive that."""
    lines = [_line(60.0, 100.0), _line(200.0, 100.5), _line(400.0, 99.7)]
    rows = layout.group_rows(lines)
    assert len(rows) == 1
    assert len(rows[0]) == 3


def test_group_rows_orders_cells_left_to_right():
    rows = layout.group_rows([_line(400.0, 100.0, "c"), _line(60.0, 100.0, "a")])
    assert [line.text for line in rows[0]] == ["a", "c"]


def test_group_rows_separates_distinct_rows():
    lines = [_line(60.0, 100.0), _line(200.0, 100.0), _line(60.0, 130.0)]
    rows = layout.group_rows(lines)
    assert [len(row) for row in rows] == [2, 1]


def test_group_rows_on_nothing():
    assert layout.group_rows([]) == []


def test_max_word_gap_distinguishes_prose_from_a_table_row():
    prose = Line(
        [
            Word("the", BBox(60.0, 100.0, 80.0, 110.0)),
            Word("model", BBox(83.0, 100.0, 120.0, 110.0)),
        ],
        BBox(60.0, 100.0, 120.0, 110.0),
    )
    row = Line(
        [
            Word("Model", BBox(60.0, 100.0, 100.0, 110.0)),
            Word("28.4", BBox(300.0, 100.0, 330.0, 110.0)),
        ],
        BBox(60.0, 100.0, 330.0, 110.0),
    )
    assert prose.max_word_gap < layout.WIDE_GAP_PT
    assert row.max_word_gap >= layout.WIDE_GAP_PT


def test_max_word_gap_of_a_single_word_is_zero():
    assert _line(60.0, 100.0).max_word_gap == 0.0
