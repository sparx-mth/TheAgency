"""Rebuilding indentation from word positions."""
from __future__ import annotations

import pytest

from sparx_agency.tools.pdf_parser import textgrid
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.layout import Line, Word

CHAR_PT = 5.0


def word_at(text: str, x: float, y: float) -> Word:
    """One word laid out at five points per character."""
    return Word(text, BBox(x, y, x + len(text) * CHAR_PT, y + 10.0))


def line_of(*words: Word) -> Line:
    """A line around the given words."""
    return Line(
        list(words),
        BBox(
            min(word.bbox.x_min for word in words),
            min(word.bbox.y_min for word in words),
            max(word.bbox.x_max for word in words),
            max(word.bbox.y_max for word in words),
        ),
    )


def test_char_width_is_measured_from_the_text():
    line = line_of(word_at("abcd", 0.0, 0.0))
    assert textgrid.estimate_char_width([line]) == pytest.approx(CHAR_PT)


def test_char_width_falls_back_when_there_is_nothing_to_measure():
    assert textgrid.estimate_char_width([]) == textgrid.FALLBACK_CHAR_WIDTH_PT


def test_indentation_is_reconstructed():
    """The point of the module: nesting is the algorithm."""
    outer = line_of(word_at("for", 60.0, 100.0))
    inner = line_of(word_at("if", 80.0, 112.0))
    rendered = textgrid.render_lines([outer, inner], left_pt=60.0, char_width_pt=CHAR_PT)
    assert rendered.splitlines() == ["for", "    if"]


def test_lines_are_ordered_top_to_bottom():
    lower = line_of(word_at("second", 60.0, 200.0))
    upper = line_of(word_at("first", 60.0, 100.0))
    rendered = textgrid.render_lines([lower, upper], left_pt=60.0, char_width_pt=CHAR_PT)
    assert rendered.splitlines() == ["first", "second"]


def test_words_on_one_line_keep_their_spacing():
    line = line_of(word_at("1:", 60.0, 100.0), word_at("return", 100.0, 100.0))
    rendered = textgrid.render_line(line, left_pt=60.0, char_width_pt=CHAR_PT)
    assert rendered == "1:      return"


def test_words_never_overwrite_each_other():
    """A slightly wrong width estimate must not eat characters."""
    line = line_of(word_at("aaaa", 60.0, 100.0), word_at("bbbb", 62.0, 100.0))
    rendered = textgrid.render_line(line, left_pt=60.0, char_width_pt=CHAR_PT)
    assert "aaaa" in rendered and "bbbb" in rendered


def test_common_indent_is_removed_by_default():
    line = line_of(word_at("body", 200.0, 100.0))
    assert textgrid.render_lines([line]) == "body"


def test_no_lines_render_as_nothing():
    assert textgrid.render_lines([]) == ""
