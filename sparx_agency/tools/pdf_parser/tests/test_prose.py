"""Telling a paragraph from the inside of a figure or a table."""
from __future__ import annotations

from sparx_agency.tools.pdf_parser import prose
from sparx_agency.tools.pdf_parser.geometry import BBox
from sparx_agency.tools.pdf_parser.layout import Block, Line, Word

COLUMN_WIDTH_PT = 440.0


def line_at(text: str, x: float = 60.0, y: float = 100.0, gap: float = 3.0) -> Line:
    """Lay a string out with ordinary prose spacing."""
    words = []
    cursor = x
    for token in text.split():
        width = len(token) * 5.0
        words.append(Word(token, BBox(cursor, y, cursor + width, y + 10.0)))
        cursor += width + gap
    return Line(words, BBox(x, y, max(cursor - gap, x + 1.0), y + 10.0))


def row_at(cells, starts, y: float = 100.0) -> Line:
    """Lay cells out at fixed column positions, with gutters between them."""
    words = []
    for cell, start in zip(cells, starts):
        words.append(Word(cell, BBox(start, y, start + len(cell) * 5.0, y + 10.0)))
    return Line(words, BBox(starts[0], y, starts[-1] + len(cells[-1]) * 5.0, y + 10.0))


def block_of(*lines: Line) -> Block:
    """A block around the given lines."""
    return Block(
        list(lines),
        BBox(
            min(line.bbox.x_min for line in lines),
            min(line.bbox.y_min for line in lines),
            max(line.bbox.x_max for line in lines),
            max(line.bbox.y_max for line in lines),
        ),
    )


PARAGRAPH = (
    "the model learns a policy over observations and produces actions which the "
    "controller then tracks along the planned route"
)

NARROW_LABELS = ("multi head dot product norm gate", "scaled cross attend key value",
                 "output logits soft max top one")
"""Three short lines: enough words to look like prose, too narrow to be it."""


def narrow_label_block() -> Block:
    """A wordy block that only fails the line-fill test."""
    return block_of(
        *[line_at(text, x=120.0, y=100.0 + 12.0 * index)
          for index, text in enumerate(NARROW_LABELS)]
    )


def test_a_paragraph_is_prose():
    block = block_of(line_at(PARAGRAPH))
    assert prose.is_body_text(block, COLUMN_WIDTH_PT)


def test_a_table_row_is_not_prose():
    """Caught by word spacing: gutters are far wider than a sentence's spaces."""
    block = block_of(row_at(["Baseline", "71.2", "18ms"], [60.0, 250.0, 430.0]))
    assert prose.is_tabular(block)
    assert not prose.is_body_text(block, COLUMN_WIDTH_PT)


def test_scattered_figure_labels_are_not_prose():
    """Caught by words per line: enough words in total, one or two per line.

    The case that made a full-page architecture diagram invisible.
    """
    labels = (
        "Encoder Decoder Softmax Linear Embedding Attention Norm Feed Forward "
        "Masked Multi Head Output Input Positional Probabilities"
    ).split()
    block = block_of(
        *[line_at(word, x=120.0, y=100.0 + 12.0 * index)
          for index, word in enumerate(labels)]
    )
    assert block.word_count >= prose.MIN_WORDS, "fixture must clear the length test"
    assert not prose.is_tabular(block), "fixture must clear the spacing test"
    assert not prose.is_body_text(block, COLUMN_WIDTH_PT)


def test_a_narrow_column_of_wordy_labels_is_not_prose():
    """Caught by line fill: prose reaches across its column, labels do not."""
    block = narrow_label_block()
    assert block.word_count >= prose.MIN_WORDS, "fixture must clear the length test"
    assert block.word_count / len(block.lines) >= prose.MIN_WORDS_PER_LINE
    assert not prose.is_body_text(block, COLUMN_WIDTH_PT)


def test_the_line_fill_test_is_skipped_without_a_column_width():
    """Callers that do not know the column still get the other three tests."""
    assert prose.is_body_text(narrow_label_block())


def test_a_short_block_is_never_prose():
    assert not prose.is_body_text(block_of(line_at("A short caption.")), COLUMN_WIDTH_PT)


def test_an_empty_block_is_neither():
    empty = Block([], BBox(0.0, 0.0, 1.0, 1.0))
    assert not prose.is_tabular(empty)
    assert not prose.is_body_text(empty, COLUMN_WIDTH_PT)
