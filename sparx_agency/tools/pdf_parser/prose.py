"""Telling running prose from the contents of an exhibit.

One question, asked constantly: is this block a paragraph, or is it part of a
figure, a table or an algorithm? It is what bounds every region — a figure ends
where the prose above it begins — so getting it wrong loses the figure entirely.

The obvious test, word count, is the one that fails. An architecture diagram
carries thirty words in its boxes and is not a paragraph; a two-line table row
carries six and is not one either. What actually separates them is *how the
words sit*:

- **Word spacing.** Justified prose stretches its spaces a little. A table
  gutter, or the space between a figure's scattered labels, is several times
  wider than any space inside a sentence.
- **Words per line.** Prose fills its lines. Labels come one or two to a line.
- **Line fill.** Prose reaches both margins on every line but the last. A
  diagram's labels reach neither.

A block has to pass all of them, plus a minimum length, to count as prose. Each
test catches a case the others miss, which is why they are all here: word
spacing alone lets a wordy diagram through, and words-per-line alone lets a
narrow quotation through.
"""
from __future__ import annotations

from sparx_agency.tools.pdf_parser.layout import WIDE_GAP_PT, Block

TABULAR_LINE_FRACTION = 0.5
"""Share of a block's lines that must have gutters for the block to be tabular."""

MIN_WORDS = 15
"""Words below which a block is too short to be running prose, whatever its spacing."""

MIN_WORDS_PER_LINE = 5.0
"""Mean words per line below which a block is labels rather than prose.

The test word count gets wrong. A figure with a dozen boxes in it carries plenty
of words, one or two to a line, and poppler groups them into a single block. By
word count that block is a paragraph; by words per line it is obviously not, and
mistaking it for one makes the figure above it invisible.
"""

MIN_LINE_FILL = 0.45
"""Mean share of the column width a block's lines must span to be prose.

The same discriminator from the other direction, and the one that survives a
diagram whose labels happen to be wordy.
"""


def is_tabular(block: Block) -> bool:
    """True when most of a block's lines have gutters rather than word spaces.

    Args:
        block: The block to classify.

    Returns:
        True for a table body or a row of scattered figure labels.
    """
    if not block.lines:
        return False
    wide = sum(1 for line in block.lines if line.max_word_gap >= WIDE_GAP_PT)
    return wide / len(block.lines) >= TABULAR_LINE_FRACTION


def is_body_text(block: Block, column_width: float = 0.0) -> bool:
    """True when a block is running prose, and so bounds an exhibit.

    Args:
        block: The block to classify.
        column_width: Width of the column the block sits in, in points. Pass it
            when known; the line-fill test is skipped without it.

    Returns:
        True for a paragraph, False for a table body, a figure's labels, an
        algorithm listing or anything too short to be a paragraph.
    """
    if not block.lines or block.word_count < MIN_WORDS:
        return False
    if is_tabular(block):
        return False
    if block.word_count / len(block.lines) < MIN_WORDS_PER_LINE:
        return False
    if column_width > 0.0:
        mean_fill = sum(line.bbox.width for line in block.lines) / (
            len(block.lines) * column_width
        )
        if mean_fill < MIN_LINE_FILL:
            return False
    return True
