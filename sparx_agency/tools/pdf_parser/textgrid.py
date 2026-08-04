"""Putting positioned words back onto a fixed-width grid.

Some things on a page mean something by where they sit, not by what they say.
Pseudocode indentation *is* the control flow; an aligned column of numbers *is*
the comparison being made. Joining a line's words with single spaces throws that
away, so anything of that kind is rebuilt here instead: estimate the width of one
character, then write each word into a character buffer at the position its box
says it starts at.

This is the trick ``pdftotext -layout`` uses internally. It is reimplemented
rather than reused because ``-layout`` can only be asked for a whole page, and
what is wanted is one region of one page with everything else left out.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from sparx_agency.tools.pdf_parser.layout import Line

FALLBACK_CHAR_WIDTH_PT = 5.0
"""Character width assumed when a region has too little text to measure one.

Roughly a 10 pt monospace character. Only reached on regions of a few words,
where the indentation being reconstructed is trivial anyway.
"""

MAX_INDENT_CHARS = 200
"""Ceiling on reconstructed indentation, so one stray box cannot emit a huge line."""


def estimate_char_width(lines: Sequence[Line]) -> float:
    """Estimate the mean width of one character across some lines, in points.

    Measured rather than assumed because papers set algorithm blocks and table
    bodies smaller than body text, and an indentation computed with the wrong
    character width drifts further to the right on every level.

    Args:
        lines: The lines to measure.

    Returns:
        Mean points per character, or :data:`FALLBACK_CHAR_WIDTH_PT` if there is
        nothing to measure.
    """
    total_width = 0.0
    total_chars = 0
    for line in lines:
        for word in line.words:
            if word.text:
                total_width += word.bbox.width
                total_chars += len(word.text)
    if total_chars == 0 or total_width <= 0.0:
        return FALLBACK_CHAR_WIDTH_PT
    return total_width / total_chars


def render_line(line: Line, left_pt: float, char_width_pt: float) -> str:
    """Rebuild one line as text, with every word at its horizontal position.

    Args:
        line: The line to render.
        left_pt: The x coordinate that maps to column zero.
        char_width_pt: Points per character.

    Returns:
        The line as a string, right-stripped. Words that would collide because
        the estimate is slightly off are separated by a single space rather than
        overwriting each other.
    """
    buffer: List[str] = []
    for word in sorted(line.words, key=lambda item: item.bbox.x_min):
        column = int(round((word.bbox.x_min - left_pt) / max(char_width_pt, 0.1)))
        column = max(0, min(column, MAX_INDENT_CHARS))
        if buffer and column <= len(buffer):
            column = len(buffer) + 1
        buffer.extend(" " * (column - len(buffer)))
        buffer.extend(word.text)
    return "".join(buffer).rstrip()


def render_lines(
    lines: Sequence[Line],
    left_pt: Optional[float] = None,
    char_width_pt: Optional[float] = None,
) -> str:
    """Rebuild a group of lines as fixed-width text.

    Args:
        lines: The lines, which are sorted top to bottom before rendering.
        left_pt: The x coordinate mapping to column zero; defaults to the
            leftmost word, so the result has no common leading indent.
        char_width_pt: Points per character; measured from the lines if omitted.

    Returns:
        The lines as one newline-joined string, empty if there were no lines.
    """
    ordered = sorted(lines, key=lambda line: (line.bbox.y_min, line.bbox.x_min))
    if not ordered:
        return ""

    origin = left_pt if left_pt is not None else min(line.bbox.x_min for line in ordered)
    width = char_width_pt if char_width_pt is not None else estimate_char_width(ordered)
    return "\n".join(render_line(line, origin, width) for line in ordered)
