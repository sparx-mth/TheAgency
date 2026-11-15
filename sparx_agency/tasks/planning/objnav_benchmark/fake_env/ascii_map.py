"""The text a fake building is drawn in: one character per cell, the north row first.

A **test rig, not a benchmark.** A building drawn as text is one a reader can
check by eye, so every fake map in the tests and the smoke run is drawn so.
What the text must never do is parse into a building nobody drew:

* **A map that reads upside down.** ASCII row 0 is the north (maximum ``y``)
  row, so the picture reads like a map; the masks come back with row 0 at
  minimum ``y``, the repo's grid convention. A missed flip mirrors every goal
  north-south, and nothing downstream notices.
* **A map that parses into something else.** Ragged rows, a character that is
  neither wall, floor nor in the legend, and a legend that redefines the wall
  or the floor are refused with a message that says how to fix them -- never
  padded, guessed or skipped.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import collections.abc
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavError

#: The ASCII map character of a wall cell.
WALL_CHAR = "#"
#: The ASCII map character of a free floor cell.
FREE_CHAR = "."


def parse_ascii_map(rows: Sequence[str], legend: Mapping[str, str]
                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The wall mask and the object masks of a map drawn as text.

    ``#`` is a wall, ``.`` free floor, and a legend character a cell of its
    category. Several characters may name one category; only categories that
    appear on the map get a mask, in the legend's order.

    Args:
        rows: The map, ASCII row 0 the north (maximum ``y``) row.
        legend: One map character to a category.

    Returns:
        ``(walls, objects)``: a ``(rows, cols)`` boolean wall mask and each
        category's boolean mask of the same shape, stored with row 0 the
        *south* row (the repo's grid convention).

    Raises:
        ObjNavError: On rows that are not non-empty strings of one length, a
            malformed legend, or a character that is neither ``#``, ``.`` nor
            a legend character.
    """
    lines = _ascii_rows(rows)
    symbols = _ascii_legend(legend)
    height, width = len(lines), len(lines[0])
    walls = np.zeros((height, width), dtype=bool)
    cells: Dict[str, List[Tuple[int, int]]] = {}
    for ascii_row, line in enumerate(lines):
        row = height - 1 - ascii_row  # ASCII row 0 is north: maximum y
        for col, char in enumerate(line):
            if char == WALL_CHAR:
                walls[row, col] = True
            elif char in symbols:
                cells.setdefault(symbols[char], []).append((row, col))
            elif char != FREE_CHAR:
                raise ObjNavError(
                    "map row %d column %d holds %r, which is neither '%s' "
                    "(wall), '%s' (free) nor a legend character (%s); add "
                    "it to the legend" % (ascii_row, col, char, WALL_CHAR,
                                          FREE_CHAR, sorted(symbols)))
    objects = {}
    for category in dict.fromkeys(symbols.values()):
        if category in cells:
            mask = np.zeros((height, width), dtype=bool)
            mask[tuple(np.array(cells[category]).T)] = True
            objects[category] = mask
    return walls, objects


def _ascii_rows(rows) -> List[str]:
    """The map's rows once they are known to be equal-length non-empty strings."""
    if isinstance(rows, (str, bytes)) or not isinstance(
            rows, collections.abc.Sequence):
        raise ObjNavError(
            "the map must be a sequence of row strings, north row first, got "
            "%r" % (rows,))
    lines = list(rows)
    if not lines or not all(isinstance(line, str) and line for line in lines):
        raise ObjNavError("the map needs at least one non-empty row string, "
                          "got %r" % (lines,))
    widths = sorted({len(line) for line in lines})
    if len(widths) != 1:
        raise ObjNavError(
            "every map row must have the same length, got lengths %r (row "
            "lengths: %r); pad the short rows with '%s'"
            % (widths, [len(line) for line in lines], WALL_CHAR))
    return lines


def _ascii_legend(legend) -> Dict[str, str]:
    """The legend once every key is one object character and every value a category."""
    if not isinstance(legend, collections.abc.Mapping):
        raise ObjNavError("the legend must map one map character to a "
                          "category, got %r" % (legend,))
    symbols = {}
    for char, category in legend.items():
        if (not isinstance(char, str) or len(char) != 1 or char.isspace()
                or char in (WALL_CHAR, FREE_CHAR)):
            raise ObjNavError(
                "legend key %r must be one printable character other than "
                "'%s' (wall) and '%s' (free)" % (char, WALL_CHAR, FREE_CHAR))
        if not isinstance(category, str) or not category.strip():
            raise ObjNavError("legend entry %r must name a category, got %r"
                              % (char, category))
        symbols[char] = category
    return symbols
