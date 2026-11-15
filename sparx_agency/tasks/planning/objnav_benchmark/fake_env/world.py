"""A 2.5-D building on a grid: the fake environment's world -- its walls, floor and objects, and where each cell lies.

A **test rig, not a benchmark.** Each cell is a wall, free floor, or an object
of a named category, extruded to 2.5-D for a depth camera: walls to full
height, objects to a table's height, a floor and a ceiling. It exists so the
whole ObjectNav loop -- environment, agent, converter, runner, scoring -- runs
in seconds without a simulator, and so a privileged oracle can show the loop
*can* score SR 1.0.

:class:`GridWorld` holds the building and nothing derived from it. What an
agent can do in it is worked out once, one module per question, and the
environment and the oracle both read the same answer -- because every quiet
failure of a fake is a test that passes for the wrong reason:

* :mod:`.ascii_map` -- the text a building is drawn in, north row first, so a
  map never reads upside down.
* :mod:`.raster` -- where an agent of a radius can stand, where it has found
  an object, and the voxels the depth camera sees, each measured to a cell's
  square rather than its centre.
* :mod:`.geodesics` -- how far a goal is, by moves that never cut a corner.

What the building guards itself: a cell holds one thing, every mask it hands
out is read-only, and a mask handed back to it must be of its shape.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import collections.abc
import math
import numbers
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_length,
    is_real,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.ascii_map import (
    parse_ascii_map,
)

#: A grid cell, ``(row, col)``; row 0 is the minimum-``y`` row.
Cell = Tuple[int, int]


def _positive(name: str, value) -> float:
    if not (is_length(value) and value > 0):
        raise ObjNavError("%s must be a positive finite number, got %r"
                          % (name, value))
    return float(value)


def _bool_grid(name: str, value) -> np.ndarray:
    """``value`` once it is known to be a 2-D boolean array."""
    if (not isinstance(value, np.ndarray) or value.dtype != np.bool_
            or value.ndim != 2):
        got = ("shape %r dtype %s" % (value.shape, value.dtype)
               if isinstance(value, np.ndarray) else type(value).__name__)
        raise ObjNavError("%s must be a 2-D boolean numpy array, got %s"
                          % (name, got))
    return value


def _read_only(array: np.ndarray) -> np.ndarray:
    """A private copy of ``array`` that cannot be written through."""
    copy = np.array(array)
    copy.setflags(write=False)
    return copy


class GridWorld:
    """A building on a square grid: walls, floor, and objects of named categories.

    Immutable: every mask it hands out is read-only, and every mask it
    computes is a new array.

    Args:
        walls: ``(rows, cols)`` boolean wall mask; row 0 is minimum ``y``.
        objects: Category to its boolean mask, same shape. Each object cell
            is non-traversable and holds one thing only. Iteration order is
            :attr:`categories`' order.
        resolution_m: Cell edge, metres.
        origin_xy: World ``(x, y)`` of cell ``(0, 0)``'s lower-left corner.
        wall_height_m: How tall walls are for the depth camera, metres.
        object_height_m: How tall objects are, metres; at most the walls'.

    Raises:
        ObjNavError: On a mask that is not a non-empty 2-D boolean array of
            one shape, a blank category, an object with no cell, a cell held
            by two things, or a size or height that is not positive and
            finite.
    """

    def __init__(self, walls: np.ndarray, objects: Mapping[str, np.ndarray],
                 resolution_m: float = 0.25,
                 origin_xy: Tuple[float, float] = (0.0, 0.0),
                 wall_height_m: float = 2.5,
                 object_height_m: float = 0.8) -> None:
        walls = _bool_grid("walls", walls)
        if walls.size == 0:
            raise ObjNavError("walls must have at least one cell, got shape "
                              "%r" % (walls.shape,))
        if not isinstance(objects, collections.abc.Mapping):
            raise ObjNavError("objects must map a category to its mask, got "
                              "%r" % (objects,))
        self._resolution = _positive("resolution_m", resolution_m)
        self._origin = self._check_origin(origin_xy)
        self._wall_height = _positive("wall_height_m", wall_height_m)
        self._object_height = _positive("object_height_m", object_height_m)
        if self._object_height > self._wall_height:
            raise ObjNavError(
                "object_height_m=%r exceeds wall_height_m=%r; the ceiling "
                "would cut through the objects" % (object_height_m,
                                                   wall_height_m))
        taken = walls.copy()
        masks: Dict[str, np.ndarray] = {}
        for category, mask in objects.items():
            if not isinstance(category, str) or not category.strip():
                raise ObjNavError("an object category must be a non-blank "
                                  "string, got %r" % (category,))
            mask = _bool_grid("the mask of %r" % category, mask)
            if mask.shape != walls.shape:
                raise ObjNavError("the mask of %r has shape %r, the walls %r"
                                  % (category, mask.shape, walls.shape))
            if not mask.any():
                raise ObjNavError("category %r has no cell; leave it out of "
                                  "the world instead" % (category,))
            clash = np.argwhere(mask & taken)
            if len(clash):
                raise ObjNavError(
                    "category %r overlaps a wall or another object at cells "
                    "%r; a cell holds one thing"
                    % (category, [tuple(int(i) for i in c) for c in clash[:5]]))
            taken |= mask
            masks[category] = _read_only(mask)
        self._walls = _read_only(walls)
        self._objects = masks
        self._blocked = _read_only(taken)

    @staticmethod
    def _check_origin(origin_xy) -> Tuple[float, float]:
        try:
            x, y = origin_xy
        except (TypeError, ValueError):
            raise ObjNavError("origin_xy must be an (x, y) pair, got %r"
                              % (origin_xy,))
        if not all(is_real(v) and math.isfinite(v) for v in (x, y)):
            raise ObjNavError("origin_xy must be finite, got %r"
                              % (origin_xy,))
        return float(x), float(y)

    @classmethod
    def from_ascii(cls, rows: Sequence[str], legend: Mapping[str, str],
                   resolution_m: float = 0.25,
                   origin_xy: Tuple[float, float] = (0.0, 0.0),
                   wall_height_m: float = 2.5,
                   object_height_m: float = 0.8) -> GridWorld:
        """A world drawn as text, north row first, so it reads like a map.

        ``#`` is a wall, ``.`` free floor, and a legend character a
        non-traversable cell of its category; :func:`.ascii_map.parse_ascii_map`
        reads it.

        Args:
            rows: The map, ASCII row 0 the north (maximum ``y``) row.
            legend: One map character to a category.
            resolution_m: Cell edge, metres.
            origin_xy: World ``(x, y)`` of the south-west corner.
            wall_height_m: Wall height, metres.
            object_height_m: Object height, metres.

        Returns:
            The world, stored with row 0 the *south* row (the repo's grid
            convention).

        Raises:
            ObjNavError: On a map the parse refuses, or anything the
                constructor refuses.
        """
        walls, objects = parse_ascii_map(rows, legend)
        return cls(walls, objects, resolution_m, origin_xy, wall_height_m,
                   object_height_m)

    # -- what the world is ----------------------------------------------------

    @property
    def resolution_m(self) -> float:
        """Cell edge, metres."""
        return self._resolution

    @property
    def origin_xy(self) -> Tuple[float, float]:
        """World ``(x, y)`` of cell ``(0, 0)``'s lower-left corner."""
        return self._origin

    @property
    def shape(self) -> Tuple[int, int]:
        """``(rows, cols)``."""
        rows, cols = self._walls.shape
        return int(rows), int(cols)

    @property
    def categories(self) -> Tuple[str, ...]:
        """Every object category in the world, in construction order."""
        return tuple(self._objects)

    @property
    def wall_height_m(self) -> float:
        """Wall height, metres."""
        return self._wall_height

    @property
    def object_height_m(self) -> float:
        """Object height, metres."""
        return self._object_height

    @property
    def wall_mask(self) -> np.ndarray:
        """Read-only boolean mask of the walls."""
        return self._walls

    @property
    def blocked_mask(self) -> np.ndarray:
        """Read-only boolean mask of everything non-traversable: walls and objects."""
        return self._blocked

    def object_mask(self, category: str) -> np.ndarray:
        """Read-only boolean mask of the cells of ``category``.

        Raises:
            ObjNavError: If the world has no object of ``category``.
        """
        mask = self._objects.get(category) if isinstance(category, str) else None
        if mask is None:
            raise ObjNavError("the world has no %r; it has %s"
                              % (category, ", ".join(map(repr, self._objects))))
        return mask

    # -- cells and points -----------------------------------------------------

    def cell_of(self, x: float, y: float) -> Cell:
        """The cell a world point lies in (floor), possibly out of bounds.

        Raises:
            ObjNavError: On a non-finite coordinate.
        """
        for name, value in (("x", x), ("y", y)):
            if not (is_real(value) and math.isfinite(value)):
                raise ObjNavError("cell_of: %s must be finite, got %r"
                                  % (name, value))
        ox, oy = self._origin
        return (int(math.floor((y - oy) / self._resolution)),
                int(math.floor((x - ox) / self._resolution)))

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """World ``(x, y)`` of a cell's centre.

        Raises:
            ObjNavError: If ``row`` or ``col`` is not an integer.
        """
        if not all(isinstance(v, numbers.Integral) and not isinstance(v, bool)
                   for v in (row, col)):
            raise ObjNavError("cell_center needs integer (row, col), got %r"
                              % ((row, col),))
        ox, oy = self._origin
        return (ox + (int(col) + 0.5) * self._resolution,
                oy + (int(row) + 0.5) * self._resolution)

    def in_bounds(self, row: int, col: int) -> bool:
        """Whether ``(row, col)`` is a cell of the grid."""
        rows, cols = self.shape
        return 0 <= row < rows and 0 <= col < cols

    def check_mask(self, name: str, mask) -> np.ndarray:
        """``mask``, once it is known to be a boolean mask of this world's shape.

        Every function that takes a mask of the world starts here, so a mask
        of another world, or of numbers, is refused by name instead of
        indexing the wrong cells.

        Args:
            name: What the mask is, for the message.
            mask: The candidate.

        Returns:
            ``mask`` itself.

        Raises:
            ObjNavError: If ``mask`` is not a 2-D boolean numpy array of
                :attr:`shape`.
        """
        mask = _bool_grid(name, mask)
        if mask.shape != self._walls.shape:
            raise ObjNavError("%s has shape %r but the world is %r"
                              % (name, mask.shape, self.shape))
        return mask

    def __repr__(self) -> str:
        return ("GridWorld(shape=%r, resolution_m=%r, categories=%r)"
                % (self.shape, self._resolution, self.categories))
