"""Boxes on a PDF page, and the one unit conversion everything else depends on.

Poppler reports every bounding box in PDF *points* — 1/72 inch — with the origin
at the top-left of the page and ``y`` growing downward. ``pdftoppm`` crops in
*pixels* at whatever resolution it was asked for, with the same origin and the
same direction. So the only conversion in this package is ``points x dpi / 72``,
and it lives here so that no caller has to remember which of the two units the
number in its hand is in.

Boxes are used for two things: deciding whether two pieces of a page belong to
each other (a caption and the artwork above it, a word and a table column), and
handing a rectangle to the renderer. Both want cheap overlap arithmetic rather
than a geometry library, so that is all this module is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

POINTS_PER_INCH = 72.0
"""PDF user-space units per inch. Fixed by the PDF specification, never varies."""


@dataclass(frozen=True)
class BBox:
    """An axis-aligned rectangle in PDF points, origin top-left, ``y`` downward.

    Attributes:
        x_min: Left edge, points from the left of the page.
        y_min: Top edge, points from the top of the page.
        x_max: Right edge.
        y_max: Bottom edge.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError(
                "inverted BBox: ({}, {}) to ({}, {})".format(
                    self.x_min, self.y_min, self.x_max, self.y_max
                )
            )

    @property
    def width(self) -> float:
        """Width in points."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Height in points."""
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        """Horizontal midpoint in points."""
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        """Vertical midpoint in points."""
        return 0.5 * (self.y_min + self.y_max)

    def union(self, other: "BBox") -> "BBox":
        """Return the smallest box containing both this box and ``other``."""
        return BBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max),
        )

    def x_overlap(self, other: "BBox") -> float:
        """Return the width of the horizontal overlap with ``other``, in points."""
        return max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))

    def x_overlap_ratio(self, other: "BBox") -> float:
        """Return the horizontal overlap as a fraction of the *narrower* box.

        Using the narrower box as the denominator is what makes this work across
        a two-column page: a one-column figure sitting under a full-width
        caption still scores 1.0, which is the answer we want when asking "is
        this block in the same column as that caption".
        """
        narrower = min(self.width, other.width)
        if narrower <= 0.0:
            return 0.0
        return self.x_overlap(other) / narrower

    def vertical_gap(self, other: "BBox") -> float:
        """Return the vertical distance to ``other``, or 0.0 if they overlap."""
        if self.y_max <= other.y_min:
            return other.y_min - self.y_max
        if other.y_max <= self.y_min:
            return self.y_min - other.y_max
        return 0.0

    def padded(self, margin: float, page: Optional["BBox"] = None) -> "BBox":
        """Grow the box by ``margin`` points on every side, clipped to ``page``.

        A crop taken exactly on the measured extent of a figure shaves the
        outermost stroke of the drawing, so every crop this package renders is
        padded a little.
        """
        grown = BBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin,
        )
        if page is None:
            return grown
        return BBox(
            max(grown.x_min, page.x_min),
            max(grown.y_min, page.y_min),
            min(grown.x_max, page.x_max),
            min(grown.y_max, page.y_max),
        )

    def to_pixels(self, dpi: float) -> Tuple[int, int, int, int]:
        """Convert to ``(x, y, width, height)`` in pixels for ``pdftoppm``.

        Args:
            dpi: The resolution the page will be rendered at.

        Returns:
            Integer pixel offsets and extents, rounded outward so that nothing
            inside the box is cut off. Width and height are at least 1.
        """
        scale = dpi / POINTS_PER_INCH
        x = int(self.x_min * scale)
        y = int(self.y_min * scale)
        width = max(1, int(round(self.x_max * scale)) - x)
        height = max(1, int(round(self.y_max * scale)) - y)
        return x, y, width, height


def union_all(boxes: Iterable[BBox]) -> Optional[BBox]:
    """Return the smallest box containing every box given, or None if empty."""
    merged: Optional[BBox] = None
    for box in boxes:
        merged = box if merged is None else merged.union(box)
    return merged
