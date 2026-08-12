"""The exploration area, as five numbers a person can check against a floor plan.

FALCON wants eighteen: a minimum and a maximum on each axis for three nested
boxes. Those three are not independent — ``map`` is ``box`` plus a margin,
``vbox`` is ``box`` plus a smaller one — so writing them out by hand is eighteen
chances to break a containment rule that only fails once the container is up.

This module holds the five numbers that actually vary and derives the rest.

The three boxes, and what each one is for:

``box``
    What FALCON explores and plans inside. Its edges must fall on walls. An edge
    in open space is a trap: frontiers appear along the cut, half of every
    viewpoint ring lands outside and is discarded, and the planner cycles.
``map``
    The voxel grid that gets allocated, in full, on the first tick. Bigger than
    ``box`` because the sensor sees past the region being explored, and every
    cubic metre of it is paid for whether or not the aircraft ever goes there.
``vbox``
    What the map recorder draws. A thin horizontal slab at cruise height, because
    the recorder wants one cut through the map and every extra layer is another
    full pass over the grid inside the node servicing the depth callbacks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# How far ``vbox`` reaches past ``box`` horizontally. Half a metre, so the drawn
# map shows the wall the exploration box stops at rather than cutting it off.
VBOX_MARGIN_M = 0.5

# Thickness of the drawn slab, centred on cruise height.
VBOX_SLAB_M = 0.2


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in world coordinates, metres."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def size(self) -> Tuple[float, float, float]:
        """Extent on each axis, in metres."""
        return (self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z)

    @property
    def volume(self) -> float:
        """Volume in cubic metres."""
        x, y, z = self.size
        return x * y * z

    def contains(self, inner: "Box") -> bool:
        """Whether this box encloses ``inner``, touching faces allowed."""
        return (
            self.min_x <= inner.min_x
            and self.min_y <= inner.min_y
            and self.min_z <= inner.min_z
            and self.max_x >= inner.max_x
            and self.max_y >= inner.max_y
            and self.max_z >= inner.max_z
        )

    def grid_shape(self, resolution: float) -> Tuple[int, int, int]:
        """Voxel count on each axis at a given resolution.

        Mirrors FALCON's own ``map_size_idx_(i) = ceil(map_size_(i) / resolution)``
        in ``map_server.cpp``, including the rounding, so the number this reports
        is the number that gets allocated.

        Args:
            resolution: Voxel edge length in metres.

        Returns:
            Voxels on the x, y and z axes.

        Raises:
            ValueError: If ``resolution`` is not positive.
        """
        if resolution <= 0.0:
            raise ValueError("resolution must be positive, got {}".format(resolution))
        return tuple(int(math.ceil(extent / resolution)) for extent in self.size)

    def to_falcon(self, prefix: str) -> dict:
        """Flatten to the ``<prefix>_min_x``-style keys FALCON reads.

        Args:
            prefix: One of ``map``, ``box`` or ``vbox``.

        Returns:
            Six keys, ready to merge into a ``map_size`` block.
        """
        return {
            prefix + "_min_x": self.min_x,
            prefix + "_min_y": self.min_y,
            prefix + "_min_z": self.min_z,
            prefix + "_max_x": self.max_x,
            prefix + "_max_y": self.max_y,
            prefix + "_max_z": self.max_z,
        }


@dataclass(frozen=True)
class ExplorationArea:
    """What to explore, at what resolution — the whole run's map geometry.

    Attributes:
        building: ``(x0, y0, x1, y1)`` footprint to explore, in metres. Already
            inset from the outer walls; this is the exploration box, not the
            building's true outline.
        flight_band: ``(low, high)`` z range the planner may fly in. The airspace
            above the clutter, not the whole room.
        vertical_extent: ``(low, high)`` z range to allocate. Floor to ceiling —
            wider than the flight band, because the camera sees the floor and the
            mapper needs somewhere to put it.
        resolution: Voxel edge in metres, chosen explicitly. See the README for
            why this is not left to FALCON's volume rule.
        margin: How far the allocated grid reaches past the exploration box
            horizontally, in metres. A single number for a symmetric margin,
            ``(low_side, high_side)`` where the two sides differ, or
            ``(low_x, low_y, high_x, high_y)`` — same corner order as
            ``building`` — where the two axes differ too. The warehouse runs
            need the four-number form: their grids are the footprint of a
            surveyed voxel map, which is not centred on the exploration box.
        visualisation: An explicit drawn box, ``(x0, y0, z0, x1, y1, z1)``.
        visualisation_slab_at: Height to centre a thin drawn slab on, for a
            recorder that wants one horizontal cut. Set by the caller from the
            run's cruise altitude, not written in the file.

    With neither ``visualisation`` nor ``visualisation_slab_at``, the drawn box
    is the allocated grid, which is what most environments want.
    """

    building: Tuple[float, float, float, float]
    flight_band: Tuple[float, float]
    vertical_extent: Tuple[float, float]
    resolution: float
    margin: float = 2.0
    visualisation: Optional[Tuple[float, float, float, float, float, float]] = None
    visualisation_slab_at: Optional[float] = None

    @property
    def margins(self) -> Tuple[float, float, float, float]:
        """The horizontal margin as ``(low_x, low_y, high_x, high_y)``."""
        if isinstance(self.margin, (int, float)):
            side = float(self.margin)
            return (side, side, side, side)
        if len(self.margin) == 2:
            low, high = (float(value) for value in self.margin)
            return (low, low, high, high)
        return tuple(float(value) for value in self.margin)

    @property
    def box(self) -> Box:
        """The exploration box: the footprint, over the flight band."""
        x0, y0, x1, y1 = self.building
        return Box(x0, y0, self.flight_band[0], x1, y1, self.flight_band[1])

    @property
    def map(self) -> Box:
        """The allocated voxel grid: the box plus margin, over the full height."""
        box = self.box
        low_x, low_y, high_x, high_y = self.margins
        return Box(
            box.min_x - low_x,
            box.min_y - low_y,
            self.vertical_extent[0],
            box.max_x + high_x,
            box.max_y + high_y,
            self.vertical_extent[1],
        )

    @property
    def vbox(self) -> Box:
        """What the recorder draws.

        Three cases, in order: an explicit box if the file gives one, a thin slab
        at cruise height if a caller asked for one, and otherwise the allocated
        grid — every layer the recorder draws is another full pass over the grid
        inside the node servicing depth callbacks, so a slab is much cheaper
        where a horizontal cut is all that is wanted.
        """
        if self.visualisation is not None:
            return Box(*self.visualisation)

        if self.visualisation_slab_at is not None:
            box = self.box
            half = VBOX_SLAB_M / 2.0
            return Box(
                box.min_x - VBOX_MARGIN_M,
                box.min_y - VBOX_MARGIN_M,
                self.visualisation_slab_at - half,
                box.max_x + VBOX_MARGIN_M,
                box.max_y + VBOX_MARGIN_M,
                self.visualisation_slab_at + half,
            )

        return self.map

    @classmethod
    def from_dict(cls, spec: dict) -> "ExplorationArea":
        """Build from a run file's ``area`` block.

        Args:
            spec: The mapping under ``map_config.area``.

        Returns:
            The parsed area.

        Raises:
            ValueError: If a key is missing, or a sequence is the wrong length.
        """
        required = ("building", "flight_band", "vertical_extent", "resolution")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                "map_config.area is missing {}. It needs building (4 numbers), "
                "flight_band (2), vertical_extent (2) and resolution.".format(
                    ", ".join(missing)
                )
            )

        def sequence(key: str, length: int) -> tuple:
            value = spec[key]
            try:
                values = tuple(float(item) for item in value)
            except (TypeError, ValueError):
                raise ValueError(
                    "map_config.area.{} must be {} numbers, got {!r}".format(
                        key, length, value
                    )
                )
            if len(values) != length:
                raise ValueError(
                    "map_config.area.{} must be {} numbers, got {}".format(
                        key, length, len(values)
                    )
                )
            return values

        margin = spec.get("margin", 2.0)
        if isinstance(margin, (list, tuple)):
            if len(margin) not in (2, 4):
                raise ValueError(
                    "map_config.area.margin must be one number, two for an "
                    "asymmetric grid, or four (low_x, low_y, high_x, high_y) "
                    "where the axes differ too, got {}".format(len(margin))
                )
            margin = tuple(float(value) for value in margin)
        else:
            margin = float(margin)

        return cls(
            building=sequence("building", 4),
            flight_band=sequence("flight_band", 2),
            vertical_extent=sequence("vertical_extent", 2),
            resolution=float(spec["resolution"]),
            margin=margin,
            visualisation=sequence("visualisation", 6)
            if "visualisation" in spec
            else None,
        )
