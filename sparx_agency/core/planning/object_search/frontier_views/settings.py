"""What counts as a view of a frontier node: how far, how much of it, and what a look AT it is worth beside a look THROUGH it.

One frozen record for the whole selector (``sight.py`` reads the geometry,
``selection.py`` the choice), so a replay that varies one number varies it
everywhere. Every default is a starting value, set from the first offline
replay (``tasks/planning/objnav_benchmark_runtime/frontier_view_replay.py``)
and meant to be moved by the next.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class ViewSettings:
    """The selector's geometry and thresholds. Distances in metres, shares in ``[0, 1]``.

    Attributes:
        max_range_m: Farthest ground distance from a frame's camera at which
            a cell of the node counts as shown by that frame. The colour
            image has no range limit, but a doorway eight metres away is a
            dozen pixels wide.
        good_range_m: Up to this distance a view counts in full; beyond it
            by apparent size, ``good_range_m / distance`` (the literature's
            three metres for telling an object from a colour frame).
        band_m: Heights above the floor, ``(low, high)``, of what a picture
            must show at a cell for the cell to be in it: the body of
            whatever stands at the boundary, floor to door handle. A level
            camera 0.88 m up shows all of it from a metre and a half away;
            one tilted 30 degrees down shows the lower two thirds; one
            tilted 60 degrees shows the floor at its feet and nothing of a
            boundary two metres off.
        min_band_visible: The share of ``band_m`` that must be inside the
            frame's rows for the cell to count as in the picture; the view's
            worth is scaled by the share.
        border_fraction: Margin from the left and right image borders, as a
            share of the width: a cell on the very edge of a picture is
            shown without what stands beside it.
        unknown_slack_m: DEEP unknown -- unknown that touches no known floor
            -- a sight line may cross between the camera and the cell and
            still count as clear. (The fringe of the map and the floor
            inside the camera's own blind radius are passed over whatever
            this says: ``sight.py``.) A cell and a half: the floor is
            sampled sparsely at range and leaves holes that hide nothing.
        corner_slack_cells: How many cells to either side of the direct
            sight line a parallel line may run and still count as the line
            to the cell. One: the edge of what a camera saw past a door jamb
            is that camera's own line tangent to the jamb, and on a grid a
            tangent clips the corner cell -- without the slack every cell of
            such a boundary reads as hidden from the very frame that mapped
            it (the first replay: 63 % of the nodes scored without it, 81 %
            with it). 0 is the direct line alone, which is what a measure of
            what a frame COULD show should use.
        through_run_m: A view is THROUGH a cell when the sight line,
            continued past the cell, runs over this much unknown that the
            search has not settled: the picture shows what lies beyond the
            boundary there, not only the boundary.
        at_weight: What a cell seen but not seen through is worth, beside 1
            for one seen through. 0 is the strict rule -- only a frame that
            looks past the boundary is a view of it -- which the first
            replay showed leaves most nodes without any frame: within the
            depth camera's range the map's own boundary IS the edge of what
            every frame saw, so no frame looks past it, and the picture of
            the doorway is all there is.
        storey_tolerance_m: A frame is of the node's storey when the floor
            under it was within this height of the floor under the agent
            now.
        max_frames: Frames kept per node at most.
        min_gain_ratio: A further frame is kept only when it raises the
            node's score by at least this share of what the frames already
            chosen reach ...
        min_gain: ... and by at least this much outright.
        min_seen_fraction: The first frame must show at least this share of
            the node's cells ...
        min_seen_cells: ... and at least this many of them ...
        min_score: ... and reach this score, or the node has no usable view
            yet.

    Raises:
        ValueError: On a value outside its range.
    """

    max_range_m: float = 8.0
    good_range_m: float = 3.0
    band_m: Tuple[float, float] = (0.0, 1.5)
    min_band_visible: float = 0.25
    border_fraction: float = 0.05
    unknown_slack_m: float = 0.15
    corner_slack_cells: int = 1
    through_run_m: float = 0.3
    at_weight: float = 0.5
    storey_tolerance_m: float = 0.6
    max_frames: int = 3
    min_gain_ratio: float = 0.2
    min_gain: float = 0.02
    min_seen_fraction: float = 0.15
    min_seen_cells: int = 3
    min_score: float = 0.08

    def __post_init__(self) -> None:
        for name in ("max_range_m", "good_range_m", "through_run_m", "storey_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be positive and finite, got %r" % (name, value))
        if self.good_range_m > self.max_range_m:
            raise ValueError("good_range_m (%r) cannot exceed max_range_m (%r)" % (self.good_range_m, self.max_range_m))
        if isinstance(self.unknown_slack_m, bool) or not math.isfinite(self.unknown_slack_m) or self.unknown_slack_m < 0.0:
            raise ValueError("unknown_slack_m must be finite and non-negative, got %r" % (self.unknown_slack_m,))
        band = self.band_m
        if (not isinstance(band, tuple) or len(band) != 2
                or any(isinstance(v, bool) or not math.isfinite(v) for v in band) or not 0.0 <= band[0] < band[1]):
            raise ValueError("band_m must be (low, high) with 0 <= low < high, got %r" % (band,))
        for name in ("min_band_visible", "at_weight", "min_gain_ratio", "min_gain", "min_seen_fraction", "min_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("%s must lie in [0, 1], got %r" % (name, value))
        if self.min_band_visible <= 0.0:
            raise ValueError("min_band_visible must be positive: a cell none of whose band is in the frame is not in it")
        if isinstance(self.border_fraction, bool) or not math.isfinite(self.border_fraction) or not 0.0 <= self.border_fraction < 0.5:
            raise ValueError("border_fraction must lie in [0, 0.5), got %r" % (self.border_fraction,))
        for name in ("max_frames", "min_seen_cells"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer, got %r" % (name, value))
        if type(self.corner_slack_cells) is not int or not 0 <= self.corner_slack_cells <= 3:
            raise ValueError("corner_slack_cells must be an integer in [0, 3] (more gets round a wall, not a corner), got %r"
                             % (self.corner_slack_cells,))
