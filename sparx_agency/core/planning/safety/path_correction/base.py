"""Path-correction interface: reshape a planned path against a live grid.

A *path corrector* takes a planned :class:`Path2D` (from A*, NavDP, RRT*, ... --
it is deliberately planner-agnostic) plus the current :class:`OccupancyGrid2D`
and returns a corrected :class:`Path2D` that is **never less safe** than the
input. Concrete strategies -- repulsive potential field, ESDF ridge-following,
... -- each live in their own module and subclass :class:`PathCorrector`, so the
ROS node that owns the "receive a path -> correct -> republish" wiring can swap
strategies by name without changing (single responsibility).

Pure / ROS-free; Python 3.8 compatible (the FALCON Noetic adapter imports core
under 3.8): no PEP 604 unions, no ``match``/``case``, no ``slots=``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sparx_agency.core.common.types import Path2D
from sparx_agency.core.planning.environment import OccupancyGrid2D


@dataclass
class PathCorrectionResult:
    """Outcome of one :meth:`PathCorrector.correct` call.

    Attributes:
        path: The corrected path (or the input unchanged when nothing moved).
        num_points: Number of waypoints in ``path``.
        num_moved: How many waypoints actually shifted (> 1 mm) from the input.
    """

    path: Path2D
    num_points: int = 0
    num_moved: int = 0


class PathCorrector(ABC):
    """Reshape a planned path against a live occupancy grid.

    Implementations move waypoints to a safer / more central position (e.g. off
    walls toward a corridor centre). The contract is intentionally minimal so any
    strategy can satisfy it: given a path and the current grid, return a
    corrected path. Implementations MUST keep the result never less safe than the
    input (e.g. via a per-waypoint collision re-check) and SHOULD leave the start
    waypoint fixed (it is the robot's current pose).
    """

    #: Stable identifier used by the corrector factory / registry.
    name: str = "base"

    @abstractmethod
    def correct(self, path: Path2D, grid: OccupancyGrid2D) -> PathCorrectionResult:
        """Return a corrected copy of ``path`` against ``grid``.

        Args:
            path: The planned path to reshape (>= 2 waypoints).
            grid: The live occupancy grid to correct against.

        Returns:
            A :class:`PathCorrectionResult` wrapping the corrected path.
        """
        raise NotImplementedError
