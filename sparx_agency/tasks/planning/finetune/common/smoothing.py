"""Collision-aware smoothing of a corrected trajectory (numpy, ROS-free).

The per-waypoint PF/ESDF corrector pushes each point off the walls *independently*,
so a strongly-pushed waypoint next to an un-pushed one leaves a kink -- or, when two
neighbours are pushed opposite ways, a zigzag. That is not something NavDP (whose
output is a smooth cumsum of small deltas) would ever emit, and it makes a poor
behaviour-cloning label.

This applies a **count-preserving** smoothing pass over the corrected waypoints:
each interior point is relaxed toward the midpoint of its neighbours (a Gauss-Seidel
Laplacian), so kinks and reversals flatten out, **but only when the relaxed position
keeps the path clear** of the (inflated) obstacles -- so smoothing never undoes the
wall-avoidance the corrector just bought. Endpoints (robot start, pinned goal) never
move, and the waypoint count is unchanged so the label horizon is preserved.

It reuses the repo's tested primitives -- ``smooth_zigzags_2d`` (the simplifier's
Gauss-Seidel relaxation) and ``InflatedGridCollisionChecker`` (the corrector's own
obstacle test) -- rather than reimplementing either.
"""
from __future__ import annotations

from math import hypot, radians
from typing import Tuple

from sparx_agency.core.common.types import Path2D
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.path_simplification.simplifier_2d import smooth_zigzags_2d
from sparx_agency.core.planning.safety.path_correction.grid_collision import (
    InflatedGridCollisionChecker,
)


def smooth_path(
    path: Path2D,
    occupancy: OccupancyGrid2D,
    clearance_m: float = 0.25,
    strength: float = 0.5,
    passes: int = 10,
    angle_deg: float = 5.0,
) -> Tuple[Path2D, int]:
    """Relax kinks/zigzags out of ``path`` while keeping it clear of obstacles.

    Args:
        path: The corrected ``Path2D`` (body FLU) to smooth.
        occupancy: The single-frame occupancy the correction used (defines the
            obstacles the smoothed path must still clear).
        clearance_m: Obstacle inflation for the clear test -- the hard margin the
            smoother must not cross (the smoother may relax a point down to this
            distance from a wall, but never nearer). Keep it at/below the
            corrector's target clearance.
        strength: Fraction (0..1) of the way each moved point travels toward its
            neighbour midpoint per pass.
        passes: Gauss-Seidel sweeps; more -> smoother (converges toward the
            straightest clear path).
        angle_deg: Only vertices whose heading change exceeds this are relaxed;
            small values smooth gentle wiggles too, large values touch only sharp
            kinks.

    Returns:
        ``(smoothed_path, num_smoothed)`` -- same waypoint count as ``path``;
        ``num_smoothed`` is how many interior points actually moved.
    """
    pts = path.points
    if len(pts) <= 2 or strength <= 0.0 or passes <= 0:
        return path, 0

    clear_fn = InflatedGridCollisionChecker(occupancy, clearance_m).segment_clear
    relaxed = smooth_zigzags_2d(pts, radians(angle_deg), strength, passes, clear_fn)
    moved = sum(1 for a, b in zip(pts, relaxed) if hypot(b.x - a.x, b.y - a.y) > 1e-3)
    smoothed = Path2D(points=tuple(relaxed), frame_id=path.frame_id,
                      metadata=dict(path.metadata))
    return smoothed, moved
