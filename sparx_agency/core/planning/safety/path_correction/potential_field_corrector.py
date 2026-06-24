"""Repulsive-potential-field path corrector (ROS-free).

Wraps the two reusable core primitives -- ``PotentialFieldLayer`` (builds the
repulsive field ``U_rep`` + distance field ``D_obs`` from an occupancy grid) and
:class:`TrajectorySafetyCorrector` (recentres a Path2D against that field) --
into a single :class:`PathCorrector`: given a planned path and the live
occupancy grid it returns a path recentred off the walls toward corridor
centres. It adds the two map-aware safety steps the algorithm needs in the field:

* **unknown-area damping** -- scale each waypoint's shift by the fraction of
  KNOWN cells around its corrected position, so a push into half-mapped space
  (no opposing wall to balance it) is damped, and fades back to full strength as
  the map fills in;
* **per-waypoint collision clip** -- pull any corrected waypoint back toward its
  input position just far enough to keep both of its adjacent segments clear of
  inflated obstacles, so the corrected path is never less safe than the input.

The field generation matches the BEV frame exactly (cell-centre half-cell origin
shift). All maths is numpy-only. Python 3.8 compatible (the FALCON Noetic adapter
imports core under 3.8): no PEP 604 unions, no ``match``/``case``.

This module owns ONLY the potential-field correction strategy (single
responsibility). A different strategy (e.g. an ESDF ridge follower) is a new
sibling module implementing :class:`PathCorrector`; see :func:`make_path_corrector`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer
from sparx_agency.core.planning.environment import OccupancyGrid2D

from ..potential_field_sampler import PotentialFieldSampler
from ..trajectory_safety_corrector import TrajectorySafetyCorrector
from ..types import TrajectoryCorrectionParams
from .base import PathCorrector, PathCorrectionResult
from .grid_collision import InflatedGridCollisionChecker


@dataclass(frozen=True)
class PotentialFieldCorrectorConfig:
    """Tuning for :class:`PotentialFieldPathCorrector`.

    These code defaults match the historical ``astar_planner`` node's *code*
    defaults (its ``rospy.get_param`` fallbacks), so a bare run is unchanged. The
    nav_stack launch passes the same ``~apf_*`` values it always did (which already
    overrode some of these), so a launched run is unchanged too -- behaviour is
    identical to before either way.

    Field generation (``PotentialFieldLayer``):
        occ_thresh: Probability above which a cell counts as occupied.
        sigma_m: Gaussian repulsion spread (the only spatial field knob).

    Centring (``TrajectorySafetyCorrector`` -- see its params for full detail):
        centering: ``"line_search"`` (medial-axis, scale-free, single-pass) or
            ``"descent"`` (legacy gradient descent).
        center_step_m: Normal-sample spacing for line_search.
        corner_swing: Extra lateral search range per 90 deg of turn (line_search).
        iterations, gain, max_step_m, smoothing_passes: descent-mode knobs.
        max_total_shift_m: Max per-waypoint displacement (line_search: the lateral
            search half-range -- make it >= the corridor half-width).
        min_clearance_m: > 0 also pushes waypoints to a minimum distance-to-wall.
        lateral_only: Project pushes perpendicular to the path (no fore/aft slide).
        pin_last: Keep the final waypoint fixed (it is the hard goal).

    Map-aware safety steps:
        collision_recheck: Per-waypoint clip of the corrected path against
            inflated obstacles (never less safe than the input).
        inflate_radius_m: Obstacle inflation for that clip; match the planner's.
        treat_unknown_as_free: Recentre waypoints over UNKNOWN cells too, pushed
            only by known walls (matches A* planning through unknown as free).
        unknown_damping: Scale each waypoint's shift by local map confidence.
        unknown_radius_m: Confidence-window radius (m) for the damping.
    """

    # field generation
    occ_thresh: float = 0.65
    sigma_m: float = 0.6
    # centring
    centering: str = "line_search"
    center_step_m: float = 0.05
    corner_swing: float = 1.0
    iterations: int = 5
    gain: float = 1.0
    max_step_m: float = 0.4
    max_total_shift_m: float = 2.0
    smoothing_passes: int = 2
    min_clearance_m: float = 0.0
    lateral_only: bool = True
    pin_last: bool = True
    # map-aware safety steps
    collision_recheck: bool = True
    inflate_radius_m: float = 0.4
    treat_unknown_as_free: bool = True
    unknown_damping: bool = True
    unknown_radius_m: float = 0.75


class PotentialFieldPathCorrector(PathCorrector):
    """Recentre a planned path off walls using a repulsive potential field."""

    name = "potential_field"

    def __init__(self, config: Optional[PotentialFieldCorrectorConfig] = None) -> None:
        self.cfg = config or PotentialFieldCorrectorConfig()
        self._layer = PotentialFieldLayer(
            occ_thresh=self.cfg.occ_thresh,
            sigma_m=self.cfg.sigma_m,
        )
        self._corrector = TrajectorySafetyCorrector(TrajectoryCorrectionParams(
            centering=self.cfg.centering,
            center_step_m=self.cfg.center_step_m,
            corner_swing=self.cfg.corner_swing,
            iterations=self.cfg.iterations,
            gain=self.cfg.gain,
            max_step_m=self.cfg.max_step_m,
            max_total_shift_m=self.cfg.max_total_shift_m,
            smoothing_passes=self.cfg.smoothing_passes,
            min_clearance_m=self.cfg.min_clearance_m,
            lateral_only=self.cfg.lateral_only,
            pin_last=self.cfg.pin_last,
        ))

    @property
    def field(self) -> Optional[PotentialFieldSampler]:
        """The repulsive field sampler from the most recent :meth:`correct`.

        Exposed so a node can draw the force ``F_rep = -grad U_rep`` arrows
        (``field.descent(x, y)``) for visualisation. ``None`` before the first
        correction.
        """
        return self._corrector.field

    @property
    def params(self) -> TrajectoryCorrectionParams:
        """The underlying centring parameters (for logging / introspection)."""
        return self._corrector.params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def correct(self, path: Path2D, grid: OccupancyGrid2D) -> PathCorrectionResult:
        """Recentre ``path`` away from the walls in ``grid`` toward corridor centres.

        Builds the repulsive field from ``grid``, recentres the path against it,
        damps the push where it heads into unknown space, then clips any corrected
        waypoint back to keep the path clear of inflated obstacles. The start
        waypoint (and, with ``pin_last``, the goal) are held fixed.

        Raises:
            Any error from the underlying field/correction maths propagates to the
            caller, which should fall back to the un-corrected path.
        """
        self._build_field(grid)
        safe = self._corrector.correct_path(path)
        pts = safe.points
        if self.cfg.unknown_damping:
            # Damp the push where it heads into unknown (unbalanced) BEFORE the
            # collision clip, so both only ever pull the path back toward input.
            pts = self._dampen_unknown(path.points, pts, grid)
        if self.cfg.collision_recheck:
            pts = self._clip_to_clear(path.points, pts, grid)
        moved = sum(1 for r, s in zip(path.points, pts)
                    if math.hypot(s.x - r.x, s.y - r.y) > 1e-3)
        corrected = Path2D(points=tuple(pts), frame_id=path.frame_id,
                           metadata=dict(safe.metadata))
        return PathCorrectionResult(path=corrected, num_points=len(pts), num_moved=moved)

    # ------------------------------------------------------------------
    # Field build
    # ------------------------------------------------------------------
    def _build_field(self, grid: OccupancyGrid2D) -> None:
        """Install the grid's repulsive field into the corrector (same world frame).

        The ``OccupancyGrid2D`` in its native (un-flipped) ROS orientation has
        row<->y, col<->x about ``(origin_x, origin_y)`` -- exactly the convention
        ``PotentialFieldSampler`` expects -- so ``U_rep``, the distance field and
        the path all share one metric frame. No flip.
        """
        raw = grid.grid                                    # int16 (H, W) BEV values
        # BEV ints -> probability grid: occupied -> 1, free -> 0, unknown -> NaN.
        # PotentialFieldLayer treats NaN as FREE, so U_rep and D_obs draw their
        # repulsion ONLY from known walls -- unknown space is open (exactly like A*
        # with unknown_blocked=False).
        p_occ = np.where(raw == grid.values.occupied, 1.0,
                         np.where(raw == grid.values.free, 0.0, np.nan)
                         ).astype(np.float32)
        # known_mask gates which waypoints the corrector may move. Default
        # (treat_unknown_as_free) passes None -> unknown counts as free, so a
        # waypoint over unknown space is still recentred (pushed only by the known
        # walls). Pass the observed mask to restore the freeze-over-unknown.
        known = None if self.cfg.treat_unknown_as_free else (raw != grid.values.unknown)
        u_rep, d_obs = self._layer.compute_from_prob_grid(p_occ, grid.resolution)
        # Sample at cell CENTRES: OccupancyGrid2D -- and the path it produces via
        # grid_to_world -- places cell (gx, gy) at origin+(gx+0.5)*res, while the
        # sampler indexes from the grid corner. Shift the origin by half a cell so
        # U_rep / D_obs align exactly with the path frame (no ~res/2 bias).
        half = 0.5 * grid.resolution
        self._corrector.set_field(
            u_rep, grid.resolution, grid.origin_x + half, grid.origin_y + half,
            d_obs=d_obs, known_mask=known)

    # ------------------------------------------------------------------
    # Unknown-area damping
    # ------------------------------------------------------------------
    def _map_confidence(self, grid: OccupancyGrid2D, x: float, y: float) -> float:
        """Fraction of KNOWN cells in a disk of ``unknown_radius_m`` around (x, y).

        1.0 = fully mapped neighbourhood (walls observed on both sides, so the
        repulsive forces balance); lower when (x, y) is in or near unknown space,
        where the push is unbalanced. Used to scale down the correction there.
        """
        rad = max(1, int(round(self.cfg.unknown_radius_m / grid.resolution)))
        gx, gy = grid.world_to_grid(x, y)
        x0, x1 = max(0, gx - rad), min(grid.width, gx + rad + 1)
        y0, y1 = max(0, gy - rad), min(grid.height, gy + rad + 1)
        win = grid.grid[y0:y1, x0:x1]
        if win.size == 0:
            return 1.0
        return float(np.count_nonzero(win != grid.values.unknown)) / float(win.size)

    def _dampen_unknown(self, raw_points, corrected_points,
                        grid: OccupancyGrid2D) -> Tuple[Pose2D, ...]:
        """Scale each waypoint's correction by the map confidence at its corrected
        position, so a push into/near unknown space (no opposing wall to balance
        it) is damped while a push to the centre of a fully-mapped corridor is kept
        at full strength. ``final = raw + confidence * (corrected - raw)``.
        """
        out: List[Pose2D] = []
        for r, c in zip(raw_points, corrected_points):
            conf = self._map_confidence(grid, c.x, c.y)
            if conf >= 1.0 - 1e-6:
                out.append(c)
            else:
                out.append(Pose2D(r.x + conf * (c.x - r.x),
                                  r.y + conf * (c.y - r.y), c.yaw))
        return tuple(out)

    # ------------------------------------------------------------------
    # Per-waypoint collision clip
    # ------------------------------------------------------------------
    def _clip_to_clear(self, raw_points, safe_points,
                       grid: OccupancyGrid2D) -> Tuple[Pose2D, ...]:
        """Per-waypoint safety clip of the corrected path against inflation.

        Each interior waypoint is pulled back toward its raw position only as far
        as needed to keep BOTH its adjacent segments clear (bisection), so a single
        corner-cut reverts just that waypoint while the rest stay centred. Never
        less safe than the input: a waypoint that cannot be cleared falls back to
        its raw position. Endpoints stay pinned.
        """
        checker = InflatedGridCollisionChecker(grid, self.cfg.inflate_radius_m)
        out = list(safe_points)
        n = len(out)
        if n < 3:
            return tuple(out)

        def clear(a: Pose2D, b: Pose2D) -> bool:
            return checker.segment_clear(a, b)

        # Re-evaluate EVERY interior waypoint to its most-centred clear position
        # each sweep (not only colliding ones): pulling one waypoint back can later
        # free a neighbour to return to full correction, so we recompute rather than
        # latch a one-time revert. Converges in a few sweeps; endpoints stay pinned.
        for _sweep in range(3):
            changed = False
            for i in range(1, n - 1):
                full = safe_points[i]                # t=1: full correction
                if clear(out[i - 1], full) and clear(full, out[i + 1]):
                    new = full
                else:
                    rx, ry = raw_points[i].x, raw_points[i].y
                    sx, sy = safe_points[i].x, safe_points[i].y
                    lo, hi, best = 0.0, 1.0, raw_points[i]   # t: 0=raw .. 1=corrected
                    for _ in range(6):           # bisect for the most-centred clear t
                        t = 0.5 * (lo + hi)
                        cand = Pose2D(rx + t * (sx - rx), ry + t * (sy - ry), full.yaw)
                        if clear(out[i - 1], cand) and clear(cand, out[i + 1]):
                            best, lo = cand, t
                        else:
                            hi = t
                    new = best
                if math.hypot(new.x - out[i].x, new.y - out[i].y) > 1e-6:
                    out[i] = new
                    changed = True
            if not changed:
                break
        return tuple(out)
