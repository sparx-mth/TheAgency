"""ESDF gradient path corrector (ROS-free).

Builds the Euclidean distance field ``D(x)`` from the BEV occupancy grid
(:class:`EsdfLayer`, exact cv2 distance transform) and ascends each waypoint up
``+∇D`` -- away from the nearest wall toward open space -- so the route is nudged
toward the centre of a corridor or doorway. This is FALCON's B-spline
``safe_distance`` idea (push control points up the ESDF) applied to a planned path.

It is a distinct strategy from :class:`PotentialFieldPathCorrector` (which builds
a Gaussian *repulsive potential* and recentres via a medial-axis line search):
here the drive is the raw distance-field gradient, sampled bilinearly by the same
:class:`PotentialFieldSampler` the potential strategy uses (the distance field is
passed as ``d_obs``; ``clearance``/``clearance_ascent`` read ``D`` and ``+∇D``).
The two share the same map-aware safety afterwards -- unknown-area damping and the
per-waypoint collision clip (:mod:`map_safety`) -- and the same pinned endpoints,
so a corrected path is never less safe than the input.

Python 3.8 compatible (the FALCON Noetic adapter imports core under 3.8): no PEP
604 unions, no ``match``/``case``. cv2-backed (via EsdfLayer); numpy-only otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.mapping.costmap.esdf_layer import EsdfLayer
from sparx_agency.core.planning.environment import OccupancyGrid2D

from ..potential_field_sampler import PotentialFieldSampler
from .base import PathCorrector, PathCorrectionResult
from .map_safety import clip_to_clear, dampen_unknown


@dataclass(frozen=True)
class EsdfCorrectorConfig:
    """Tuning for :class:`EsdfPathCorrector`.

    Field generation (:class:`EsdfLayer`):
        occ_thresh: Probability at/above which a cell counts as an obstacle.
        smooth_sigma_m: Light Gaussian blur (m) on the distance field for a cleaner
            gradient (0 disables).

    Gradient ascent:
        target_clearance_m: Stop ascending a waypoint once it is this far from the
            nearest wall (a safety floor; mirrors FALCON's ``safe_distance``). ``0``
            (default) keeps ascending until the gradient vanishes -- i.e. centre the
            waypoint on the ESDF ridge (the middle of the corridor/doorway).
        max_step_m: Cap on a single ascent step.
        max_total_shift_m: Cap on a waypoint's total displacement from its input.
        iterations: Max ascent steps per waypoint.
        lateral_only: Project each step perpendicular to the path tangent, so
            centring does not slide waypoints fore/aft and distort spacing.
        pin_last: Keep the final waypoint fixed (it is the hard goal).

    Map-aware safety (shared with the potential-field strategy; see :mod:`map_safety`):
        collision_recheck, inflate_radius_m, treat_unknown_as_free,
        unknown_damping, unknown_radius_m.
    """

    # field generation
    occ_thresh: float = 0.65
    smooth_sigma_m: float = 0.1
    # gradient ascent
    target_clearance_m: float = 0.0
    max_step_m: float = 0.2
    max_total_shift_m: float = 1.0
    iterations: int = 12
    lateral_only: bool = True
    pin_last: bool = True
    # map-aware safety
    collision_recheck: bool = True
    inflate_radius_m: float = 0.4
    treat_unknown_as_free: bool = True
    unknown_damping: bool = True
    unknown_radius_m: float = 0.75


class EsdfPathCorrector(PathCorrector):
    """Recentre a planned path off walls by ascending the ESDF gradient."""

    name = "esdf"

    def __init__(self, config: Optional[EsdfCorrectorConfig] = None) -> None:
        self.cfg = config or EsdfCorrectorConfig()
        self._layer = EsdfLayer(occ_thresh=self.cfg.occ_thresh,
                                smooth_sigma_m=self.cfg.smooth_sigma_m)
        self._field = None  # type: Optional[PotentialFieldSampler]

    @property
    def field(self) -> Optional[PotentialFieldSampler]:
        """The distance-field sampler from the most recent :meth:`correct`.

        Sample ``+∇D`` via ``field.clearance_ascent(x, y)`` to visualise the ascent
        direction. ``None`` before the first correction.
        """
        return self._field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def correct(self, path: Path2D, grid: OccupancyGrid2D) -> PathCorrectionResult:
        """Ascend each waypoint up ``+∇D`` toward the corridor centre / target clearance."""
        self._build_field(grid)
        pts = self._ascend(path.points)
        if self.cfg.unknown_damping:
            pts = dampen_unknown(path.points, pts, grid, self.cfg.unknown_radius_m)
        if self.cfg.collision_recheck:
            pts = clip_to_clear(path.points, pts, grid, self.cfg.inflate_radius_m)
        moved = sum(1 for r, s in zip(path.points, pts)
                    if math.hypot(s.x - r.x, s.y - r.y) > 1e-3)
        corrected = Path2D(points=tuple(pts), frame_id=path.frame_id,
                           metadata=dict(path.metadata, safety_corrected=True,
                                         corrector=self.name))
        return PathCorrectionResult(path=corrected, num_points=len(pts), num_moved=moved)

    # ------------------------------------------------------------------
    # Field build
    # ------------------------------------------------------------------
    def _build_field(self, grid: OccupancyGrid2D) -> None:
        """Install the grid's distance field into a bilinear sampler (same frame).

        The distance field is passed as the sampler's ``d_obs`` (so ``clearance``
        and ``clearance_ascent`` read ``D`` and ``+∇D``) and also as its primary
        field argument (unused for ascent, but the sampler requires a finite one).
        The half-cell origin shift aligns the field with ``grid_to_world`` cell
        centres, exactly as the potential-field strategy does.
        """
        raw = grid.grid
        # BEV ints -> probability grid: occupied -> 1, free -> 0, unknown -> NaN.
        # EsdfLayer treats NaN as FREE, so D is drawn ONLY from known walls
        # (unknown space is open, matching A* with unknown_blocked=False).
        p_occ = np.where(raw == grid.values.occupied, 1.0,
                         np.where(raw == grid.values.free, 0.0, np.nan)
                         ).astype(np.float32)
        known = None if self.cfg.treat_unknown_as_free else (raw != grid.values.unknown)
        esdf = self._layer.compute_from_prob_grid(p_occ, grid.resolution)
        half = 0.5 * grid.resolution
        self._field = PotentialFieldSampler(
            esdf, grid.resolution, grid.origin_x + half, grid.origin_y + half,
            d_obs=esdf, known_mask=known)

    # ------------------------------------------------------------------
    # Gradient ascent
    # ------------------------------------------------------------------
    def _ascend(self, points: Sequence[Pose2D]) -> Tuple[Pose2D, ...]:
        """Walk each interior, observed waypoint up ``+∇D`` to a ridge / target.

        Each step moves along the (optionally lateral-only) ascent direction by a
        capped amount, accepting it only while it raises the clearance; the walk
        stops at the ESDF ridge (gradient vanishes -- the corridor centre), at the
        ``target_clearance_m`` floor, or at the total-shift cap. Normals are taken
        from the ORIGINAL geometry so moving an early waypoint does not skew a later
        one's lateral direction. Endpoints are pinned; unobserved waypoints stay put.
        """
        n = len(points)
        out = [Pose2D(p.x, p.y, p.yaw) for p in points]
        if n < 3:
            return tuple(out)
        f = self._field
        hi = n - 1 if self.cfg.pin_last else n
        for i in range(1, hi):
            if not f.is_observed(points[i].x, points[i].y):
                continue
            nrm = self._unit_normal(points, i, n) if self.cfg.lateral_only else None
            if self.cfg.lateral_only and nrm is None:
                continue
            x, y = points[i].x, points[i].y
            total = 0.0
            for _ in range(max(0, self.cfg.iterations)):
                d = f.clearance(x, y)
                if d is None:
                    break
                if self.cfg.target_clearance_m > 0.0 and d >= self.cfg.target_clearance_m:
                    break                              # safety floor reached -> stop
                g = f.clearance_ascent(x, y)
                if g is None:
                    break
                gx, gy = float(g[0]), float(g[1])
                if nrm is not None:                    # keep only the sideways push
                    proj = gx * nrm[0] + gy * nrm[1]
                    gx, gy = proj * nrm[0], proj * nrm[1]
                mag = math.hypot(gx, gy)
                if mag < 1e-6:
                    break                              # at the ridge / flat field
                if self.cfg.target_clearance_m > 0.0:
                    step = min(self.cfg.max_step_m, self.cfg.target_clearance_m - d)
                else:
                    step = self.cfg.max_step_m         # centre: march toward the ridge
                step = min(step, self.cfg.max_total_shift_m - total)
                if step <= 1e-6:
                    break                              # total-shift cap reached
                nx, ny = x + (gx / mag) * step, y + (gy / mag) * step
                d_new = f.clearance(nx, ny)
                if d_new is None or d_new <= d + 1e-9:
                    break                              # no improvement -> past the ridge
                x, y, total = nx, ny, total + step
            if total > 1e-9:
                out[i] = Pose2D(x, y, points[i].yaw)
        return tuple(out)

    @staticmethod
    def _unit_normal(pts: Sequence[Pose2D], i: int, n: int) -> Optional[Tuple[float, float]]:
        """Unit normal (left of travel) of the path at ``i`` (central difference)."""
        if 0 < i < n - 1:
            tx, ty = pts[i + 1].x - pts[i - 1].x, pts[i + 1].y - pts[i - 1].y
        elif i + 1 < n:
            tx, ty = pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y
        elif i > 0:
            tx, ty = pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y
        else:
            return None
        norm = math.hypot(tx, ty)
        if norm < 1e-9:
            return None
        return (-ty / norm, tx / norm)
