"""Trajectory simplification / cleanup for 2D waypoint paths (ROS-free).

Four geometry-only passes turn a (potential-field-corrected) waypoint path into a
cleaner one that a stop-and-turn follower can fly without spurious yaws, applied
in this order:

  1. ``thin_by_spacing_2d`` (merge)   — collapse near-duplicate points the field
     left almost on top of each other (no turn protection: even at a corner two
     coincident points are one point).
  2. ``smooth_zigzags_2d``            — when a middle point makes the path reverse
     (right→left→right), pull it toward the line between its neighbours instead of
     deleting it, so the exaggerated swing flattens. (Step 1 then repeats, because
     smoothing pulls the swing points close together and the residue should merge.)
  3. ``simplify_collinear_capped_2d`` — drop middle points that lie on the "same
     plane" (heading change below a threshold), but never let dropping one create
     a leg longer than ``max_segment`` (so you never get a too-coarse path).
  4. ``thin_by_spacing_2d`` (spacing) — enforce a minimum spacing on the straights
     while KEEPING genuine turns, which may legitimately sit closer together.

Why a ``clear_fn``: steps that move (2) or remove (1, 3, 4) waypoints could let a
new straight segment clip an obstacle the corrector just avoided. To stay map-free
yet never degrade safety, every such step is gated by an injected
``clear_fn(a, b) -> bool`` (True iff the world segment ``a→b`` is obstacle-free),
exactly the pattern :func:`...corner_rounding_2d.chamfer_corners_2d` uses. With no
``clear_fn`` (e.g. unit tests) the gate is a no-op and the passes are pure geometry.

This is NOT the potential field: it derives no forces and never pushes toward open
space. It only re-shapes the waypoint set and checks the result stays clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, radians
from typing import Callable, List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D

# Injected obstacle test: True iff the straight world segment a->b is clear.
ClearFn = Callable[[Pose2D, Pose2D], bool]


# ── geometry helpers ─────────────────────────────────────────────────────────
def _turn_angle(prev: Pose2D, v: Pose2D, nxt: Pose2D) -> float:
    """Heading deviation at vertex ``v`` on ``prev -> v -> nxt`` (rad, 0=straight)."""
    ux, uy = v.x - prev.x, v.y - prev.y
    wx, wy = nxt.x - v.x, nxt.y - v.y
    lu, lw = hypot(ux, uy), hypot(wx, wy)
    if lu < 1e-9 or lw < 1e-9:
        return 0.0
    return abs(atan2(ux * wy - uy * wx, ux * wx + uy * wy))


def _clear(a: Pose2D, b: Pose2D, clear_fn: Optional[ClearFn]) -> bool:
    """True if no ``clear_fn`` was injected, else delegate to it."""
    return clear_fn is None or clear_fn(a, b)


# ── individual passes (pure functions) ───────────────────────────────────────
def thin_by_spacing_2d(
    points: Sequence[Pose2D],
    min_spacing: float,
    protect_turn_rad: float = 0.0,
    max_segment: float = 0.0,
    clear_fn: Optional[ClearFn] = None,
) -> List[Pose2D]:
    """Drop points that crowd within ``min_spacing`` of the previously kept one.

    A point is kept when ANY of: it is at least ``min_spacing`` from the last kept
    point; it is a genuine turn (heading change >= ``protect_turn_rad`` > 0);
    dropping it would leave a bypass leg longer than ``max_segment`` (> 0); or that
    bypass leg would route through an obstacle (``clear_fn``). ``protect_turn_rad
    <= 0`` protects nothing (a pure merge); ``max_segment <= 0`` disables the cap.
    The turn angle uses the ORIGINAL-input neighbours (points[i-1], points[i+1])
    so a genuinely sharp vertex is judged a turn regardless of which neighbours
    survived the thinning. Endpoints are always kept; consequently the final leg
    may be shorter than ``min_spacing`` (the endpoint is never dropped).
    """
    if len(points) <= 2 or min_spacing <= 0.0:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        p = points[i]
        protected = (protect_turn_rad > 0.0
                     and _turn_angle(points[i - 1], p, points[i + 1]) >= protect_turn_rad)
        too_close = out[-1].distance_to(p) < min_spacing
        gap_ok = max_segment <= 0.0 or out[-1].distance_to(points[i + 1]) <= max_segment
        if ((not protected) and too_close and gap_ok
                and _clear(out[-1], points[i + 1], clear_fn)):
            continue                       # drop the crowding, non-turn point
        out.append(p)
    out.append(points[-1])
    return out


def smooth_zigzags_2d(
    points: Sequence[Pose2D],
    angle_rad: float,
    strength: float,
    passes: int = 1,
    clear_fn: Optional[ClearFn] = None,
) -> List[Pose2D]:
    """Flatten reversing middle points toward the line between their neighbours.

    A vertex whose heading change exceeds ``angle_rad`` is moved a fraction
    ``strength`` (0..1) of the way to the midpoint of its two neighbours, but only
    if both new half-segments stay clear (``clear_fn``) — so a genuine corner that
    hugs an obstacle (its bypassing line would clip) is left alone, while a
    field-induced swing in open space is relaxed. Endpoints never move.

    The update is **Gauss-Seidel** (in-place, left to right) rather than Jacobi:
    when vertex ``i`` is moved its left neighbour ``pts[i-1]`` is already the
    committed (final) position for this pass, so ``clear(pts[i-1], nv)`` validates
    the REALIZED left leg; and when ``i+1`` is processed its left neighbour is the
    committed ``pts[i]``, so the realized leg between two adjacent moved vertices is
    validated too. A Jacobi pass (reading only old positions) would leave that
    shared leg unchecked — a path could then clip an obstacle the corrector avoided.
    Repeated ``passes`` converge the smoothing; ``strength`` MUST be in [0, 1] (a
    convex blend) — larger values overshoot and diverge (the config validates it).
    """
    pts = list(points)
    if len(pts) <= 2 or strength <= 0.0 or angle_rad <= 0.0:
        return pts
    for _ in range(max(0, int(passes))):
        for i in range(1, len(pts) - 1):
            prev, v, nxt = pts[i - 1], pts[i], pts[i + 1]   # prev already committed
            if _turn_angle(prev, v, nxt) < angle_rad:
                continue
            tx, ty = 0.5 * (prev.x + nxt.x), 0.5 * (prev.y + nxt.y)
            nv = Pose2D(v.x + strength * (tx - v.x), v.y + strength * (ty - v.y))
            if _clear(prev, nv, clear_fn) and _clear(nv, nxt, clear_fn):
                pts[i] = nv                # commit; realized legs stay validated
    return pts


def simplify_collinear_capped_2d(
    points: Sequence[Pose2D],
    angle_rad: float,
    max_segment: float,
    clear_fn: Optional[ClearFn] = None,
) -> List[Pose2D]:
    """Drop near-collinear middle points, capped so no leg exceeds ``max_segment``.

    A middle point is removed when the path barely bends there (heading change
    below ``angle_rad``) AND the segment that would bypass it stays within
    ``max_segment`` (``<= 0`` disables the cap) AND that segment is clear
    (``clear_fn``). This collapses "same-plane" runs to their endpoints — e.g.
    (1,3),(2,3),(4,3) -> (1,3),(4,3) — without ever creating a gap so long the
    follower loses the corridor. Endpoints are always kept.
    """
    if len(points) <= 2 or angle_rad <= 0.0:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        nxt = points[i + 1]
        nearly_straight = _turn_angle(out[-1], points[i], nxt) < angle_rad
        gap_ok = max_segment <= 0.0 or out[-1].distance_to(nxt) <= max_segment
        if nearly_straight and gap_ok and _clear(out[-1], nxt, clear_fn):
            continue                       # drop the redundant collinear point
        out.append(points[i])
    out.append(points[-1])
    return out


# ── config / result / orchestrator ───────────────────────────────────────────
@dataclass(frozen=True)
class TrajectorySimplifierConfig:
    """Knobs for :class:`TrajectorySimplifier2D` (meters / degrees).

    Each pass has an ``*_enabled`` flag so any one can be isolated. Angles are
    heading-change thresholds in degrees; distances are in meters.
    """
    merge_enabled: bool = True
    merge_radius_m: float = 0.30          # points closer than this collapse to one
    zigzag_enabled: bool = True
    zigzag_angle_deg: float = 60.0        # heading change above this = a swing to flatten
    zigzag_strength: float = 0.5          # 0..1 fraction moved toward neighbour midpoint
    zigzag_passes: int = 1
    collinear_enabled: bool = True
    collinear_angle_deg: float = 10.0     # below this a middle point is "same plane"
    max_segment_m: float = 3.0            # never drop a point if the bypass leg exceeds this
    min_spacing_enabled: bool = True
    min_spacing_m: float = 1.0            # straight-run minimum spacing
    turn_keep_deg: float = 25.0           # turns sharper than this may sit closer than min_spacing

    def __post_init__(self) -> None:
        """Reject misconfiguration loudly (no silent absurd output)."""
        if not 0.0 <= self.zigzag_strength <= 1.0:
            raise ValueError(
                "zigzag_strength must be in [0, 1] (a convex blend); got %r -- "
                "larger values overshoot and the smoothing diverges"
                % (self.zigzag_strength,))
        if self.zigzag_passes < 1:
            raise ValueError("zigzag_passes must be >= 1, got %r" % (self.zigzag_passes,))
        for name in ("merge_radius_m", "zigzag_angle_deg", "collinear_angle_deg",
                     "max_segment_m", "min_spacing_m", "turn_keep_deg"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))


@dataclass(frozen=True)
class SimplifyResult:
    """Output of :meth:`TrajectorySimplifier2D.simplify`."""
    points: Tuple[Pose2D, ...]
    num_in: int
    num_out: int
    num_moved: int                        # points repositioned by zigzag smoothing


def _count_moved(original: Sequence[Pose2D], out: Sequence[Pose2D]) -> int:
    """Output points not coincident with any input point (i.e. smoothed)."""
    keys = set((round(p.x, 6), round(p.y, 6)) for p in original)
    return sum(1 for p in out if (round(p.x, 6), round(p.y, 6)) not in keys)


class TrajectorySimplifier2D:
    """Run the enabled cleanup passes in order; pure geometry + optional ``clear_fn``."""

    def __init__(self, config: Optional[TrajectorySimplifierConfig] = None) -> None:
        self.cfg = config or TrajectorySimplifierConfig()

    def simplify(
        self, points: Sequence[Pose2D], clear_fn: Optional[ClearFn] = None
    ) -> SimplifyResult:
        cfg = self.cfg
        turn_keep_rad = radians(cfg.turn_keep_deg)
        original = list(points)
        pts = list(points)
        if len(pts) >= 3:
            # Merge protects genuine turns (turn_keep_rad): a sharp corner that
            # happens to sit within merge_radius is kept; only non-turning
            # near-duplicates the field left behind collapse.
            if cfg.merge_enabled:
                pts = thin_by_spacing_2d(pts, cfg.merge_radius_m, turn_keep_rad,
                                         0.0, clear_fn)
            if cfg.zigzag_enabled:
                pts = smooth_zigzags_2d(pts, radians(cfg.zigzag_angle_deg),
                                        cfg.zigzag_strength, cfg.zigzag_passes, clear_fn)
                # Smoothing pulls the swing points together; re-merge the
                # non-turning residue (still protecting any real corner).
                if cfg.merge_enabled:
                    pts = thin_by_spacing_2d(pts, cfg.merge_radius_m, turn_keep_rad,
                                             0.0, clear_fn)
            if cfg.collinear_enabled:
                pts = simplify_collinear_capped_2d(pts, radians(cfg.collinear_angle_deg),
                                                   cfg.max_segment_m, clear_fn)
            # min-spacing carries the SAME max_segment cap as the collinear pass, so
            # it never drops a point the cap deliberately kept (no over-long leg).
            if cfg.min_spacing_enabled:
                pts = thin_by_spacing_2d(pts, cfg.min_spacing_m, turn_keep_rad,
                                         cfg.max_segment_m, clear_fn)
        return SimplifyResult(points=tuple(pts), num_in=len(original),
                              num_out=len(pts), num_moved=_count_moved(original, pts))
