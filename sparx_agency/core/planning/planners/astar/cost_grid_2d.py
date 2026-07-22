"""Cost-grid construction for the weighted 2D A* planner (ROS-free, 3.8-safe).

Turns an occupancy grid into the three arrays the search needs:

  * ``cost``      -- per-cell traversal cost (1.0 free, ``inf`` blocked),
  * ``lethal``    -- boolean collision mask (inflated confirmed obstacles),
  * ``clearance`` -- exact Euclidean distance to the nearest obstacle (meters).

Three ideas separate this from a classic binary costmap, and all exist because a
monocular-depth BEV is noisy:

**Clearance shaping (soft wall avoidance).** Inflation is a *constraint*: widen
it and routes disappear. The clearance layer is a *preference* -- cells near a
wall cost more, fading to zero over ``clearance_margin_m`` -- so hugging a wall
is expensive but never impossible. Because the middle of a corridor is the
farthest point from both walls, this pulls the route to the centre.

**Confidence-weighted lethality (noise tolerance).** A single-frame depth
speckle painted OCCUPIED must not make the map infeasible. When a per-cell
confidence grid is supplied, only cells whose evidence reaches
``lethal_confidence`` block the search; an unconfirmed cell costs
``soft_obstacle_cost`` (high, but finite), so the route bends around it when
there is room and drives through it when there is not.

**A relaxable standoff.** Even a confirmed obstacle set can pinch a corridor
below twice the preferred radius -- a stably mis-detected cell, or a genuinely
narrow spot. Treating the preferred radius as an ultimatum means stopping, so
the build is split in two: :func:`build_cost_fields` does the expensive,
radius-*independent* work (the distance transforms), and
:func:`assemble_cost_grid` applies a radius as a mere threshold. Retrying at a
smaller standoff therefore costs a comparison, not another transform. Note that
assembly always shapes the soft cost around the *preferred* radius whatever
threshold it is given, so a relaxed plan still rides as close to the middle as
the pinch physically allows instead of hugging one side.

UNKNOWN ("gray") cells are a flat ``unknown_cost`` *and* a centring boundary:
they repel the soft layer like a wall, so a route prefers the middle of the
known-free band over the gray frontier -- without gray ever becoming lethal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D

from ..common.clearance_2d import clearance_field
from .params import WeightedAStarParams

# Tolerance on the lethal test so a cell at exactly the inflation radius (an
# integer number of cells away) is not excluded by float round-off.
_EPS = 1e-9


@dataclass(frozen=True)
class CostFields:
    """The radius-independent half of a cost grid (see :func:`build_cost_fields`).

    Attributes:
        clearance: Distance (m) from every cell to the nearest *confirmed*
            obstacle. Thresholding this at a radius yields the lethal mask, which
            is what makes a relaxed standoff cheap.
        soft_clearance: Distance (m) to the nearest cell of anything the route
            would rather avoid -- confirmed walls, gray frontier and unconfirmed
            speckle. Drives the clearance shaping.
        unknown: Boolean mask of UNKNOWN cells.
        unconfirmed: Boolean mask of OCCUPIED cells below ``lethal_confidence``.
    """
    clearance: np.ndarray
    soft_clearance: np.ndarray
    unknown: np.ndarray
    unconfirmed: np.ndarray


def split_by_confidence(
    occupied: np.ndarray, confidence: Optional[np.ndarray], lethal_confidence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Split OCCUPIED cells into confirmed (blocking) and unconfirmed (costly).

    Args:
        occupied: ``(H, W)`` boolean mask of OCCUPIED cells.
        confidence: Co-registered ``(H, W)`` per-cell evidence in ``[0, 1]``, or
            None when the caller has no confidence grid.
        lethal_confidence: Evidence a cell needs to block the search. ``<= 0``
            disables the split (every OCCUPIED cell is confirmed), which is the
            classic behaviour.

    Returns:
        ``(confirmed, unconfirmed)`` boolean masks; ``unconfirmed`` is empty
        whenever the split is disabled or no confidence grid was supplied.

    Raises:
        ValueError: If ``confidence`` is not co-registered with ``occupied``.
    """
    if confidence is None or lethal_confidence <= 0.0:
        return occupied, np.zeros_like(occupied)
    if confidence.shape != occupied.shape:
        raise ValueError(
            "confidence grid %s is not co-registered with the occupancy grid %s"
            % (confidence.shape, occupied.shape))
    confirmed = occupied & (confidence >= lethal_confidence)
    return confirmed, occupied & ~confirmed


def build_cost_fields(
    grid: OccupancyGrid2D,
    params: WeightedAStarParams,
    confidence: Optional[np.ndarray] = None,
) -> CostFields:
    """Compute the distance transforms a cost grid is assembled from.

    This is the expensive half, and it does not depend on the standoff radius --
    so a caller retrying at a relaxed radius should reuse these fields rather
    than rebuild them.

    Args:
        grid: Source occupancy grid (uses ``grid.values`` for OCCUPIED/UNKNOWN).
        params: Supplies the confidence split and the band worth transforming.
        confidence: Optional ``(H, W)`` per-cell OCCUPIED confidence in
            ``[0, 1]``, co-registered with ``grid``.

    Returns:
        The :class:`CostFields` for ``grid``.

    Raises:
        ValueError: If ``confidence`` is not co-registered with ``grid``.
    """
    res = float(grid.resolution)
    data = grid.grid
    occupied = data == grid.values.occupied
    unknown = data == grid.values.unknown
    confirmed, unconfirmed = split_by_confidence(
        occupied, confidence, params.lethal_confidence)

    # Distances are only ever read inside the shaped band, so bound the transform
    # to it (the cost layer treats everything beyond as equally open).
    band = max(params.inflate_radius_m + params.clearance_margin_m, res)
    clearance = clearance_field(confirmed, res, band)

    # The soft layer repels from everything the route would rather keep away
    # from: confirmed walls, the gray frontier, and unconfirmed speckle alike.
    # Re-use the obstacle field when those add nothing, so the common
    # fully-known map pays for a single transform.
    if unknown.any() or unconfirmed.any():
        barrier = confirmed | unknown | unconfirmed
        soft_clearance = clearance_field(barrier, res, band)
    else:
        soft_clearance = clearance
    return CostFields(clearance, soft_clearance, unknown, unconfirmed)


def assemble_cost_grid(
    fields: CostFields, params: WeightedAStarParams, inflate_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a standoff radius to precomputed fields: the cheap half.

    Args:
        fields: Output of :func:`build_cost_fields` for the grid in question.
        params: Supplies the shaping weights and the flat UNKNOWN / unconfirmed
            costs.
        inflate_m: Standoff to enforce. Cells within this of a confirmed obstacle
            are lethal. Pass ``params.inflate_radius_m`` for the preferred
            standoff, or less to relax it.

    Returns:
        ``(cost, lethal)``.
    """
    lethal = fields.clearance <= inflate_m + _EPS

    cost = np.ones(fields.clearance.shape, dtype=np.float64)
    if params.clearance_weight > 0.0 and params.clearance_margin_m > 0.0:
        # Shaped around the PREFERRED radius, not the one being enforced: under a
        # relaxed standoff the band between the two is passable but maximally
        # penalised, so a squeeze still rides as centred as it physically can.
        deficit = 1.0 - ((fields.soft_clearance - params.inflate_radius_m)
                         / params.clearance_margin_m)
        np.clip(deficit, 0.0, 1.0, out=deficit)
        cost += float(params.clearance_weight) * deficit

    # Flat costs override the shaping (gray is not "near a wall", it is gray);
    # lethal overrides everything.
    cost[fields.unknown] = (np.inf if params.unknown_blocked
                            else float(params.unknown_cost))
    cost[fields.unconfirmed] = float(params.soft_obstacle_cost)
    cost[lethal] = np.inf
    return cost, lethal


def build_cost_grid(
    grid: OccupancyGrid2D,
    params: WeightedAStarParams,
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the cost map, collision mask and clearance field at the preferred standoff.

    Convenience wrapper over :func:`build_cost_fields` + :func:`assemble_cost_grid`
    for callers that do not need to retry at a relaxed radius.

    Args:
        grid: Source occupancy grid (uses ``grid.values`` for OCCUPIED/UNKNOWN).
        params: Weighting, inflation and clearance-shaping parameters.
        confidence: Optional ``(H, W)`` per-cell OCCUPIED confidence in
            ``[0, 1]``, co-registered with ``grid``. Used only when
            ``params.lethal_confidence > 0``.

    Returns:
        ``(cost, lethal, clearance)``. ``cost`` is ``(H, W)`` float64 (1.0 for
        open known-free, rising toward walls, ``inf`` where blocked); ``lethal``
        is the boolean mask used for line-of-sight and collision checks;
        ``clearance`` is the distance in meters from every cell to the nearest
        *confirmed* obstacle, exact up to the shaped band and clamped above it.

    Raises:
        ValueError: If ``confidence`` is not co-registered with ``grid``.
    """
    fields = build_cost_fields(grid, params, confidence)
    cost, lethal = assemble_cost_grid(fields, params, params.inflate_radius_m)
    return cost, lethal, fields.clearance


def build_collision_mask(
    grid: OccupancyGrid2D, params: WeightedAStarParams, inflate_m: float
) -> np.ndarray:
    """Inflated mask of EVERY OCCUPIED cell, whatever its confidence.

    The detection counterpart of :func:`assemble_cost_grid`'s ``lethal``. Planning
    is allowed to route *through* an unconfirmed cell so that one noisy depth
    frame cannot make the map infeasible -- but a caller asking "is my route
    about to hit something?" must still be told about that cell, because its own
    confirmation gates decide whether to act and they are what bound the time a
    robot may fly toward a cell the map keeps flagging. Applying the relaxation
    here would disarm those backstops, so it is deliberately left out.

    Args:
        grid: Source occupancy grid.
        params: Supplies nothing but the resolution floor; the radius is explicit.
        inflate_m: Standoff to test against. Pass the radius the committed route
            was actually planned at -- testing a relaxed route against the
            preferred radius would report a collision on every frame.

    Returns:
        ``(H, W)`` boolean mask, True where a route would collide.
    """
    res = float(grid.resolution)
    occupied = grid.grid == grid.values.occupied
    # Only the lethal threshold is read, so bound the transform there rather than
    # at the (much wider) shaped band: everything beyond clamps above it anyway.
    band = max(inflate_m, res)
    return clearance_field(occupied, res, band) <= inflate_m + _EPS
