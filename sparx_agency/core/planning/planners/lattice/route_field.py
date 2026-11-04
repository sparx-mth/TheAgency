"""The 2D route as a cost-to-go field: the lattice search's heuristic and its search domain at once.

The lattice refines a route the 2D planner already found through the same
map. The route is therefore both the best available estimate of the way to
the goal from anywhere near it and the corridor the executable plan should
stay in: the field below gives every cell within ``corridor_m`` of the route
the route's remaining COST from its nearest point plus the distance to that
point, and ``inf`` everywhere else. The search reads it as a heuristic
(cost-weighted metres, which it divides by the step) and refuses to expand a
state the field does not cover.

The remaining cost is the route's length weighted by the cost grid it was
planned on, not its bare length: a doorway cell costs several times an open
one (the clearance and centreline layers), and a heuristic that counts it as
one step is surprised by every doorway -- on a 32 m synthetic route through
three of them plain A* flooded 60,000 states without a plan, and with the
weighted remaining it planned in a few thousand. The plan refines the route,
so the route's own cost is the right estimate of what lies ahead.

Python 3.8 syntax; numpy 1.17 API only.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

#: Spacing of the route samples the field is grown from, metres. Coarser than the grid is fine: every
#: sample paints a disc ``corridor_m`` wide, so neighbouring discs overlap many times over.
SAMPLE_SPACING_M = 0.2


def _resample(points: Sequence[Tuple[float, float]], spacing_m: float, weight=None):
    """The polyline resampled at most ``spacing_m`` apart, with the remaining (weighted) length at every sample.

    ``weight(x, y)`` prices the metre at a point; None prices every metre 1.
    """
    xs, ys = [], []
    for i in range(len(points) - 1):
        (ax, ay), (bx, by) = points[i], points[i + 1]
        length = math.hypot(bx - ax, by - ay)
        count = max(1, int(math.ceil(length / spacing_m)))
        for j in range(count):
            t = j / float(count)
            xs.append(ax + (bx - ax) * t)
            ys.append(ay + (by - ay) * t)
    xs.append(float(points[-1][0]))
    ys.append(float(points[-1][1]))
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    legs = np.hypot(np.diff(xs), np.diff(ys))
    if weight is not None:
        legs = legs * np.asarray([weight(xs[i + 1], ys[i + 1]) for i in range(len(legs))], dtype=np.float64)
    remaining = np.concatenate([np.cumsum(legs[::-1])[::-1], [0.0]])
    return xs, ys, remaining


def route_cost_to_go(points: Sequence[Tuple[float, float]], origin_x: float, origin_y: float,
                     resolution: float, shape: Tuple[int, int], corridor_m: float,
                     cost: Optional[np.ndarray] = None, blocked_cost: float = 10.0) -> np.ndarray:
    """Remaining route cost plus offset for every cell within ``corridor_m`` of the route, ``inf`` beyond.

    The first of :func:`route_corridor_fields`; see there.
    """
    return route_corridor_fields(points, origin_x, origin_y, resolution, shape, corridor_m, cost, blocked_cost)[0]


def route_corridor_fields(points: Sequence[Tuple[float, float]], origin_x: float, origin_y: float,
                          resolution: float, shape: Tuple[int, int], corridor_m: float,
                          cost: Optional[np.ndarray] = None, blocked_cost: float = 10.0
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """``(cost_to_go, offset)``: the route's remaining cost plus offset, and the lateral offset alone, per cell.

    Both are ``inf`` outside the corridor. The offset field is what prices
    straying from the route (``LatticePlannerParams.route_offset_cost``): a
    plan that must leave the medial-axis route to line up a lattice heading
    pays for every step it spends off it, and comes back.

    Args:
        points: The route, world ``(x, y)`` pairs, the goal last. At least two.
        origin_x: World x of the grid's cell ``(0, 0)`` corner.
        origin_y: World y of that corner.
        resolution: Metres per cell.
        shape: ``(H, W)`` of the grid the field is indexed like.
        corridor_m: Half-width of the corridor around the route.
        cost: The per-cell cost grid the route was planned on, to weight its
            remaining length; None weights every metre 1.
        blocked_cost: The weight of a route metre through a cell the grid
            blocks (a route planned at another standoff may cross one).

    Returns:
        Two ``(H, W)`` float64 arrays: cost-weighted metres to go, and metres
        of lateral offset.

    Raises:
        ValueError: On fewer than two points, a non-positive corridor, or a
            cost grid of another shape.
    """
    if len(points) < 2:
        raise ValueError("a route needs at least two points")
    if corridor_m <= 0.0 or resolution <= 0.0:
        raise ValueError("corridor_m and resolution must be positive")
    h, w = shape
    weight = None
    if cost is not None:
        if cost.shape != (h, w):
            raise ValueError("cost grid %s does not match the field shape %s" % (cost.shape, (h, w)))

        def weight(x, y):
            cx, cy = int(math.floor((x - origin_x) / resolution)), int(math.floor((y - origin_y) / resolution))
            if not (0 <= cx < w and 0 <= cy < h):
                return blocked_cost
            value = float(cost[cy, cx])
            return value if math.isfinite(value) else blocked_cost
    field = np.full((h, w), np.inf, dtype=np.float64)
    offset = np.full((h, w), np.inf, dtype=np.float64)
    xs, ys, remaining = _resample(points, max(resolution, SAMPLE_SPACING_M), weight)
    radius = int(math.ceil(corridor_m / resolution))
    span = np.arange(-radius, radius + 1, dtype=np.float64)
    oy, ox = np.meshgrid(span, span, indexing="ij")
    for x, y, rest in zip(xs, ys, remaining):
        # The sample's fractional cell position: distances are measured from the sample itself,
        # not from the centre of the cell it falls in, so neighbouring samples agree to the grid.
        fx, fy = (x - origin_x) / resolution, (y - origin_y) / resolution
        cx, cy = int(math.floor(fx)), int(math.floor(fy))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        wx0, wy0 = x0 - (cx - radius), y0 - (cy - radius)
        wx1, wy1 = wx0 + (x1 - x0), wy0 + (y1 - y0)
        dx = (ox[wy0:wy1, wx0:wx1] + cx + 0.5) - fx
        dy = (oy[wy0:wy1, wx0:wx1] + cy + 0.5) - fy
        dist = np.hypot(dx, dy) * resolution
        inside = dist <= corridor_m
        np.minimum(field[y0:y1, x0:x1], np.where(inside, rest + dist, np.inf), out=field[y0:y1, x0:x1])
        np.minimum(offset[y0:y1, x0:x1], np.where(inside, dist, np.inf), out=offset[y0:y1, x0:x1])
    return field, offset
