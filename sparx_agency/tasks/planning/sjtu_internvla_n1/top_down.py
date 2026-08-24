"""The right-hand panel of a recorded run: N1's route, drawn on the building.

Split out of :mod:`recording` because it grew a second job. It draws two views
of the same flight side by side, and the pair is the point:

* **overview** -- the whole hospital at once, with the trail on it. This is what
  answers "did it enter every room": a route is only coverage if you can see the
  rooms it did not reach.
* **local** -- a fixed-span window that follows the aircraft, where a 0.25 m
  action step is actually several pixels wide and the committed route, the
  speculative tail and the heading are legible.

A single view cannot do both. The hospital is 25.6 x 56 m; drawn whole into a
640 x 480 panel it is about 8 px per metre, at which a step the policy actually
commits to is two pixels long.

When no map is configured the renderer falls back to its original behaviour --
graph paper with axes fitted to the trail -- so the package still runs against a
world whose map has not been built.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is present in the venv and ROS env
    cv2 = None

from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    OccupancyMapImage,
)

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX; kept as a literal so this imports without cv2.

TRAIL_BGR = (0, 200, 0)
COMMITTED_BGR = (0, 255, 255)
PLAN_BGR = (0, 140, 220)
POSE_BGR = (255, 255, 255)
START_BGR = (255, 160, 0)


class TopDownRenderer:
    """Accumulate the flight and draw N1's route from above.

    Stateful: it remembers where the drone has been so the panel grows a
    breadcrumb trail, which is what makes "reach every area at least once"
    legible as coverage rather than as a single moving dot.

    Args:
        size: ``(width, height)`` of the whole right panel.
        margin_m: Padding around the fitted extent, used only without a map.
        trail_max: How many trail points to keep.
        backdrop: The occupancy map to draw on, or ``None`` for graph paper.
        local_span_m: The side of the world square the following view shows.
        overview_fraction: Share of the panel width given to the whole-building
            view. The rest goes to the local view.
    """

    def __init__(self, size=(640, 480), margin_m=1.5, trail_max=4000,
                 backdrop=None, local_span_m=14.0, overview_fraction=0.42):
        # type: (Tuple[int, int], float, int, Optional[OccupancyMapImage], float, float) -> None
        self.w, self.h = int(size[0]), int(size[1])
        self.margin_m = float(margin_m)
        self.trail = []  # type: List[Tuple[float, float]]
        self.trail_max = int(trail_max)
        self.backdrop = backdrop
        self.local_span_m = float(local_span_m)
        self.overview_fraction = float(overview_fraction)
        self._bounds = None  # (min_x, min_y, max_x, max_y)

    def add_pose(self, x, y):
        # type: (float, float) -> None
        """Record where the drone is; extends the trail and the view bounds.

        A non-finite sample is dropped rather than stored. The dedup test below
        is False for NaN, so a single NaN from a violent contact would be kept
        and every subsequent frame would raise out of the 10 Hz timer -- taking
        the recorder down and, through the launch file's `on_exit=Shutdown()`,
        the flight with it.
        """
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        if self.trail and abs(x - self.trail[-1][0]) < 0.05 and abs(y - self.trail[-1][1]) < 0.05:
            return
        self.trail.append((float(x), float(y)))
        if len(self.trail) > self.trail_max:
            self.trail = self.trail[-self.trail_max:]

    # -- rendering -------------------------------------------------------

    def render(self, pose, committed_xy, full_xy):
        # type: (Optional[Tuple[float, float, float]], Optional[np.ndarray], Optional[np.ndarray]) -> np.ndarray
        """Draw the top-down panel.

        Args:
            pose: ``(x, y, yaw)`` current world pose, or None.
            committed_xy: ``(N, 2)`` world polyline N1 is committed to, or None.
            full_xy: ``(M, 2)`` world polyline of the whole prediction, or None.

        Returns:
            A ``(h, w, 3)`` BGR panel.
        """
        if self.backdrop is None:
            return self._render_fitted(pose, committed_xy, full_xy)
        return self._render_on_map(pose, committed_xy, full_xy)

    def _render_on_map(self, pose, committed_xy, full_xy):
        # The two widths must SUM to self.w. Clamping each independently lets
        # the composed frame come out wider than the panel the VideoWriter was
        # opened for -- and cv2 drops a wrong-sized frame silently, with no
        # exception and no return value, so the recorder counts frames it never
        # wrote and the run reports a successful recording of a 258-byte file.
        overview_w = int(round(self.w * min(max(self.overview_fraction, 0.15), 0.7)))
        overview_w = max(120, min(overview_w, self.w - 120))
        local_w = self.w - overview_w

        overview = self.backdrop.whole((overview_w, self.h))
        left = overview.image.copy()
        self._draw_trail(left, overview.extent)
        self._polyline(left, committed_xy, overview.extent, COMMITTED_BGR, 2)
        self._draw_pose(left, pose, overview.extent, arrow_px=10, dot_px=3)
        _banner(left, "HOSPITAL (whole)")

        centre = self._focus(pose)
        local = self.backdrop.window(centre[0], centre[1], self.local_span_m,
                                     (local_w, self.h))
        right = local.image.copy()
        self._range_rings(right, centre, local.extent)
        self._draw_trail(right, local.extent, thickness=2)
        self._polyline(right, full_xy, local.extent, PLAN_BGR, 1)
        self._polyline(right, committed_xy, local.extent, COMMITTED_BGR, 3)
        self._draw_pose(right, pose, local.extent, arrow_px=22, dot_px=5)
        _banner(right, "N1 ROUTE  committed=yellow plan=orange trail=green")

        panel = np.hstack((left, right))
        cv2.line(panel, (overview_w, 0), (overview_w, self.h), (90, 90, 95), 1)
        return panel

    def _render_fitted(self, pose, committed_xy, full_xy):
        """The no-map fallback: graph paper with axes fitted to the flight."""
        panel = np.full((self.h, self.w, 3), 24, dtype=np.uint8)
        extra = []  # type: List[Tuple[float, float]]
        for arr in (committed_xy, full_xy):
            if arr is not None and len(arr):
                extra.extend((float(p[0]), float(p[1])) for p in np.asarray(arr).reshape(-1, 2))
        if pose is not None:
            extra.append((pose[0], pose[1]))
        extent = self._fit(extra)
        _graph_paper(panel, extent)
        self._draw_trail(panel, extent)
        self._polyline(panel, full_xy, extent, PLAN_BGR, 1)
        self._polyline(panel, committed_xy, extent, COMMITTED_BGR, 3)
        self._draw_pose(panel, pose, extent, arrow_px=18, dot_px=4)
        _banner(panel, "N1 ROUTE (top-down)  committed=yellow plan=orange trail=green")
        return panel

    # -- helpers ---------------------------------------------------------

    def _focus(self, pose):
        # type: (Optional[Tuple[float, float, float]]) -> Tuple[float, float]
        """Where the following view is centred: the drone, else the last known spot."""
        if pose is not None:
            return (float(pose[0]), float(pose[1]))
        if self.trail:
            return self.trail[-1]
        return (0.5 * (self.backdrop.origin_x + self.backdrop.max_x),
                0.5 * (self.backdrop.origin_y + self.backdrop.max_y))

    def _fit(self, extra):
        # type: (List[Tuple[float, float]]) -> Tuple[float, float, float, float]
        pts = list(self.trail) + list(extra)
        if not pts:
            return (-1.0, -1.0, 1.0, 1.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo_x, hi_x = min(xs) - self.margin_m, max(xs) + self.margin_m
        lo_y, hi_y = min(ys) - self.margin_m, max(ys) + self.margin_m
        # Keep at least a few metres of span so a hovering start is not zoomed in absurdly.
        if hi_x - lo_x < 4.0:
            c = 0.5 * (lo_x + hi_x)
            lo_x, hi_x = c - 2.0, c + 2.0
        if hi_y - lo_y < 4.0:
            c = 0.5 * (lo_y + hi_y)
            lo_y, hi_y = c - 2.0, c + 2.0
        return (lo_x, lo_y, hi_x, hi_y)

    def _draw_trail(self, panel, extent, thickness=2):
        if len(self.trail) >= 2:
            pts = np.array([_to_px(panel, extent, x, y) for x, y in self.trail],
                           dtype=np.int32)
            cv2.polylines(panel, [pts], False, TRAIL_BGR, thickness, cv2.LINE_AA)
        if self.trail:
            cv2.circle(panel, _to_px(panel, extent, *self.trail[0]), 4,
                       START_BGR, -1, cv2.LINE_AA)

    @staticmethod
    def _polyline(panel, xy, extent, color, thick):
        if xy is None or not len(xy):
            return
        pts = np.array([_to_px(panel, extent, float(p[0]), float(p[1]))
                        for p in np.asarray(xy).reshape(-1, 2)], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(panel, [pts], False, color, thick, cv2.LINE_AA)
        cv2.circle(panel, tuple(int(v) for v in pts[-1]), 4, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_pose(panel, pose, extent, arrow_px, dot_px):
        if pose is None:
            return
        px, py = _to_px(panel, extent, pose[0], pose[1])
        hx = int(px + arrow_px * np.cos(pose[2]))
        hy = int(py - arrow_px * np.sin(pose[2]))
        cv2.arrowedLine(panel, (px, py), (hx, hy), POSE_BGR, 2, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(panel, (px, py), dot_px, POSE_BGR, -1, cv2.LINE_AA)

    @staticmethod
    def _range_rings(panel, centre, extent):
        """Metre rings around the aircraft, so distances are readable at a glance."""
        cx, cy = _to_px(panel, extent, centre[0], centre[1])
        span_x = extent[2] - extent[0]
        px_per_m = panel.shape[1] / max(1e-6, span_x)
        for metres in (2.0, 5.0):
            cv2.circle(panel, (cx, cy), int(metres * px_per_m), (70, 70, 74), 1, cv2.LINE_AA)


def _to_px(panel, extent, x, y):
    # type: (np.ndarray, Tuple[float, float, float, float], float, float) -> Tuple[int, int]
    """World metres -> pixel in ``panel``, y-up to y-down.

    Half-open in both axes, and the row is measured DOWN from ``max_y`` rather
    than up from ``min_y`` minus one. Mixing a half-open column scale with a
    closed-interval row offset -- which is the natural thing to write -- puts
    the overlay one pixel off the backdrop it is drawn on, and one pixel of the
    overview is 0.12 m in a building whose doorways are 0.93 m.
    """
    min_x, min_y, max_x, max_y = extent
    h, w = panel.shape[:2]
    sx = w / max(1e-6, max_x - min_x)
    sy = h / max(1e-6, max_y - min_y)
    col = int(np.floor((x - min_x) * sx))
    row = int(np.floor((max_y - y) * sy))
    return (min(max(col, 0), w - 1), min(max(row, 0), h - 1))


def _graph_paper(panel, extent):
    min_x, min_y, max_x, max_y = extent
    for gx in range(int(np.floor(min_x)), int(np.ceil(max_x)) + 1):
        cv2.line(panel, _to_px(panel, extent, gx, min_y),
                 _to_px(panel, extent, gx, max_y), (40, 40, 40), 1)
    for gy in range(int(np.floor(min_y)), int(np.ceil(max_y)) + 1):
        cv2.line(panel, _to_px(panel, extent, min_x, gy),
                 _to_px(panel, extent, max_x, gy), (40, 40, 40), 1)


def _banner(panel, text):
    w = panel.shape[1]
    cv2.rectangle(panel, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 17), _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
