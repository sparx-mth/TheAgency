"""Draw one frame of a flight's plan view: the map, the route, and the aircraft.

``inspect_recording.plan_view`` draws the same four things *once*, for the whole
flight, which answers "did this recording work". This answers a different
question -- "what was happening at second 34" -- and so has to be drawn per frame
and kept cheap enough to draw a thousand times.

What is on it, and why each is needed to read a flight:

* the **surveyed map** of the whole building, at its full extent rather than
  cropped to the flight. A route that looks sensible in a crop and absurd in the
  building is a real failure mode, and the crop hides it.
* the **goal**, because a flight is only good or bad relative to where it was
  going;
* the **planned A\\* route**, which is what the aircraft was *asked* to do;
* the **path flown so far**, which is what it did -- ending exactly at the
  aircraft marker, so the two can never disagree;
* the **aircraft**, with a heading arrow. Position and heading are ground truth
  from ``poses.npy``.

OpenCV rather than matplotlib, deliberately: a nine-hundred-frame flight is a
second of work this way and most of a minute the other, and nothing here needs
more than lines and circles.

Frames: the map is FLU/ENU with +y up, so every panel is flipped vertically at
the end -- a numpy row index counts down the image and north counts up.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

FLOWN_COLOUR = (255, 190, 60)      # BGR, cyan-blue: what it did
PLANNED_COLOUR = (90, 210, 90)     # BGR, green: what it was asked to do
GOAL_COLOUR = (96, 27, 216)        # BGR, magenta
DRONE_COLOUR = (30, 30, 30)        # BGR, near-black
FREE_SHADE = 245
UNKNOWN_SHADE = 205
OCCUPIED_SHADE = 70

MIN_PANEL_PX = 360
"""Smallest panel side worth drawing. Below this the strokes collide."""


class MapPanel:
    """Draws plan-view frames of one flight over one surveyed map.

    The map raster is rendered once and reused, because it is the same in every
    frame and by far the most expensive part.

    Args:
        grid: The scene's surveyed :class:`~...environment.OccupancyGrid2D`.
        size_px: Target side of the panel, pixels. The map's aspect ratio is
            preserved, so the result is at most this on its longer side.

    Raises:
        ValueError: If ``size_px`` is too small to draw legibly.
    """

    def __init__(self, grid, size_px: int = 540):
        if size_px < MIN_PANEL_PX:
            raise ValueError(f"panel side must be at least {MIN_PANEL_PX} px, "
                             f"got {size_px}")
        self._grid = grid
        rows, columns = grid.grid.shape
        self._scale = size_px / max(rows, columns)
        self.width = max(1, int(round(columns * self._scale)))
        self.height = max(1, int(round(rows * self._scale)))
        self._base = self._render_map()
        self._stroke = max(1, int(round(size_px / 270)))

    def _render_map(self) -> np.ndarray:
        """The static map raster, in BGR, at panel resolution."""
        values = self._grid.values
        shades = np.full(self._grid.grid.shape, UNKNOWN_SHADE, np.uint8)
        shades[self._grid.grid == values.free] = FREE_SHADE
        shades[self._grid.grid == values.occupied] = OCCUPIED_SHADE
        resized = cv2.resize(shades, (self.width, self.height),
                             interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    def to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """World metres to panel pixels, before the vertical flip."""
        column = (x - self._grid.origin_x) / self._grid.resolution
        row = (y - self._grid.origin_y) / self._grid.resolution
        return int(round(column * self._scale)), int(round(row * self._scale))

    def _polyline(self, canvas, points: Sequence[Sequence[float]], colour,
                  width: int) -> None:
        """Draw a world-frame polyline, if it has at least two points."""
        if len(points) < 2:
            return
        pixels = np.array([self.to_pixel(x, y) for x, y in points], np.int32)
        cv2.polylines(canvas, [pixels], False, colour, width, cv2.LINE_AA)

    def draw(self, flown: np.ndarray, upto: int, pose: Sequence[float],
             planned: Optional[Sequence[Sequence[float]]] = None,
             goal: Optional[Sequence[float]] = None) -> np.ndarray:
        """One panel frame.

        Args:
            flown: ``(N, 2)`` world positions, one per recorded frame.
            upto: How many of them have happened. The last one drawn is where the
                aircraft is, which is why the marker is taken from here rather
                than passed in separately -- the trail and the marker cannot then
                disagree, and a panel where they did is what made a correctly
                located aircraft look mislocalised once already.
            pose: ``(x, y, yaw)`` now. Only ``yaw`` is read; position comes from
                ``flown`` for the reason above.
            planned: The A* route as world ``(x, y)`` waypoints, or None.
            goal: World ``(x, y)`` the flight is aiming at, or None.

        Returns:
            A BGR panel, ``(height, width, 3)``, already flipped so +y is up.
        """
        canvas = self._base.copy()
        if planned:
            # Drawn wider than the flown path and underneath it, so where the two
            # agree the plan shows as a halo around the flight rather than being
            # painted out -- "it followed the route" has to be visible, and with
            # equal strokes the later line simply hides the earlier one.
            self._polyline(canvas, planned, PLANNED_COLOUR, self._stroke + 4)
            for point in planned:
                cv2.circle(canvas, self.to_pixel(point[0], point[1]),
                           self._stroke + 2, PLANNED_COLOUR, -1, cv2.LINE_AA)
        if goal is not None:
            self._draw_goal(canvas, goal)

        clipped = int(max(0, min(upto, len(flown))))
        if clipped >= 2:
            self._polyline(canvas, flown[:clipped], FLOWN_COLOUR,
                           self._stroke + 1)
        if clipped >= 1:
            self._draw_drone(canvas, flown[clipped - 1], float(pose[2]))
        return np.flipud(canvas)

    def _draw_goal(self, canvas, goal: Sequence[float]) -> None:
        """A ringed dot, which stays readable over both free space and geometry."""
        centre = self.to_pixel(goal[0], goal[1])
        radius = 3 * self._stroke
        cv2.circle(canvas, centre, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, radius, GOAL_COLOUR, self._stroke, cv2.LINE_AA)
        cv2.circle(canvas, centre, self._stroke, GOAL_COLOUR, -1, cv2.LINE_AA)

    def _draw_drone(self, canvas, position: Sequence[float], yaw: float) -> None:
        """The aircraft: a filled dot with an arrow along its heading."""
        centre = self.to_pixel(position[0], position[1])
        reach = 1.2                      # metres of arrow, so it reads as a heading
        tip = self.to_pixel(position[0] + reach * math.cos(yaw),
                            position[1] + reach * math.sin(yaw))
        cv2.arrowedLine(canvas, centre, tip, DRONE_COLOUR, self._stroke + 1,
                        cv2.LINE_AA, tipLength=0.4)
        cv2.circle(canvas, centre, 2 * self._stroke, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, 2 * self._stroke, DRONE_COLOUR, self._stroke,
                   cv2.LINE_AA)


def route_points(meta: dict, first_pose: Sequence[float]) -> list:
    """The A* route as world ``(x, y)``, starting where the aircraft did.

    ``planned_waypoints`` holds the route's waypoints but not its origin, so a
    route drawn from them alone begins at the first waypoint and appears to leave
    the aircraft behind. ``start_xy`` is prepended to close that gap.

    Args:
        meta: A recording's ``meta.json``.
        first_pose: The recording's first ``(x, y)``, used when the metadata has
            no start.

    Returns:
        ``[(x, y), ...]``, possibly empty.
    """
    waypoints = meta.get("planned_waypoints") or []
    if not waypoints:
        return []
    start = meta.get("route_start_xy") or meta.get("start_xy") or first_pose
    return [(float(start[0]), float(start[1]))] + [
        (float(point[0]), float(point[1])) for point in waypoints
    ]
