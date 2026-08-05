"""Draw one frame of "FALCON is exploring this building".

The point of the video is to make three things visible at once, because each one
explains the others:

* **what FALCON knows** -- the occupancy slab at cruise height, growing out of
  the dark. Unknown is the background, so the map literally appears;
* **where the aircraft has been** -- its trail, which is what turned the unknown
  into the known;
* **the gap between plan and aircraft** -- the reference point FALCON is
  currently commanding, drawn with a line to where the aircraft actually is.
  That line is the whole reason the outer-loop controller exists, and in a
  geometry-only simulator it would have zero length.

Pure numpy + OpenCV, no ROS and no display: a headless container renders these
straight into an MP4. Kept Python 3.8 compatible and free of anything newer than
OpenCV 4.2, because the ROS Noetic container is what runs it.

The view auto-orients. Several of the exploration boxes are long and thin (a
51 m north-south spine is 3x taller than it is wide), and drawing those upright
wastes most of a landscape frame, so the canvas is turned 90 degrees when that
fits better. A compass in the corner says which way was chosen.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

COLOUR_UNKNOWN = (26, 22, 20)      # BGR: near-black, the unexplored background
COLOUR_FREE = (86, 74, 58)         # muted slate -- known to be empty
COLOUR_OCCUPIED = (96, 170, 240)   # warm amber -- known to be solid
COLOUR_TRAIL = (200, 200, 90)      # cyan, where the aircraft has flown
COLOUR_AIRCRAFT = (120, 255, 120)  # green
COLOUR_REFERENCE = (220, 110, 235) # magenta, what FALCON is asking for
COLOUR_ERROR = (90, 90, 235)       # red, the plan-to-aircraft gap
COLOUR_TEXT = (225, 225, 225)
COLOUR_PANEL = (40, 34, 30)

MARGIN_PX = 44
TRAIL_MAX_POINTS = 6000


class ExplorationCanvas:
    """Maps a world-frame exploration box onto a fixed-size video frame.

    Args:
        bounds: ``(min_x, min_y, max_x, max_y)`` of the region to draw, metres.
        size: Output ``(width, height)`` in pixels.
        fov_deg: Horizontal field of view drawn as the aircraft's sight wedge.
        sight_m: How far the sight wedge reaches, metres. Match FALCON's
            ``tsdf/raycast_max`` -- beyond it the aircraft sees nothing that gets
            mapped, and drawing further would overstate what it knows.

    Raises:
        ValueError: If the bounds are empty on either axis.
    """

    def __init__(self, bounds, size=(1280, 720), fov_deg=90.0, sight_m=5.0):
        # type: (tuple, tuple, float, float) -> None
        min_x, min_y, max_x, max_y = (float(v) for v in bounds)
        if not (max_x > min_x and max_y > min_y):
            raise ValueError("empty draw bounds %r" % (bounds,))
        self.bounds = (min_x, min_y, max_x, max_y)
        self.width, self.height = int(size[0]), int(size[1])
        self.fov_deg = float(fov_deg)
        self.sight_m = float(sight_m)

        span_x, span_y = max_x - min_x, max_y - min_y
        # Turn the view when the region is taller than it is wide: the frame is
        # landscape, and an upright 51 m corridor would use a fifth of it.
        self.rotated = span_y > span_x
        across, along = (span_y, span_x) if self.rotated else (span_x, span_y)
        usable_w = self.width - 2 * MARGIN_PX
        usable_h = self.height - 2 * MARGIN_PX
        self.scale = min(usable_w / across, usable_h / along)
        self._offset = (
            MARGIN_PX + 0.5 * (usable_w - across * self.scale),
            MARGIN_PX + 0.5 * (usable_h - along * self.scale),
        )

    def to_pixels(self, points):
        # type: (np.ndarray) -> np.ndarray
        """Project world ``(N, 2)`` metres to ``(N, 2)`` integer pixels.

        Screen y grows downward, so the world axis drawn vertically is flipped;
        without that the map comes out mirrored and every left turn looks like a
        right one.
        """
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        min_x, min_y, max_x, max_y = self.bounds
        if self.rotated:
            across = points[:, 1] - min_y
            along = max_x - points[:, 0]
        else:
            across = points[:, 0] - min_x
            along = max_y - points[:, 1]
        pixels = np.empty_like(points)
        pixels[:, 0] = self._offset[0] + across * self.scale
        pixels[:, 1] = self._offset[1] + along * self.scale
        return np.rint(pixels).astype(np.int32)

    def _heading_pixels(self, yaw, length_m):
        # type: (float, float) -> tuple
        """A world-frame heading as a pixel-space offset of the given length."""
        direction = np.array([[math.cos(yaw), math.sin(yaw)]], dtype=np.float32) * length_m
        origin = self.to_pixels(np.zeros((1, 2), np.float32))
        tip = self.to_pixels(direction)
        return int(tip[0, 0] - origin[0, 0]), int(tip[0, 1] - origin[0, 1])

    def blank(self):
        # type: () -> np.ndarray
        """A frame containing nothing but unexplored space."""
        frame = np.empty((self.height, self.width, 3), np.uint8)
        frame[:, :] = COLOUR_UNKNOWN
        return frame

    def draw_cells(self, frame, points, colour, radius=1):
        # type: (np.ndarray, np.ndarray, tuple, int) -> None
        """Paint one occupancy class as a block of pixels per voxel.

        Drawn by direct index assignment rather than one ``cv2.circle`` per
        point: a whole-building slab is a few hundred thousand voxels twice a
        second, and a Python loop over them does not keep up with the video.
        """
        if points is None or len(points) == 0:
            return
        pixels = self.to_pixels(points)
        inside = ((pixels[:, 0] >= radius) & (pixels[:, 0] < self.width - radius)
                  & (pixels[:, 1] >= radius) & (pixels[:, 1] < self.height - radius))
        pixels = pixels[inside]
        if len(pixels) == 0:
            return
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                frame[pixels[:, 1] + dy, pixels[:, 0] + dx] = colour

    def draw_trail(self, frame, trail):
        # type: (np.ndarray, list) -> None
        """Draw where the aircraft has flown, brightening toward the present."""
        if trail is None or len(trail) < 2:
            return
        pixels = self.to_pixels(np.asarray(trail[-TRAIL_MAX_POINTS:], np.float32))
        cv2.polylines(frame, [pixels.reshape(-1, 1, 2)], False, COLOUR_TRAIL, 1,
                      cv2.LINE_AA)
        cv2.polylines(frame, [pixels[-120:].reshape(-1, 1, 2)], False,
                      (255, 255, 190), 2, cv2.LINE_AA)

    def draw_aircraft(self, frame, position, yaw):
        # type: (np.ndarray, tuple, float) -> None
        """Draw the aircraft and the wedge it can currently see into."""
        centre = self.to_pixels(np.asarray([position[:2]], np.float32))[0]
        half = math.radians(self.fov_deg) / 2.0
        wedge = [centre]
        for step in range(9):
            angle = yaw - half + (2.0 * half) * step / 8.0
            dx, dy = self._heading_pixels(angle, self.sight_m)
            wedge.append((centre[0] + dx, centre[1] + dy))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(wedge, np.int32)], (70, 120, 70))
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, frame)

        nose = self._heading_pixels(yaw, 1.2)
        left = self._heading_pixels(yaw + math.radians(140.0), 0.8)
        right = self._heading_pixels(yaw - math.radians(140.0), 0.8)
        body = np.array([
            [centre[0] + nose[0], centre[1] + nose[1]],
            [centre[0] + left[0], centre[1] + left[1]],
            [centre[0] + right[0], centre[1] + right[1]],
        ], np.int32)
        cv2.fillPoly(frame, [body], COLOUR_AIRCRAFT)

    def draw_reference(self, frame, position, reference):
        # type: (np.ndarray, tuple, tuple) -> None
        """Draw FALCON's commanded point, and the gap to where the aircraft is.

        The gap is the interesting part. FALCON was validated on a simulator that
        fed its own command back as the aircraft's state, so this line was always
        zero length there; here it is what the outer-loop tracker is fighting.
        """
        if reference is None:
            return
        target = self.to_pixels(np.asarray([reference[:2]], np.float32))[0]
        here = self.to_pixels(np.asarray([position[:2]], np.float32))[0]
        cv2.line(frame, tuple(here), tuple(target), COLOUR_ERROR, 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(target), 4, COLOUR_REFERENCE, -1, cv2.LINE_AA)

    def draw_hud(self, frame, lines, title=""):
        # type: (np.ndarray, list, str) -> None
        """Write the status panel and the compass."""
        panel_h = 26 + 20 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, panel_h), COLOUR_PANEL, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
        cv2.putText(frame, title, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    COLOUR_TEXT, 1, cv2.LINE_AA)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (14, 43 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, COLOUR_TEXT, 1, cv2.LINE_AA)
        self._draw_compass(frame)

    def _draw_compass(self, frame):
        # type: (np.ndarray) -> None
        """A north arrow, because the view may have been turned 90 degrees."""
        origin = (self.width - 46, self.height - 46)
        dx, dy = self._heading_pixels(math.pi / 2.0, 1.0)
        length = math.hypot(dx, dy) or 1.0
        tip = (int(origin[0] + 26 * dx / length), int(origin[1] + 26 * dy / length))
        cv2.arrowedLine(frame, origin, tip, COLOUR_TEXT, 1, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "+y", (tip[0] - 8, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOUR_TEXT, 1, cv2.LINE_AA)

    def draw_scale_bar(self, frame, metres=5.0):
        # type: (np.ndarray, float) -> None
        """A labelled bar, so distances in the video can be read off it."""
        length = int(metres * self.scale)
        if length < 20 or length > self.width // 2:
            return
        y = self.height - 24
        cv2.line(frame, (20, y), (20 + length, y), COLOUR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, "%.0f m" % metres, (20, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, COLOUR_TEXT, 1, cv2.LINE_AA)


def render(canvas, occupied, free, trail, position, yaw, reference, hud, title=""):
    # type: (ExplorationCanvas, np.ndarray, np.ndarray, list, tuple, float, tuple, list, str) -> np.ndarray
    """Compose one complete frame.

    Args:
        canvas: The projection to draw into.
        occupied: ``(N, 2)`` world xy of voxels FALCON believes are solid.
        free: ``(N, 2)`` world xy of voxels it believes are empty.
        trail: World xy the aircraft has visited, oldest first.
        position: The aircraft's current world ``(x, y[, z])``.
        yaw: Its heading, radians CCW from +x.
        reference: FALCON's commanded world ``(x, y[, z])``, or None.
        hud: Status lines for the panel.
        title: Panel headline.

    Returns:
        A BGR frame ready for ``cv2.VideoWriter.write``.
    """
    frame = canvas.blank()
    canvas.draw_cells(frame, free, COLOUR_FREE, radius=1)
    canvas.draw_cells(frame, occupied, COLOUR_OCCUPIED, radius=1)
    canvas.draw_trail(frame, trail)
    canvas.draw_reference(frame, position, reference)
    canvas.draw_aircraft(frame, position, yaw)
    canvas.draw_scale_bar(frame)
    canvas.draw_hud(frame, hud, title)
    return frame
