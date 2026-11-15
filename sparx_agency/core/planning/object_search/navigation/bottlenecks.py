"""Where a route squeezes: the doorways and narrow channels along it, read off the clearance field.

A bottleneck is geometry, not a detection: wherever the route's clearance
(the distance-transform of the confirmed obstacles the planner already
computes) drops below the channel half-width, the body passes between two
walls closer than twice that -- a doorway when the squeeze is short, a
corridor when it is long. The confirmed door landmarks say which of them
the detector saw too, and are reported, never required: a doorway the
detector missed scrapes the body just the same.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle

#: Spacing of the samples the route is read at, metres.
SAMPLE_M = 0.1
#: Two narrow runs closer than this along the route are one channel (a jamb and its frame).
MERGE_GAP_M = 0.3
#: A channel longer than this is a corridor, shorter a doorway.
CORRIDOR_MIN_LENGTH_M = 1.0


@dataclass(frozen=True)
class Channel:
    """One squeeze along a route.

    Attributes:
        kind: ``doorway`` or ``corridor``.
        entry_s_m: Arc length along the route where the clearance first drops below the half-width.
        exit_s_m: Arc length where it rises above it again.
        throat_s_m: Arc length of the tightest point.
        entry_xy: The route point at the entry.
        exit_xy: The route point at the exit.
        throat_xy: The route point at the throat.
        axis_rad: The direction of travel through the throat.
        width_m: Twice the clearance at the throat: the passage width where the route crosses it.
        door_id: The confirmed door landmark at the throat, when one is within reach; None otherwise.
    """

    kind: str
    entry_s_m: float
    exit_s_m: float
    throat_s_m: float
    entry_xy: Tuple[float, float]
    exit_xy: Tuple[float, float]
    throat_xy: Tuple[float, float]
    axis_rad: float
    width_m: float
    door_id: Optional[int] = None

    @property
    def length_m(self) -> float:
        return self.exit_s_m - self.entry_s_m

    def key(self) -> Tuple[int, int]:
        """A stable identity for counting: the throat, to half a metre."""
        return int(round(self.throat_xy[0] / 0.5)), int(round(self.throat_xy[1] / 0.5))

    def as_dict(self) -> dict:
        return {"kind": self.kind, "entry_s_m": round(self.entry_s_m, 2), "exit_s_m": round(self.exit_s_m, 2),
                "throat_xy": [round(self.throat_xy[0], 2), round(self.throat_xy[1], 2)],
                "axis_deg": round(math.degrees(self.axis_rad), 1), "width_m": round(self.width_m, 2),
                "door_id": self.door_id}


def resample_route(points: Sequence[Tuple[float, float]], spacing_m: float = SAMPLE_M):
    """``(xs, ys, s)``: the route at most ``spacing_m`` apart, with the arc length of every sample."""
    xs, ys = [float(points[0][0])], [float(points[0][1])]
    for i in range(len(points) - 1):
        (ax, ay), (bx, by) = points[i], points[i + 1]
        length = math.hypot(bx - ax, by - ay)
        count = max(1, int(math.ceil(length / spacing_m)))
        for j in range(1, count + 1):
            t = j / float(count)
            xs.append(ax + (bx - ax) * t)
            ys.append(ay + (by - ay) * t)
    xs, ys = np.asarray(xs), np.asarray(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return xs, ys, s


def route_channels(points: Sequence[Tuple[float, float]], clearance: np.ndarray, origin_x: float, origin_y: float,
                   resolution: float, half_width_m: float, doors=(), door_match_m: float = 0.6) -> List[Channel]:
    """The channels along ``points``, in route order.

    Args:
        points: The route, world ``(x, y)`` pairs.
        clearance: The distance-to-confirmed-obstacle field, metres.
        origin_x: World x of the grid's cell ``(0, 0)`` corner.
        origin_y: World y of that corner.
        resolution: Metres per cell.
        half_width_m: The clearance below which the route is in a channel.
        doors: Confirmed door landmarks (objects with ``id`` and ``xy``).
        door_match_m: How near a throat a landmark must be to name it.

    Returns:
        The channels, possibly none.
    """
    if len(points) < 2:
        return []
    h, w = clearance.shape
    xs, ys, s = resample_route(points)
    gx = np.floor((xs - origin_x) / resolution).astype(int)
    gy = np.floor((ys - origin_y) / resolution).astype(int)
    inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    c = np.full(len(xs), np.inf)
    c[inside] = clearance[gy[inside], gx[inside]]
    narrow = c < half_width_m
    runs = []
    i = 0
    while i < len(narrow):
        if not narrow[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(narrow) and narrow[j + 1]:
            j += 1
        if runs and s[i] - s[runs[-1][1]] <= MERGE_GAP_M:
            runs[-1] = (runs[-1][0], j)
        else:
            runs.append((i, j))
        i = j + 1
    channels = []
    for a, b in runs:
        throat = a + int(np.argmin(c[a:b + 1]))
        lo, hi = max(0, throat - 2), min(len(xs) - 1, throat + 2)
        axis = math.atan2(ys[hi] - ys[lo], xs[hi] - xs[lo]) if hi > lo else 0.0
        throat_xy = (float(xs[throat]), float(ys[throat]))
        door_id = None
        best = door_match_m
        for door in doors:
            distance = math.dist(throat_xy, tuple(door.xy))
            if distance <= best:
                best, door_id = distance, door.id
        length = float(s[b] - s[a])
        channels.append(Channel(
            kind="corridor" if length > CORRIDOR_MIN_LENGTH_M else "doorway",
            entry_s_m=float(s[a]), exit_s_m=float(s[b]), throat_s_m=float(s[throat]),
            entry_xy=(float(xs[a]), float(ys[a])), exit_xy=(float(xs[b]), float(ys[b])), throat_xy=throat_xy,
            axis_rad=float(axis), width_m=float(2.0 * c[throat]), door_id=door_id))
    return channels


def route_position(points: Sequence[Tuple[float, float]], xy: Tuple[float, float]) -> float:
    """The arc length along the route of the sample nearest ``xy``."""
    xs, ys, s = resample_route(points)
    index = int(np.argmin(np.hypot(xs - xy[0], ys - xy[1])))
    return float(s[index])


def nearest_channel(channels: Sequence[Channel], xy: Tuple[float, float], reach_m: float) -> Optional[Channel]:
    """The channel whose throat is nearest ``xy`` within ``reach_m``, or None."""
    best, best_distance = None, reach_m
    for channel in channels:
        distance = math.dist(xy, channel.throat_xy)
        if distance <= best_distance:
            best, best_distance = channel, distance
    return best


def heading_error(channel: Channel, yaw: float) -> float:
    """The unsigned angle between a heading and the channel's axis of travel, radians."""
    return abs(normalize_angle(yaw - channel.axis_rad))
