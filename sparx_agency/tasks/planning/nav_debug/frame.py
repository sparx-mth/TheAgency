"""Data contracts for one navigation-debug moment.

:class:`NavFrame` is the nav analogue of the object-approach ``FrameResult``:
everything the renderer needs to draw one instant, already resolved from the run
folder + certainty CSV, so :mod:`.render` does no lookups and stays pure. All
fields are optional and default to ``None`` so a partial recording (e.g. no CSV,
so no drift/quality) still renders what it has instead of raising.

Poses/velocities are in the core body-frame convention (REP-103: ``+vx`` forward,
``+vy`` left, ``+wz`` CCW). Axis counts are the XTEND virtual-controller integers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

XY = Tuple[float, float]


@dataclass(frozen=True)
class GaugeScales:
    """Full-scale references for the eight command gauges.

    Defaults are the XTEND envelope (forward/lateral/vertical top out at 600
    counts ~= 0.45 m/s, yaw at 1000 counts ~= 0.65 rad/s); the CLI overrides them
    from :data:`XTEND_CALIBRATION` so the gauges track the live calibration. Kept
    here (not imported from ``robots``) so the renderer stays drone-agnostic.
    """

    our_vx: float = 0.45
    our_vy: float = 0.45
    our_vz: float = 0.45
    our_wz: float = 0.65
    drone_forward: float = 600.0
    drone_lateral: float = 600.0
    drone_vertical: float = 600.0
    drone_yaw: float = 1000.0


@dataclass(eq=False)
class BevMap:
    """A 2D occupancy grid snapshot with the geometry to place world points on it.

    ``grid`` is int8 HxW on the ROS convention (free=0, occupied=100,
    unknown=-1), row-major from ``(origin_x, origin_y)`` with ``+x`` along columns
    and ``+y`` along rows.
    """

    grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str = "world"
    stamp: float = 0.0

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_cell(self, x: float, y: float) -> Tuple[float, float]:
        """World ``(x, y)`` -> fractional ``(col, row)`` cell (no image flip)."""
        return ((x - self.origin_x) / self.resolution,
                (y - self.origin_y) / self.resolution)


@dataclass(eq=False)
class Routes:
    """The three route layers plus the goal and the current aim point, in world m.

    ``astar`` is the raw plan, ``safe`` the BEV-corrected (pushed off walls) plan,
    ``final`` the simplified route the follower actually flies. Any may be ``None``
    if that topic never arrived.
    """

    astar: Optional[List[XY]] = None
    safe: Optional[List[XY]] = None
    final: Optional[List[XY]] = None
    goal: Optional[XY] = None
    lookahead: Optional[XY] = None


@dataclass(frozen=True)
class ReplanEvent:
    """A replan/blockage event and how long before this frame it fired.

    ``kind`` is a coarse bucket (``time`` / ``rotation`` / ``obstacle`` /
    ``blockage`` / ``boxed_in`` / ``info``) derived from the raw ``text`` so the
    banner can colour it; ``text`` is the planner's own string verbatim.
    """

    stamp: float
    kind: str
    text: str
    age_s: float = 0.0
    xy: Optional[XY] = None


@dataclass(frozen=True)
class Quality:
    """AprilTag localization quality this tick (from the certainty CSV)."""

    confidence: float
    pos_std_m: float
    cmd_effectiveness: float
    coasting: bool
    age_s: float
    source: str = ""


@dataclass(frozen=True)
class Drift:
    """What the drift-PID controller learned/was doing this tick (certainty CSV)."""

    drift_vx: float
    drift_vy: float
    drift_wz: float
    cross_track_m: float
    along_track_m: float
    heading_err_deg: float
    effort: float
    speed_scale: float
    authority: str
    state: str
    escape_state: str
    blocked_axis: str


@dataclass(eq=False)
class NavFrame:
    """Everything :func:`nav_debug.render.render` needs to draw one instant."""

    stamp: float
    # Pose (world): x, y, z (m) and yaw (rad). z may be None if unknown.
    x: float
    y: float
    yaw: float
    z: Optional[float] = None
    trail: List[XY] = field(default_factory=list)          # recent poses, oldest first

    # Command WE send (cmd_vel, body-frame SI) and what the converter sends the
    # DRONE (cmd_nav axis counts). Either may be None if not recorded.
    our_cmd: Optional[Tuple[float, float, float, float]] = None   # vx, vy, vz, wz
    drone_cmd: Optional[Tuple[int, int, int, int]] = None         # fwd, lat, vert, yaw

    quality: Optional[Quality] = None
    drift: Optional[Drift] = None
    # Target waypoint: index, count, world xy.
    target: Optional[Tuple[int, int, float, float]] = None
    advanced: bool = False        # True on the tick the active waypoint index grew

    bev: Optional[BevMap] = None
    bev_conf: Optional[np.ndarray] = None
    routes: Routes = field(default_factory=Routes)
    replan: Optional[ReplanEvent] = None

    # Short trailing series for the panel strips (oldest first).
    cmd_history: List[float] = field(default_factory=list)        # our vx
    conf_history: List[float] = field(default_factory=list)       # confidence

    why: str = ""
