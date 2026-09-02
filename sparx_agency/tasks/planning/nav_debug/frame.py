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

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

XY = Tuple[float, float]
XYZ = Tuple[float, float, float]


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

    On the Sphera exploration path there is no A* chain: ``final`` carries FALCON's
    own ``/falcon/planned_path`` and ``executed`` the path it has actually flown, so
    plan and outcome can be compared on one map.
    """

    astar: Optional[List[XY]] = None
    safe: Optional[List[XY]] = None
    final: Optional[List[XY]] = None
    executed: Optional[List[XY]] = None
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


# ─────────────────────────────────────────────────────────────────────────────
# The Sphera/FALCON exploration lanes.
#
# The XTEND dataclasses above describe the A*/click-to-fly stack: a route of
# waypoints, a pure-pursuit lookahead and a drift-PID controller. The Sphera
# stack flies a different loop -- FALCON emits a B-spline, a 100 Hz reference
# streams off it, and a reference tracker plus a per-axis velocity servo turn
# that into stick counts -- so the "what is it doing and why" of that loop needs
# its own vocabulary. Every one of these is optional: a run recorded on either
# stack renders what it has.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reference:
    """The setpoint the aircraft is chasing *right now* (``/planning/pos_cmd``).

    This is the Sphera analogue of the XTEND lookahead point, and it is a far
    richer thing: FALCON's ``traj_server`` evaluates the committed B-spline at
    the current instant and emits position, velocity, acceleration and heading
    at 100 Hz. ``age_s`` matters as much as the value -- the tracker holds
    station instead of chasing a reference older than its timeout, and a stale
    reference is the documented cause of the aircraft standing still.
    """

    x: float
    y: float
    z: float
    yaw: Optional[float] = None
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_dot: float = 0.0
    age_s: float = 0.0
    traj_id: Optional[int] = None
    # traj_server sets this false at a trajectory's end, where it republishes the
    # frozen endpoint with fresh stamps -- so "fresh" does not imply "moving".
    moving: bool = False

    @property
    def speed(self) -> float:
        """Reference ground speed, m/s."""
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class Tracking:
    """The tracker's verdict on whether the aircraft is flying the plan.

    ``ReferenceTracker3D`` computes all of these every tick and the flight node
    keeps only ``position_error_m`` for a 0.5 Hz log line; the rest were being
    discarded. ``along_track_lag_m`` and ``cross_track_error_m`` are separated
    deliberately: lag means *late*, cross-track means *somewhere else*, and only
    the second one flies into walls.
    """

    position_error_m: float = 0.0
    along_track_lag_m: float = 0.0
    cross_track_error_m: float = 0.0
    yaw_error_rad: float = 0.0
    diverged: bool = False
    holding: bool = False
    reference_age_s: float = 0.0


@dataclass(frozen=True)
class ControlTerms:
    """The commanded velocity broken into the terms that produced it.

    The tracker's output is ``feed_forward + damping + correction``, then
    clamped and smoothed. Recording only the sum makes an over-aggressive gain
    indistinguishable from a large reference velocity. Each entry is a world
    ``(x, y, z)`` triple in m/s; ``clamped``/``smoothed`` are the same command
    after the envelope and the rate limiter, so a tick where the limiter -- not
    the controller -- chose the command is visible as a difference.
    """

    feed_forward: Optional[XYZ] = None
    damping: Optional[XYZ] = None
    correction: Optional[XYZ] = None
    commanded: Optional[XYZ] = None     # the raw sum, before the envelope
    clamped: Optional[XYZ] = None
    smoothed: Optional[XYZ] = None      # what actually left the tracker
    limits: Tuple[str, ...] = ()        # names of the limits that bound this tick


@dataclass(frozen=True)
class AxisTrace:
    """One actuator axis, from requested speed to the stick count sent.

    The Rooster does not take a velocity. ``rooster_twist_control_adapter``
    turns each requested m/s into counts through a measured expo curve
    (the feed-forward), then closes a PI servo on Sphera's own velocity, then
    slew-limits and clamps the result. Four separate stages can each be the
    reason the drone is not moving as asked, and none of them was observable.
    """

    name: str = ""                      # forward | lateral | yaw
    requested: float = 0.0              # m/s (or rad/s for yaw)
    measured: float = 0.0               # achieved, from /R1/velocity_truth
    error: float = 0.0                  # requested - measured
    feed_forward: float = 0.0           # counts, straight off the measured curve
    integral: float = 0.0               # servo integral state
    correction: float = 0.0             # counts added by the servo
    pre_slew: float = 0.0               # counts before the rate limiter
    counts: float = 0.0                 # counts actually sent
    saturated: bool = False             # the servo correction hit its ceiling
    slew_limited: bool = False          # the rate limiter, not the servo, chose this
    capped: bool = False                # the per-axis count ceiling bound
    feedback_stale: bool = False        # the servo ran without fresh truth


@dataclass(frozen=True)
class Actuator:
    """What the drone was actually told: the last hop before the airframe.

    ``cmd_nav`` is the adapter's request; ``manual`` is the
    ``fcu_driver_interfaces/ManualControl`` the command unit really published,
    which is the only message Sphera's physics acts on. They differ whenever the
    altitude loop writes the throttle axis or a second publisher injects a
    command, which is exactly the failure this lane exists to catch.
    """

    cmd_nav: Optional[Tuple[float, float, float]] = None   # x, y, r as requested
    manual: Optional[Tuple[float, float, float, float]] = None  # x, y, z, r sent
    buttons: int = 0
    cmd_nav_age_s: float = 0.0
    manual_age_s: float = 0.0


@dataclass(frozen=True)
class Altitude:
    """The vertical lane, which no single process owns.

    The planner's ``vz`` becomes bounded nudges of the command unit's own
    terrain-relative setpoint; a PD loop against the rangefinder then owns the
    throttle. Three processes across two ROS versions, and the only record was a
    printf. ``guard_rejected`` is the rangefinder plausibility gate, which
    returns silently -- a gate firing every tick looks identical to a healthy
    hold.
    """

    target_m: Optional[float] = None       # the hold setpoint, metres above ground
    ranger_m: Optional[float] = None       # measured rangefinder distance
    error_m: Optional[float] = None
    wanted_z: Optional[float] = None       # counts the hold loop asked for
    sent_z: Optional[float] = None         # counts actually put on the stick
    nudge_m: float = 0.0                   # setpoint change the planner requested
    at_ceiling: bool = False
    guard_rejected: bool = False           # the ranger-rate plausibility gate fired
    guard_rejects_total: int = 0
    # Why the tick ended where it did: held (the loop ran), not_holding,
    # no_ranger, no_new_sample, guard_rejected. On every reason but ``held`` the
    # error/wanted_z/sent_z above are None because the loop never computed them
    # -- without this the panel reads like a healthy hold at zero error.
    reason: str = ""


@dataclass(frozen=True)
class Truth:
    """Sphera's ground truth plus vehicle state -- the only honest yardstick.

    PX4's own estimate is not authoritative here: Sphera's physics runs off the
    vendor ManualControl pipeline, so ``/R1/sphera/state`` and the velocity
    derived from it are what "the drone actually did" means. Roll/pitch come
    from ``/R1/attitude_rpy`` because the pose the whole stack consumes is
    yaw-only by contract.
    """

    vx: Optional[float] = None             # world m/s, from /R1/velocity_truth
    vy: Optional[float] = None
    vz: Optional[float] = None
    roll: Optional[float] = None           # rad, from /R1/attitude_rpy
    pitch: Optional[float] = None
    battery_pct: Optional[float] = None
    armed: Optional[bool] = None
    flight_mode: str = ""
    status: str = ""

    @property
    def speed(self) -> Optional[float]:
        """Achieved ground speed, m/s, or None if truth was not recorded."""
        if self.vx is None or self.vy is None:
            return None
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class MapStats:
    """Per-update map-quality numbers, so a bad plan can be blamed on a bad map.

    The map FALCON plans on and the BEV drawn here are two different structures
    at two different resolutions; these counters are what
    ``mapping_sync``/``bev_publisher`` already compute and throw away.

    Every counter defaults to ``None`` rather than ``0``: several have no
    recorder yet, and a map panel reading "0 occupied, 0 free" is exactly what a
    catastrophically empty grid looks like. Absent must not imitate measured.
    """

    depth_frames: Optional[int] = None
    emitted: Optional[int] = None
    dropped: Optional[int] = None
    drop_reason: str = ""
    gate_state: str = ""                   # e.g. frozen while turning
    occupied_cells: Optional[int] = None
    free_cells: Optional[int] = None
    unknown_cells: Optional[int] = None
    outside_bbox_frac: Optional[float] = None
    pose_age_s: Optional[float] = None
    depth_age_s: Optional[float] = None
    tilt_deg: Optional[float] = None


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

    # ── Sphera/FALCON exploration lanes (all optional; None on an XTEND run) ──
    reference: Optional[Reference] = None
    tracking: Optional[Tracking] = None
    terms: Optional[ControlTerms] = None
    axes: List[AxisTrace] = field(default_factory=list)
    actuator: Optional[Actuator] = None
    altitude: Optional[Altitude] = None
    truth: Optional[Truth] = None
    map_stats: Optional[MapStats] = None
    # Trailing series for the new strips (oldest first).
    err_history: List[float] = field(default_factory=list)     # position error, m
    speed_history: List[float] = field(default_factory=list)   # achieved speed, m/s
