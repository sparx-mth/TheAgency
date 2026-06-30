"""Spline-then-Pure-Pursuit follower (ROS-free task adapter).

This is the TASK-LAYER glue that wires two pure CORE algorithms into the same
contract the FALCON node already drives for the one-axis ``WaypointFollower`` and
the ``MultiAxisFollower``:

    raw waypoints --(core HermiteSmoother)--> smooth Trajectory
                  --(core PurePursuitTracker)--> velocity command + lookahead

So the node can select it with ``~controller:=pure_pursuit`` exactly like the
other two trackers, with no change to its control loop. All the maths live in
core (``planning.smoothers.hermite`` and ``planning.trackers.pure_pursuit``);
this module only *composes* them and adapts the I/O:

  * ``set_path`` splines the incoming polyline once (not every tick) and caches
    the time-parameterized trajectory + a sampled polyline for visualization;
  * ``step`` builds a :class:`TrackerRequest`, runs Pure Pursuit and returns a
    command exposing ``vx`` / ``vy`` / ``wz`` (Pure Pursuit is holonomic here, so
    it crabs and yaws toward the lookahead) plus the usual follower bookkeeping;
  * the current **lookahead point** and the **smooth path** are exposed as
    attributes for the node to publish to the BEV viewer.

Altitude is never commanded (the 2D tracker is planar, ``vz`` stays 0). An
optional per-axis minimum-force snap mirrors the platform deadband the real drone
needs (a command below the floor is raised to it or dropped to zero).

ROS-free and unit-tested; the node owns all ROS I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import copysign, hypot
from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import (
    KinematicLimits,
    Path2D,
    Pose2D,
    Pose3D,
    State3D,
)
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker

XY = Tuple[float, float]


class PurePursuitState(str, Enum):
    """Tracker states (mirrors the other followers' ``state`` surface).

    ``IDLE`` — no trajectory yet; holds zero.
    ``RUN``  — tracking the smooth trajectory toward the lookahead.
    ``DONE`` — goal reached (within the tracker's ``goal_tolerance``).
    """

    IDLE = "IDLE"
    RUN = "RUN"
    DONE = "DONE"


@dataclass(frozen=True)
class PurePursuitCommand:
    """Output of one :meth:`PurePursuitFollower.step`.

    The public attribute surface matches the one-axis ``FollowerCommand`` and the
    ``MultiAxisCommand`` (``vx`` / ``vy`` / ``wz`` / ``state`` / ``done`` /
    ``required_axis`` / ``freeze`` / ``wp_idx`` / ``num_waypoints``) so the node
    treats all three trackers uniformly.
    """

    vx: float
    vy: float
    wz: float
    state: PurePursuitState
    done: bool
    wp_idx: int
    num_waypoints: int
    required_axis: Optional[str] = None
    freeze: Optional[bool] = None


def _shape_axis(cmd: float, min_mag: float, release_frac: float,
                zero_eps: float) -> float:
    """Per-axis minimum-force deadband-with-snap (platform deadband).

    A command below ``max(zero_eps, release_frac*min_mag)`` is dropped to zero;
    between there and ``min_mag`` it is snapped up to ``min_mag``; above it passes
    through. With ``min_mag == 0`` only numerical dust below ``zero_eps`` is
    zeroed (the snap is inert), so the feature is opt-in.
    """
    a = abs(cmd)
    drop = max(zero_eps, release_frac * min_mag)
    if a <= drop:
        return 0.0
    if a < min_mag:
        return copysign(min_mag, cmd)
    return cmd


class PurePursuitFollower:
    """Smooth-then-track follower: Hermite spline + Pure Pursuit, follower API."""

    name: str = "pure_pursuit"

    def __init__(
        self,
        tracker: PurePursuitTracker,
        smoother: HermiteSmoother,
        *,
        limits: Optional[KinematicLimits] = None,
        fixed_z: float = 0.0,
        smooth_sample_dt: float = 0.1,
        dedup_eps_m: float = 1e-3,
        min_vx: float = 0.0,
        min_vy: float = 0.0,
        min_wz: float = 0.0,
        min_release_frac: float = 0.5,
        cmd_zero_eps: float = 1e-3,
    ) -> None:
        """
        Args:
            tracker: A core :class:`PurePursuitTracker` (holonomic recommended).
            smoother: A core :class:`HermiteSmoother`.
            limits: Kinematic limits passed to both the spline (timing) and the
                tracker (yaw-rate cap). None lets each use its own defaults.
            fixed_z: Altitude reported to the planar tracker (unused by it; the
                drone holds altitude, so ``vz`` is never commanded).
            smooth_sample_dt: Time step (s) for the cached visualization polyline.
            dedup_eps_m: Consecutive raw waypoints closer than this are merged so
                the spline never sees a degenerate zero-length segment.
            min_vx / min_vy / min_wz: Platform minimum effective command per axis;
                0 disables the snap on that axis.
            min_release_frac: A command below ``min_release_frac*min_*`` is dropped
                to 0 instead of snapped up.
            cmd_zero_eps: Magnitude treated as exactly zero (numerical dust).
        """
        self._tracker = tracker
        self._smoother = smoother
        self._limits = limits
        self._fixed_z = float(fixed_z)
        self._smooth_dt = float(smooth_sample_dt)
        self._dedup_eps = float(dedup_eps_m)
        self._min = (float(min_vx), float(min_vy), float(min_wz))
        self._release = float(min_release_frac)
        self._zero_eps = float(cmd_zero_eps)
        # A forward minimum-force floor that is too large for the tracker's
        # near-goal slow-down would be snapped to zero before the goal-tolerance
        # clean-stop triggers, stalling the drone short of the goal. Forbid it
        # (mirrors the multi-axis arrive_speed_min guard): the slowest forward
        # command the tracker ever emits is its ``min_speed``, so require that to
        # stay above the drop threshold ``min_release_frac * min_vx``.
        pp_min_speed = getattr(tracker.params, "min_speed", None)
        if (pp_min_speed is not None and self._min[0] > 0.0
                and pp_min_speed <= self._release * self._min[0]):
            raise ValueError(
                "min_vx (%.3f) is too large for the tracker min_speed (%.3f): the "
                "near-goal slow-down would be snapped to zero and the drone would "
                "stall short of the goal. Lower min_vx or raise the tracker's "
                "min_speed." % (self._min[0], pp_min_speed))
        # Exposed so the node banner can read tracker tuning like the others.
        self.params = tracker.params
        self.reset()

    # ─── Public API (mirrors WaypointFollower / MultiAxisFollower) ────
    def reset(self) -> None:
        """Clear the trajectory, tracker state and visualization caches."""
        self._traj = None
        self._t = 0.0
        self._state = PurePursuitState.IDLE
        self._done = False
        self._wp_idx = 0
        self.smooth_xy: List[XY] = []      # sampled smooth path (for the BEV viewer)
        self.lookahead: Optional[XY] = None  # current aim point (for the BEV viewer)
        self._tracker.reset()

    @property
    def state(self) -> PurePursuitState:
        return self._state

    @property
    def done(self) -> bool:
        return self._done

    def required_axis(self) -> Optional[str]:
        """Pure Pursuit drives all axes at once; no per-axis handshake."""
        return None

    def set_path(self, waypoints: Sequence[Pose2D], pose: Optional[Pose2D]) -> None:
        """Spline a fresh polyline into a smooth trajectory (once per path).

        ``pose`` is accepted for interface parity (Pure Pursuit re-anchors itself
        each tick via its closest-point search, so it is not needed here).
        """
        pts = self._dedup([(float(p.x), float(p.y)) for p in waypoints])
        if len(pts) < 2:
            self.reset()
            return
        path = Path2D(points=tuple(Pose2D(x, y) for x, y in pts))
        try:
            # Build into locals first; a too-short/degenerate path (e.g. a replan
            # whose endpoints are within a sample length) makes the smoother yield
            # a single point and DiscreteTrajectory raise -- treat that like the
            # len<2 case (drop to IDLE / hold) instead of crashing the callback and
            # silently flying the stale trajectory.
            traj = self._smoother.smooth(SmootherRequest(path=path, limits=self._limits))
            smooth_xy = [(p.x, p.y) for p in traj.sample_by_time(self._smooth_dt)]
        except ValueError:
            self.reset()
            return
        self._traj = traj
        self.smooth_xy = smooth_xy
        self._tracker.reset()
        self._t = 0.0
        self._state = PurePursuitState.RUN
        self._done = False
        self._wp_idx = 0
        self.lookahead = None

    def step(
        self,
        pose: Pose2D,
        dt: float,
        *,
        axis_confirmed: bool = True,
        hold: bool = False,
        map_ready: bool = True,
    ) -> PurePursuitCommand:
        """Advance one tick: run Pure Pursuit on the smooth trajectory.

        Args mirror the other followers; ``map_ready`` is accepted for parity and
        unused (this tracker never stops to wait for a map update). While ``hold``
        or an unconfirmed axis or no trajectory, it holds zero.
        """
        del map_ready  # interface parity only
        if hold or not axis_confirmed or self._traj is None:
            return self._command(0.0, 0.0, 0.0)

        self._t += max(0.0, dt)
        req = TrackerRequest(
            state=State3D(pose=Pose3D(pose.x, pose.y, self._fixed_z, pose.yaw)),
            trajectory=self._traj,
            t=self._t,
            limits=self._limits,
        )
        res = self._tracker.step(req)
        cmd = res.command
        self._done = bool(res.metadata.get("done"))
        self._wp_idx = int(res.metadata.get("progress_idx", self._wp_idx))
        self._state = PurePursuitState.DONE if self._done else PurePursuitState.RUN
        if res.reference is not None:
            self.lookahead = (float(res.reference.x), float(res.reference.y))
        return self._command(cmd.x, cmd.y, cmd.yaw_rate)

    # ─── Helpers ─────────────────────────────────────────────────────
    def _command(self, vx: float, vy: float, wz: float) -> PurePursuitCommand:
        vx = _shape_axis(vx, self._min[0], self._release, self._zero_eps)
        vy = _shape_axis(vy, self._min[1], self._release, self._zero_eps)
        wz = _shape_axis(wz, self._min[2], self._release, self._zero_eps)
        return PurePursuitCommand(
            vx=vx, vy=vy, wz=wz, state=self._state, done=self._done,
            wp_idx=self._wp_idx, num_waypoints=len(self.smooth_xy))

    def _dedup(self, pts: List[XY]) -> List[XY]:
        """Drop consecutive near-duplicate points (degenerate spline segments)."""
        out: List[XY] = []
        for p in pts:
            if not out or hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > self._dedup_eps:
                out.append(p)
        return out
