"""Fly one route with one controller configuration, and score what happened.

The scoring is where the honesty of this rig lives, so it is worth saying what
each number is for:

* ``seconds`` — wall time to the goal. The anticipation is expected to *cost*
  some of this at every corner: a crab is capped by the weak lateral axis, so
  the last stretch into a corner is flown slower than a cruise. Quoting it first
  is deliberate.
* ``turn_s`` — time in the controller's TURN regime: pointed the wrong way,
  translation suppressed, station-keeping while the nose comes round. This is
  the manoeuvre the whole feature exists to delete.
* ``spin_s`` — the worst of that: rotating with no translation under it at all,
  which this airframe is barely able to do (~11% yaw delivery) and where it
  drifts most. The deployed tuning already softens it with ``turn_pitch_bias``,
  so expect a small number even on the classic run.
* ``stopped_s`` — time commanding no translation at all.
* ``worst_xtrack`` — how far off its line the drone ever got. The nose is
  allowed to leave the leg; the body is not, and this is the number that says
  whether the crab held it.
* ``arrive_err`` — how far from the goal it finished.
* ``peak_lead`` — how far round the nose ever led. 0 on a run with the feature
  off, and the signature of the manoeuvre when it is on.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, hypot
from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.drift_pid import (
    DriftPidFollower, DriftPidParams, DriftPidState, LocalizationQuality,
)

from .airframe import Airframe, AirframeParams

#: Control period (s). 10 Hz, matching ~ctrl_rate_hz on the drone.
DT = 0.1

#: Yaw command above which a tick counts as "rotating" for the spin metric.
_SPINNING_WZ = 0.05

#: Translation below which a tick counts as "not moving" (m/s).
_STOPPED_V = 0.02


@dataclass(frozen=True)
class FlightResult:
    """What one flight did. See the module docstring for what each means."""

    reached: bool
    seconds: float
    escape_s: float
    spin_s: float
    stopped_s: float
    worst_xtrack_m: float
    arrive_err_m: float
    peak_lead_deg: float
    turn_ticks: int
    track: Tuple[Tuple[float, float, float], ...]      # x, y, yaw
    leads: Tuple[float, ...]                           # deg, per tick


def _healthy():
    """A good AprilTag fix: the rig is not testing localization."""
    return LocalizationQuality(confidence=0.5, pos_std_m=0.02, age_s=0.05,
                               coasting=False, cmd_effectiveness=1.0,
                               valid=True)


def fly(waypoints, params, airframe_params=None, max_seconds=240.0):
    # type: (Sequence[Pose2D], DriftPidParams, Optional[AirframeParams], float) -> FlightResult
    """Fly ``waypoints`` to the end and report what the flight looked like.

    Args:
        waypoints: The route, at least two points.
        params: Controller tuning — the same object the ROS adapter builds.
        airframe_params: The modelled drone. Defaults to the measured coupling.
        max_seconds: Give up after this long, and say so via ``reached``.

    Returns:
        The flight's score and its track.
    """
    follower = DriftPidFollower(params)
    start = waypoints[0]
    heading = 0.0
    if len(waypoints) >= 2:
        from math import atan2
        heading = atan2(waypoints[1].y - start.y, waypoints[1].x - start.x)
    drone = Airframe(airframe_params or AirframeParams(),
                     Pose2D(start.x, start.y, heading))
    follower.set_path(list(waypoints), drone.pose)

    ticks = int(max_seconds / DT)
    spin = stopped = turning = escaping = 0
    worst = 0.0
    peak_lead = 0.0
    track = []                     # type: List[Tuple[float, float, float]]
    leads = []                     # type: List[float]
    used = ticks
    for i in range(ticks):
        follower.set_quality(_healthy())
        cmd = follower.step(drone.pose, DT)
        moving = hypot(cmd.vx, cmd.vy)
        if abs(cmd.wz) > _SPINNING_WZ and moving < _STOPPED_V:
            spin += 1
        if moving < _STOPPED_V:
            stopped += 1
        turning += int(cmd.state == DriftPidState.TURN)
        escaping += int(cmd.state == DriftPidState.ESCAPE)
        # Skip the first second: the drone starts on the line but the loops have
        # not settled, and a start transient is not a corner.
        if i > 10:
            worst = max(worst, abs(cmd.telemetry.cross_track_m))
        lead = degrees(cmd.telemetry.yaw_lead_rad)
        leads.append(lead)
        peak_lead = max(peak_lead, abs(lead))
        track.append((drone.pose.x, drone.pose.y, drone.pose.yaw))
        drone.step(cmd.vx, cmd.vy, cmd.wz, DT)
        if cmd.done:
            used = i + 1
            break
    goal = waypoints[-1]
    return FlightResult(
        reached=follower.done, seconds=used * DT, escape_s=escaping * DT,
        spin_s=spin * DT, stopped_s=stopped * DT, worst_xtrack_m=worst,
        arrive_err_m=hypot(drone.pose.x - goal.x, drone.pose.y - goal.y),
        peak_lead_deg=peak_lead, turn_ticks=turning,
        track=tuple(track), leads=tuple(leads))
