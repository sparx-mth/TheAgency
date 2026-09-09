"""Decide whether a streamed reference is still a plan worth flying.

A tracker fed a *trajectory object* can ask it directly: the curve knows its own
duration, so ``elapsed >= duration`` is the end of the plan and
:mod:`sparx_agency.core.control.reference` answers it that way with its
``past_end`` flag. A tracker fed a *stream of already-evaluated points* -- which
is what ``/planning/pos_cmd`` is -- has no curve to ask, and the two signals it
does have both lie:

* ``trajectory_flag`` is written once by ``traj_server`` at start-up and never
  again, so it reads READY for the life of the process (measured: READY on
  6627 of 6627 recorded samples of one flight);
* the message *stamp* is refreshed at 100 Hz even after the plan has ended,
  because ``traj_server`` republishes the frozen endpoint with a fresh stamp --
  so a staleness test detects a dead publisher and nothing else.

What is left is the reference's own commanded motion. Past the end of its curve
``traj_server`` zeroes the velocity and acceleration, so a reference that has
commanded no motion for longer than a short grace period is a plan that has run
out. That is the signal this module turns into a verdict.

**Motion includes yaw.** An exploration planner aims its sensor, so a plan that
rotates on the spot is perfectly normal and has zero translational velocity for
its whole duration. Judging on translation alone would call every such segment
finished and stop the aircraft turning -- so a reference whose heading is
changing counts as advancing even when it is not going anywhere.

The grace period matters in both directions: a genuinely slow trajectory passes
briefly through zero speed, and the gap between one trajectory ending and the
next arriving is a normal handover rather than a dead plan.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.common.types import normalize_angle

#: The plan is advancing; follow it.
FOLLOW = "follow"

#: The plan has run out. Its last point is the last valid waypoint: fly to it
#: and stop there, rather than treating a dead setpoint as a live one.
ENDPOINT = "endpoint"

#: Nothing usable arrived: no reference at all, or one older than the timeout.
#: Hold where the aircraft is; flying to a guess is worse than not flying.
STALE = "stale"


@dataclass(frozen=True)
class PlanStateParams:
    """When a streamed reference stops counting as a plan.

    Attributes:
        stopped_speed_mps: Commanded reference speed at or below which the plan
            is not advancing. ``traj_server`` zeroes the velocity exactly past a
            curve's end, so this only has to clear arithmetic noise -- it is
            deliberately not a "slow enough to ignore" threshold, which would
            discard the genuinely slow trajectories this airframe flies.
        stopped_yaw_rate_rad_s: Commanded reference yaw rate at or below which
            the heading is not advancing either. Same role as
            ``stopped_speed_mps`` on the rotational axis, and it is what keeps a
            deliberate rotate-on-the-spot segment from being read as a finished
            plan.
        endpoint_grace_s: How long the reference must sit stopped before the
            plan is called finished. Covers the zero-speed instant of a slow
            curve and the normal gap between one trajectory and the next; a
            longer silence is a plan that has ended.
        endpoint_settle_m: How large a station-keeping correction onto a
            finished plan's endpoint may be trusted to point backward, metres.

            An aircraft that overshoots its last waypoint has the endpoint
            BEHIND it, so the hold correction points backward and the
            turn-in-place gate would answer it with a 180 degree spin to
            reclaim centimetres. Within this distance the follower lets that
            correction through instead -- the aircraft occupied that point
            moments ago, so it is known-clear, exactly the argument that exempts
            the escape reflex. It bounds a station-keeping nudge, never travel:
            the position loop's own gain caps the speed at roughly this many
            metres per second.
        endpoint_reach_m: How far away a frozen endpoint may be and still be
            treated as the last valid waypoint to fly to. Beyond this the
            aircraft is not "finishing the path", it is crossing open ground
            toward a point that no live plan has vouched for since, so it holds
            position instead. Sized above the trajectory-to-trajectory handover
            gap and well below the distances a finished plan was measured
            dragging the aircraft across.
    """

    stopped_speed_mps: float = 1e-3
    stopped_yaw_rate_rad_s: float = 1e-3
    endpoint_grace_s: float = 0.5
    endpoint_reach_m: float = 3.0
    endpoint_settle_m: float = 0.35

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the classifier relies on."""
        if self.stopped_speed_mps < 0.0:
            raise ValueError("stopped_speed_mps must be >= 0, got %r"
                             % (self.stopped_speed_mps,))
        if self.stopped_yaw_rate_rad_s < 0.0:
            raise ValueError("stopped_yaw_rate_rad_s must be >= 0, got %r"
                             % (self.stopped_yaw_rate_rad_s,))
        if self.endpoint_grace_s < 0.0:
            raise ValueError("endpoint_grace_s must be >= 0, got %r"
                             % (self.endpoint_grace_s,))
        if self.endpoint_reach_m <= 0.0:
            raise ValueError("endpoint_reach_m must be > 0, got %r"
                             % (self.endpoint_reach_m,))
        if self.endpoint_settle_m < 0.0:
            raise ValueError("endpoint_settle_m must be >= 0, got %r"
                             % (self.endpoint_settle_m,))
        if self.endpoint_settle_m >= self.endpoint_reach_m:
            raise ValueError(
                "endpoint_settle_m (%r) must be below endpoint_reach_m (%r): "
                "a settle band at or beyond the reach would swallow the hold "
                "entirely and the aircraft would never fly to an endpoint."
                % (self.endpoint_settle_m, self.endpoint_reach_m))


class PlanState:
    """Classify a streamed reference as :data:`FOLLOW`, :data:`ENDPOINT` or :data:`STALE`.

    Stateful: the verdict depends on how long the reference has been stopped, so
    it must be stepped once per control tick with that tick's ``dt``.

    Args:
        params: Thresholds. Defaults suit a 100 Hz reference from FALCON's
            ``traj_server``.
    """

    def __init__(self, params=None):
        # type: (PlanStateParams) -> None
        self.params = params or PlanStateParams()
        self._stopped_for_s = 0.0
        self._trajectory_id = None   # type: object
        self._last_yaw = None        # type: object

    def reset(self):
        # type: () -> None
        """Forget the stopped-time accumulator and the trajectory being watched."""
        self._stopped_for_s = 0.0
        self._trajectory_id = None
        self._last_yaw = None

    @property
    def stopped_for_s(self):
        # type: () -> float
        """How long the reference has been commanding zero velocity, seconds."""
        return self._stopped_for_s

    def update(self, reference, reference_age, timeout_s, dt, trajectory_id=None):
        # type: (object, float, float, float, object) -> str
        """Classify this tick's reference.

        Args:
            reference: The planner's state for now, or None.
            reference_age: How long ago it was produced, seconds.
            timeout_s: Age beyond which a reference is stale.
            dt: Seconds since the previous call.
            trajectory_id: The planner's id for the curve this reference came
                from. A change resets the stopped-time accumulator, so a fresh
                trajectory is never inherited as already-finished.

        Returns:
            One of :data:`FOLLOW`, :data:`ENDPOINT`, :data:`STALE`.
        """
        if reference is None or reference_age > timeout_s:
            self._stopped_for_s = 0.0
            self._last_yaw = None
            return STALE

        if trajectory_id is not None and trajectory_id != self._trajectory_id:
            # A new curve starts its own clock. Without this a trajectory that
            # arrives during a freeze is classified finished on its first tick.
            self._trajectory_id = trajectory_id
            self._stopped_for_s = 0.0

        turning = self._turning(reference, dt)
        if _speed(reference) > self.params.stopped_speed_mps or turning:
            self._stopped_for_s = 0.0
            return FOLLOW

        self._stopped_for_s += max(0.0, float(dt))
        if self._stopped_for_s > self.params.endpoint_grace_s:
            return ENDPOINT
        return FOLLOW

    def _turning(self, reference, dt):
        # type: (object, float) -> bool
        """Whether the reference heading is advancing.

        Uses the reference's own yaw rate when it carries one, and otherwise
        differences the commanded heading across ticks -- ``traj_server``'s
        sampled points carry a yaw but the follower does not always populate a
        rate, and a rotate-on-the-spot plan must be recognised either way.
        """
        rate = getattr(reference, "yaw_rate", None)
        yaw = getattr(reference, "yaw", None)
        previous, self._last_yaw = self._last_yaw, yaw
        if rate is not None and abs(float(rate)) > self.params.stopped_yaw_rate_rad_s:
            return True
        if yaw is None or previous is None or dt <= 0.0:
            return False
        delta = abs(normalize_angle(float(yaw) - float(previous)))
        return delta / float(dt) > self.params.stopped_yaw_rate_rad_s


def _speed(reference):
    # type: (object) -> float
    """Magnitude of a reference's commanded velocity, m/s."""
    vx = float(getattr(reference, "vx", 0.0) or 0.0)
    vy = float(getattr(reference, "vy", 0.0) or 0.0)
    vz = float(getattr(reference, "vz", 0.0) or 0.0)
    return (vx * vx + vy * vy + vz * vz) ** 0.5
