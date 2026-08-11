"""Get unstuck, then replan from where you actually are.

This is the controller-agnostic recovery loop the FALCON follower node runs over
whichever controller is flying. It composes two pure primitives -- a
:class:`~.stuck_detector.StuckDetector` and an
:class:`~.escape_maneuver.EscapeManeuver` -- into the exact behaviour the operator
asked for when the drone keeps getting stuck in doorways:

  1. **Notice.** Over a trailing window, a command was issued and the pose did not
     follow -- the drone is stuck against something (often a door frame the camera
     cannot see).
  2. **Recover.** Take the command away from the controller and fly a short,
     open-loop back-out ("exit the wall in the other direction"), so the drone
     breaks contact instead of grinding harder into it.
  3. **Replan from the real position.** Once the back-out is done, the drone is off
     its planned trajectory and the route that led it here is suspect. The
     supervisor raises :attr:`RecoveryDecision.request_replan` once, and the node
     turns that into a blockage report at the drone's REAL pose. The planner then
     marks that spot and replans from where the drone actually is -- not from the
     stale point it should have been at on the trajectory.

Relationship to ``drift_pid``. The ``drift_pid`` controller carries its own,
tighter version of this loop
(:mod:`sparx_agency.core.planning.trackers.drift_pid.blockage` /
:mod:`~.escape`), wired into its confidence model, and only reports to the planner
once the reflexes are fully **spent** on a pinned drone. This supervisor is for
the controllers that have *no* such reflex (``waypoint``, ``multi_axis``,
``pure_pursuit``, ``roll_assist``), and it is deliberately quicker to escalate:
every genuine stuck episode ends in a replan from the recovered position, because
re-centring the route from where the drone actually is -- and teaching the planner
the spot it clipped -- is exactly what stops it clipping the same doorway edge
twice. Run one or the other, never both on the same command.

Python 3.8 compatible: no PEP 604 unions, no ``match``/``case``; stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sparx_agency.core.common.types import Pose2D

from .escape_maneuver import EscapeManeuver, EscapeParams
from .stuck_detector import StuckDetector, StuckParams

#: Recovery states, for telemetry / narration.
RECOVERY_NOMINAL = "NOMINAL"    # nothing wrong, controller owns the command
RECOVERY_MONITOR = "MONITOR"    # stuck confirmed, escape starting or just ended
RECOVERY_ESCAPE = "ESCAPE"      # the back-out owns the command this tick


@dataclass(frozen=True)
class RecoveryParams:
    """Tuning for :class:`RecoverySupervisor`.

    Attributes:
        enabled: False disables the whole loop -- the supervisor is transparent,
            the controller's command always passes through, nothing is reported.
        stuck: Detector tuning (see :class:`~.stuck_detector.StuckParams`).
        escape: Back-out tuning (see :class:`~.escape_maneuver.EscapeParams`). Its
            ``max_attempts`` defaults to 1: one back-out per episode, then replan.
            Raising it makes the drone probe more sides before it escalates to a
            replan; set ``allow_lateral=False`` for a one-axis follower that cannot
            command ``linear.y``.
    """

    enabled: bool = True
    stuck: StuckParams = field(default_factory=StuckParams)
    escape: EscapeParams = field(default_factory=lambda: EscapeParams(max_attempts=1))


@dataclass(frozen=True)
class RecoveryDecision:
    """What the supervisor wants done this tick.

    Attributes:
        override: True means the supervisor OWNS the command this tick -- publish
            ``(vx, vy, wz)`` instead of the controller's output.
        vx: Forward speed to publish while overriding (m/s; negative when backing).
        vy: Lateral speed to publish while overriding (m/s, + left).
        wz: Yaw rate to publish while overriding (rad/s). Always 0.
        request_replan: Edge-triggered True on the single tick a recovery concludes
            -- the node should report the blockage at the real pose so the planner
            reroutes from there. Never True on the same tick as ``override``.
        state: ``NOMINAL`` / ``MONITOR`` / ``ESCAPE``, for telemetry.
        stuck_axis: ``"forward"`` / ``"yaw"`` / ``""`` -- the axis in trouble.
        reason: Short human-readable line for ``/nav/thinking`` (``""`` = nothing
            to narrate this tick).
    """

    override: bool = False
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    request_replan: bool = False
    state: str = RECOVERY_NOMINAL
    stuck_axis: str = ""
    reason: str = ""


class RecoverySupervisor:
    """Detect a stuck drone, back it out, and ask for a replan from the real pose."""

    def __init__(self, params=None):
        # type: (Optional[RecoveryParams]) -> None
        self.params = params or RecoveryParams()
        self._detector = StuckDetector(self.params.stuck)
        self._escape = EscapeManeuver(self.params.escape)
        self.reset()

    def reset(self):
        # type: () -> None
        """Forget every window, streak and in-flight escape."""
        self._detector.reset()
        self._escape.reset()
        self._reported = False
        self._axis = ""

    @property
    def escaping(self):
        # type: () -> bool
        """True while the back-out owns the command."""
        return self._escape.active

    @property
    def stuck(self):
        # type: () -> bool
        """True while a blockage is currently confirmed."""
        return self._detector.verdict.stuck

    def update(self, pose, cmd_vx, cmd_wz, dt, pose_trustworthy=True,
               frozen=False, prefer_left=True):
        # type: (Pose2D, float, float, float, bool, bool, bool) -> RecoveryDecision
        """Advance the recovery loop one control tick.

        Args:
            pose: Current pose in the path frame (the RAW pose -- never a
                latency-led or command-propagated one, which would mask a stuck
                drone).
            cmd_vx: Forward speed PUBLISHED last tick (m/s) -- the command whose
                effect this tick's pose reflects.
            cmd_wz: Yaw rate published last tick (rad/s).
            dt: Seconds since the previous call.
            pose_trustworthy: False when the pose is stale / dead-reckoned and must
                not be used as stuck evidence.
            frozen: True when the drone is deliberately held (no GO, lost
                localization, external hold). No detection, no escape, no report --
                a held drone that "keeps trying" would otherwise invent an obstacle.
            prefer_left: Which way to probe first after backing out (toward the
                side the route continues on, when the caller can tell).

        Returns:
            The :class:`RecoveryDecision` for this tick.
        """
        if not self.params.enabled:
            return RecoveryDecision()

        verdict = self._detector.update(pose, cmd_vx, cmd_wz, dt, pose_trustworthy)
        was_active = self._escape.active

        # ── An escape already owns the command ──
        if was_active:
            if frozen:
                # Never fly an open-loop manoeuvre on a pose that may be cold, or
                # against an external hold. Abandon it; do not treat that as a
                # completed recovery below.
                self._escape.abort()
            else:
                esc = self._escape.step(dt)
                if esc.active:
                    return RecoveryDecision(
                        override=True, vx=esc.vx, vy=esc.vy, wz=esc.wz,
                        state=RECOVERY_ESCAPE, stuck_axis=self._axis,
                        reason=esc.reason)
                # esc.active False -> the manoeuvre finished on this tick; fall
                # through to the conclusion logic.

        # ── Start a new escape ──
        elif verdict.stuck and not frozen:
            if self._escape.trigger(verdict, prefer_left=prefer_left):
                self._axis = verdict.axis
                self._reported = False        # a fresh episode may report again
            # Triggered but not stepped this tick (a deliberate one-tick delay, so
            # the controller gets a final tick before the back-out takes over).
            return RecoveryDecision(state=RECOVERY_MONITOR, stuck_axis=verdict.axis)

        # ── Conclude an episode / report a replan ──
        request = False
        reason = ""
        just_finished = was_active and not self._escape.active
        if just_finished and not frozen and not self._reported:
            # The back-out is done. The drone has moved off its trajectory and the
            # route that led it here is suspect -- report the spot (at the real,
            # recovered pose) so the planner marks it and replans from there.
            self._reported = True
            request = True
            reason = ("Backed out of what I could not get through -- asking the "
                      "planner for a fresh route from where I actually am now")

        if not verdict.stuck and not self._escape.active:
            # Genuinely moving again: give the next obstacle a full set of attempts.
            self._escape.episode_over()

        state = (RECOVERY_MONITOR if (verdict.stuck or self._escape.active)
                 else RECOVERY_NOMINAL)
        return RecoveryDecision(request_replan=request, state=state,
                                stuck_axis=verdict.axis, reason=reason)

