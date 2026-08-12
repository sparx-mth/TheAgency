"""Detecting that the drone is stuck: told to move, and not moving.

Indoors, close to a wall or in a doorway, an aircraft has a failure mode the map
cannot see: it is commanded to go and does not go. Sometimes the obstacle is
genuinely invisible to the camera (glass, a thin pole, a door frame below the
height band); sometimes it is visible but the airframe is already touching it and
pinned. Either way the symptom is the same and it is *observable*: a command is
being issued and the pose is not changing.

This is the controller-agnostic sibling of
:class:`sparx_agency.core.planning.trackers.drift_pid.blockage.BlockageMonitor`.
``drift_pid`` carries its own detector wired into its confidence model
(``LocalizationQuality``, ``cmd_effectiveness``); this one takes the same idea but
only a plain ``pose_trustworthy`` flag, so *any* follower node can run it over
whichever controller is flying (``waypoint``, ``multi_axis``, ``pure_pursuit``,
``roll_assist``). The judgement is identical:

  Over a trailing window, the distance the pose actually travelled *along the
  direction that was commanded*, compared with the distance that was commanded.
  Projecting rather than taking the raw displacement matters: a drone shoved
  sideways by drift while its forward axis is blocked would otherwise look like it
  is moving fine.

Translation and yaw are tracked separately and deliberately, because they fail
separately: a drone wedged against a wall can usually still rotate, and a drone
that cannot rotate (its side pinned) can usually still back off.

**An untrustworthy pose is never used as evidence.** When the localization is
stale or dead-reckoned it is often propagated *by the commanded motion*, so it
always appears to obey commands perfectly; feeding that into a stuck detector
would guarantee it never fires exactly when the drone is most likely stuck. The
caller passes ``pose_trustworthy=False`` for those ticks and the window is dropped
rather than poisoned.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8): no
PEP 604 unions, no ``match``/``case``, no builtin generics at runtime; stdlib +
``Pose2D`` only, no numpy, no scipy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, sin
from typing import Optional

from sparx_agency.core.common.types import Pose2D, normalize_angle

#: Axis names a :class:`StuckVerdict` can name.
AXIS_NONE = ""
AXIS_FORWARD = "forward"
AXIS_YAW = "yaw"


@dataclass(frozen=True)
class StuckParams:
    """Tuning for :class:`StuckDetector`.

    Attributes:
        enabled: False disables the detector outright -- it never confirms, so no
            escape ever runs and nothing is reported to the planner. The kill
            switch for a platform that honestly under-delivers on every axis (weak
            battery, payload, trim), where the detector would read weakness as
            walls and the escapes would consume the flight.
        window_s: Length of the trailing window progress is measured over (s).
            Long enough to out-live pose noise, short enough to react before the
            drone has been grinding against a wall for a while.
        min_cmd_vx: Forward speed below which the tick does not count as "trying
            to translate" (m/s). Keep it above the platform's minimum-force floor,
            or ticks the motors ignored would count as failed attempts.
        min_cmd_wz: Yaw rate below which the tick does not count as "trying to
            rotate" (rad/s).
        min_cmd_distance_m: Commanded distance that must accumulate inside the
            window before the ratio means anything (m). Guards against declaring a
            blockage from a couple of ticks of rounding.
        min_cmd_yaw_rad: Same guard for the yaw axis (rad).
        progress_frac: Achieved/commanded ratio below which the axis counts as
            making no progress (0..1). Not zero: the calibration is approximate and
            a heavy drone always under-delivers, so ~0.3 means "barely moved", not
            "moved slightly less than asked".
        confirm_ticks: Consecutive bad ticks needed to declare a blockage. A tick
            count, never a wall-clock timer: the supervisor reacts to evidence, not
            to a clock that keeps running while the drone holds still.
        clear_ticks: Consecutive good ticks needed to clear one.
        stale_clear_s: Seconds of NOT pushing an axis after which a standing
            blockage on it is dropped as stale (0 disables). Once the follower has
            stopped driving into a blockage -- rerouted away, boxed in, or held --
            the claim "pushing here does nothing" is no longer under test, and left
            latched it would freeze the drone against a spot nothing is re-probing.
            Must comfortably exceed the idle stretch of an escape manoeuvre (its
            brake/back/settle phases), so a live escape cannot clear the very
            blockage it is escaping.
    """

    enabled: bool = True
    window_s: float = 1.2
    min_cmd_vx: float = 0.05
    min_cmd_wz: float = 0.15
    min_cmd_distance_m: float = 0.06
    min_cmd_yaw_rad: float = 0.12
    progress_frac: float = 0.30
    confirm_ticks: int = 5
    clear_ticks: int = 3
    stale_clear_s: float = 4.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the detector relies on."""
        for name in ("window_s", "min_cmd_vx", "min_cmd_wz",
                     "min_cmd_distance_m", "min_cmd_yaw_rad"):
            if getattr(self, name) <= 0.0:
                raise ValueError("StuckParams." + name + " must be > 0")
        if not 0.0 < self.progress_frac < 1.0:
            raise ValueError("StuckParams.progress_frac must be in (0, 1)")
        for name in ("confirm_ticks", "clear_ticks"):
            if getattr(self, name) < 1:
                raise ValueError("StuckParams." + name + " must be >= 1")
        if self.stale_clear_s < 0.0:
            raise ValueError("StuckParams.stale_clear_s must be >= 0")


@dataclass(frozen=True)
class StuckVerdict:
    """What the detector believes about the drone's ability to move.

    Attributes:
        axis: ``""`` when nothing is blocked, else ``"forward"`` or ``"yaw"``.
        sign: Sign of the command that is not getting through (+1/-1); 0 when
            nothing is blocked. For ``forward`` this is +1 for a forward command;
            for ``yaw``, +1 for a left (CCW) turn.
        ratio: The achieved/commanded ratio that produced the verdict.
    """

    axis: str = AXIS_NONE
    sign: int = 0
    ratio: float = 1.0

    @property
    def stuck(self):
        # type: () -> bool
        """True when any axis is confirmed blocked."""
        return self.axis != AXIS_NONE


class StuckDetector:
    """Watches commanded vs achieved motion and reports a confirmed blockage."""

    def __init__(self, params=None):
        # type: (Optional[StuckParams]) -> None
        self.params = params or StuckParams()
        self.reset()

    def reset(self):
        # type: () -> None
        """Forget the window and every confirmation streak."""
        self._samples = deque()   # (elapsed, x, y, yaw, cum_dist, cum_yaw)
        self._t = 0.0
        self._cum_dist = 0.0
        self._cum_yaw = 0.0
        self._bad_fwd = 0
        self._bad_yaw = 0
        self._good_fwd = 0
        self._good_yaw = 0
        self._idle_fwd_s = 0.0    # seconds the forward axis has gone un-pushed
        self._idle_yaw_s = 0.0    # seconds the yaw axis has gone un-pushed
        self._verdict = StuckVerdict()

    @property
    def verdict(self):
        # type: () -> StuckVerdict
        """The current (debounced) verdict, without advancing the detector."""
        return self._verdict

    def update(self, pose, cmd_vx, cmd_wz, dt, pose_trustworthy=True):
        # type: (Pose2D, float, float, float, bool) -> StuckVerdict
        """Advance the detector one tick and return the current verdict.

        Args:
            pose: Pose this tick, in the path frame.
            cmd_vx: Forward speed commanded on the PREVIOUS tick (m/s) -- that is
                the command whose effect this tick's pose reflects.
            cmd_wz: Yaw rate commanded on the previous tick (rad/s).
            dt: Seconds since the previous call.
            pose_trustworthy: False when the pose is stale or dead-reckoned and
                cannot be used as evidence (it would be propagated by the command
                and always seem obedient). The window is dropped for that tick.

        Returns:
            The debounced :class:`StuckVerdict`.
        """
        if dt <= 0.0:
            raise ValueError("StuckDetector.update: dt must be > 0")
        p = self.params
        if not p.enabled:
            # Killed by config: never confirm, and drop any standing verdict so a
            # blockage latched before the switch flipped cannot linger.
            self._verdict = StuckVerdict()
            return self._verdict

        # A stale / dead-reckoned pose is often propagated BY the command, so it
        # can only ever agree with it. Drop the window rather than poison it.
        if not pose_trustworthy:
            self._samples.clear()
            self._cum_dist = 0.0
            self._cum_yaw = 0.0
            return self._verdict

        self._t += dt
        self._cum_dist += abs(cmd_vx) * dt
        self._cum_yaw += abs(cmd_wz) * dt
        self._samples.append((self._t, pose.x, pose.y, pose.yaw,
                              self._cum_dist, self._cum_yaw))
        while len(self._samples) > 1 and self._t - self._samples[0][0] > p.window_s:
            self._samples.popleft()
        if len(self._samples) < 2:
            return self._verdict

        t0, x0, y0, yaw0, d0, a0 = self._samples[0]
        want_dist = self._cum_dist - d0
        want_yaw = self._cum_yaw - a0

        # -- forward axis --
        fwd_ratio = 1.0
        if abs(cmd_vx) >= p.min_cmd_vx and want_dist >= p.min_cmd_distance_m:
            self._idle_fwd_s = 0.0
            # Progress along the direction that was commanded at window start, not
            # raw displacement: sideways drift is not forward progress.
            moved = ((pose.x - x0) * cos(yaw0) + (pose.y - y0) * sin(yaw0))
            if cmd_vx < 0.0:
                moved = -moved
            fwd_ratio = max(0.0, moved) / want_dist
            if fwd_ratio < p.progress_frac:
                self._bad_fwd += 1
                self._good_fwd = 0
            else:
                self._good_fwd += 1
                self._bad_fwd = 0
        else:
            # Not pushing this axis. Break the bad streak -- confirmation needs
            # CONSECUTIVE bad ticks and an un-pushed tick is not evidence -- and
            # start timing the idle so a standing verdict can go stale.
            self._bad_fwd = 0
            self._idle_fwd_s += dt

        # -- yaw axis --
        yaw_ratio = 1.0
        if abs(cmd_wz) >= p.min_cmd_wz and want_yaw >= p.min_cmd_yaw_rad:
            self._idle_yaw_s = 0.0
            turned = normalize_angle(pose.yaw - yaw0)
            if cmd_wz < 0.0:
                turned = -turned
            yaw_ratio = max(0.0, turned) / want_yaw
            if yaw_ratio < p.progress_frac:
                self._bad_yaw += 1
                self._good_yaw = 0
            else:
                self._good_yaw += 1
                self._bad_yaw = 0
        else:
            self._bad_yaw = 0
            self._idle_yaw_s += dt

        return self._decide(cmd_vx, cmd_wz, fwd_ratio, yaw_ratio)

    def _decide(self, cmd_vx, cmd_wz, fwd_ratio, yaw_ratio):
        # type: (float, float, float, float) -> StuckVerdict
        """Apply the confirm/clear streaks and update the standing verdict.

        Forward wins a tie: it is both the more common blockage and the more
        dangerous one to keep pushing on.
        """
        p = self.params
        if self._bad_fwd >= p.confirm_ticks:
            self._verdict = StuckVerdict(AXIS_FORWARD, 1 if cmd_vx >= 0.0 else -1,
                                         fwd_ratio)
        elif self._bad_yaw >= p.confirm_ticks:
            self._verdict = StuckVerdict(AXIS_YAW, 1 if cmd_wz >= 0.0 else -1,
                                         yaw_ratio)
        elif self._verdict.stuck:
            if self._verdict.axis == AXIS_FORWARD:
                good, idle = self._good_fwd, self._idle_fwd_s
            else:
                good, idle = self._good_yaw, self._idle_yaw_s
            # Cleared by motion (the drone got through) OR by going stale (nobody
            # is pushing into it any more, so the claim no longer stands).
            stale = p.stale_clear_s > 0.0 and idle >= p.stale_clear_s
            if good >= p.clear_ticks or stale:
                self._verdict = StuckVerdict()
        return self._verdict

