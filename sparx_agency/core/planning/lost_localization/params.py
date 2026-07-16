"""Tuning for the lost-localization recovery (ROS-free, 3.8-safe)."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class LostLocalizationParams:
    """Tuning for :class:`~.state_machine.LostLocalizationRecovery` (SI units).

    The two thresholds are deliberately different, and the gap between them is
    the whole point: AprilTag localization at ~7 Hz drops a frame routinely, so
    ``stale_s`` is short enough to stop the drone the moment its pose goes cold,
    while ``ladder_s`` is long enough that an ordinary gap between markers never
    provokes a blind back-up manoeuvre.

    Attributes:
        enabled: Master on/off. False => every decision is inactive with no
            command (the drone is never touched); the node then publishes nothing
            and the follower keeps ownership of cmd_vel.
        stale_s: Pose age (s) beyond which localization is cold: STOP and hold.
            At ~7 Hz this is ~2 frames, so it fires on a genuine dropout rather
            than on jitter.
        ladder_s: Pose age (s) beyond which the recovery ladder starts. Must be
            > ``stale_s``: the drone has already been stopped for
            ``ladder_s - stale_s`` seconds by then, so the ladder only ever runs
            from a standstill.
        exit_confirm_poses: How many NEW localization messages must land before
            recovery hands the drone back. Counted as arrivals, not as ticks with
            a fresh-looking age: after one lone detection the age sits below
            ``stale_s`` for ``stale_s/dt`` ticks by itself, so a tick-based
            confirmation would let a single flickering frame end the recovery.
            >1 debounces that flicker.
        back_speed: Reverse speed (m/s, positive) for a back-up rung. Commanded
            as a negative body-x velocity.
        back_duration_s: How long one back-up rung drives. Distance is
            ``back_speed * back_duration_s`` and is OPEN LOOP -- with no pose
            there is no odometry to close it on.
        back_repeats: How many back-up rungs (each followed by a settle).
        dwell_s: Stationary settle (s) after every motion rung. This is where the
            recovery actually happens: the drone is still, so the camera gets its
            cleanest look for a tag. Keep it >= a few localization periods.
        climb_enabled: Run the climb rungs. Requires a platform that accepts a
            vertical velocity -- see the node's ``~climb_enabled`` docs.
        climb_speed: Climb speed (m/s, positive) for a climb rung. TREAT THIS AS A
            THRUST DIAL, NOT A SPEED. On XTEND the vertical axis is scaled by the
            FORWARD calibration (``|v| / 0.3 * 400``) -- there is no vertical
            constant anywhere in the stack -- and the drone climbs much harder at a
            given number than it flies forward. The default is deliberately far
            below anything that reads like a sensible climb rate, because it is not
            one.
        climb_duration_s: How long one climb rung drives. The REAL climb is bigger
            than ``climb_speed * climb_duration_s``: commands are hold-style and the
            drone keeps rising through the stop. There is no altitude feedback
            anywhere in this stack, so nothing catches an overshoot -- the ceiling
            does. Raise this only against a measured climb.
        climb_repeats: How many climb rungs (each followed by a settle). The total
            height put on is roughly ``climb_repeats * climb_duration_s`` of thrust
            plus that many overshoots; keep the product well inside your headroom.
        turn_enabled: Run the final slow 360 sweep.
        turn_rate: Yaw rate (rad/s, positive) for the sweep. Slow on purpose: the
            sweep exists to let the camera find a tag, and motion blur defeats it.
        turn_dir: Sweep direction: +1 = left/CCW, -1 = right/CW.
        turn_target_rad: Rotation to sweep before giving up (default a full turn).
        turn_timeout_s: Hard cap (s) on the sweep. This is what ends the sweep
            when no independent yaw source is available, so it MUST exceed
            ``turn_target_rad / turn_rate`` or the sweep can never complete.

    Raises:
        ValueError: On a non-positive rate/duration, a threshold ordering that
            would make the ladder unreachable, or a turn that cannot finish.
    """

    enabled: bool = True
    stale_s: float = 0.3
    ladder_s: float = 1.0
    exit_confirm_poses: int = 2
    back_speed: float = 0.25
    back_duration_s: float = 1.0
    back_repeats: int = 2
    dwell_s: float = 1.5
    climb_enabled: bool = True
    climb_speed: float = 0.08
    climb_duration_s: float = 0.4
    climb_repeats: int = 2
    turn_enabled: bool = True
    turn_rate: float = 0.30
    turn_dir: int = 1
    turn_target_rad: float = 2.0 * pi
    turn_timeout_s: float = 40.0

    def __post_init__(self) -> None:
        if self.stale_s <= 0.0:
            raise ValueError("stale_s must be > 0, got %r" % (self.stale_s,))
        if self.ladder_s <= self.stale_s:
            raise ValueError(
                "ladder_s (%r) must be > stale_s (%r): the ladder only ever runs "
                "from the stop that stale_s already commanded"
                % (self.ladder_s, self.stale_s))
        if self.exit_confirm_poses < 1:
            raise ValueError("exit_confirm_poses must be >= 1, got %r"
                             % (self.exit_confirm_poses,))
        if self.dwell_s <= 0.0:
            raise ValueError("dwell_s must be > 0, got %r" % (self.dwell_s,))
        for name in ("back_speed", "back_duration_s", "climb_speed",
                     "climb_duration_s", "turn_rate", "turn_target_rad",
                     "turn_timeout_s"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, value))
        for name in ("back_repeats", "climb_repeats"):
            if getattr(self, name) < 0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))
        if self.turn_dir not in (-1, 1):
            raise ValueError("turn_dir must be +1 (left) or -1 (right), got %r"
                             % (self.turn_dir,))
        if self.turn_enabled:
            # Without an independent yaw source the timeout is the ONLY thing that
            # ends the sweep, so a timeout below the nominal sweep time silently
            # truncates it -- a "360" that is really an arc.
            nominal_s = self.turn_target_rad / self.turn_rate
            if self.turn_timeout_s < nominal_s:
                raise ValueError(
                    "turn_timeout_s (%.1fs) is below the %.1fs the sweep needs at "
                    "turn_rate=%.2f rad/s -- the turn could never complete"
                    % (self.turn_timeout_s, nominal_s, self.turn_rate))
