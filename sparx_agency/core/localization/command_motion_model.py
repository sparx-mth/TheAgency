"""Command-aware motion prior with self-learned trust (ROS-free, 3.8-safe).

The localization filter needs a prediction of how the drone moved between two
camera fixes. The commands sent to the platform are an obvious source — but on
this platform they are also a *lying* one: the drone regularly ends up pressed
against a wall or an obstacle, the follower keeps commanding motion, and nothing
moves. A prior that believes those commands would walk the pose through the wall
the drone is stuck on, which is the exact situation where an honest pose matters
most.

So the model earns its trust instead of assuming it. It integrates the commanded
twist between fixes, and every time a good measurement arrives it compares what
was commanded with what the camera actually saw:

    effectiveness = (achieved displacement along the commanded direction)
                    / (commanded displacement)

smoothed over recent fixes, separately for translation and yaw (a drone wedged
against a wall can often still rotate). Prediction is then the commanded motion
scaled by that effectiveness — a drone that has been moving as told is predicted
at nearly full command, a stuck one at nearly nothing. The effectiveness itself
is exposed, because "commanded but not achieving" is precisely the stuck signal
the mission wants.

Trust is bounded three more ways, all guarding the same failure:

* learning only happens from confident fixes over meaningful commanded motion,
  so a noisy or absent measurement can never teach the model anything;
* a command older than ``cmd_timeout_s`` stops integrating, so a stale twist
  cannot keep pushing the pose after the follower has gone quiet;
* one prediction step is hard-capped in metres and radians, so even full trust
  cannot move the pose far without a measurement to answer to.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import Optional, Tuple


@dataclass(frozen=True)
class CommandMotionParams:
    """Tuning for :class:`CommandMotionModel` (SI units).

    Attributes:
        trust_max: Ceiling on how much of the commanded motion is ever applied,
            even at effectiveness 1.0. Keeps the prior a *prior*: measurements
            must stay the deciding voice. 0 disables the model entirely.
        eff_alpha: EMA rate for the effectiveness estimates. 0.2 converges in
            roughly a second of 10 Hz fixes — fast enough to notice hitting a
            wall, slow enough that one bad fix cannot flip the verdict.
        eff_initial: Effectiveness before anything has been learned. Starts low
            on purpose: a fresh model must prove the commands work before the
            prior leans on them.
        conf_floor: Minimum fix confidence for a learning update. Below it the
            measured displacement is as likely to be solve noise as motion, and
            learning stuck-ness from noise would be self-fulfilling.
        min_learn_disp_m / min_learn_dyaw_rad: Minimum COMMANDED motion between
            fixes before the ratio is informative. Hovering teaches nothing.
        cmd_timeout_s: A command stops integrating this long after it was set.
        max_step_m / max_step_rad: Hard cap on a single prediction step.
    """

    trust_max: float = 0.7
    eff_alpha: float = 0.2
    eff_initial: float = 0.3
    conf_floor: float = 0.3
    min_learn_disp_m: float = 0.01
    min_learn_dyaw_rad: float = 0.03
    cmd_timeout_s: float = 0.5
    max_step_m: float = 0.15
    max_step_rad: float = 0.35


class CommandMotionModel:
    """Integrate commanded twists and predict pose motion with earned trust."""

    def __init__(self, params: Optional[CommandMotionParams] = None) -> None:
        self.p = params or CommandMotionParams()
        self._cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # vx, vy, wz body
        self._cmd_t: Optional[float] = None
        self._acc_body = [0.0, 0.0]   # integrated commanded displacement, body frame
        self._acc_dyaw = 0.0
        self._eff_lin = self.p.eff_initial
        self._eff_yaw = self.p.eff_initial

    # ── command ingestion ────────────────────────────────────────────
    def set_command(self, vx: float, vy: float, wz: float, stamp_sec: float) -> None:
        """Record the twist now being executed (body frame, m/s and rad/s)."""
        self._integrate_to(stamp_sec)
        self._cmd = (float(vx), float(vy), float(wz))
        self._cmd_t = float(stamp_sec)

    def _integrate_to(self, t: float) -> None:
        """Advance the commanded-motion ledger to time ``t``."""
        if self._cmd_t is None:
            return
        dt = t - self._cmd_t
        if dt <= 0.0:
            return
        live = min(dt, self.p.cmd_timeout_s)  # a stale command stops pushing
        vx, vy, wz = self._cmd
        self._acc_body[0] += vx * live
        self._acc_body[1] += vy * live
        self._acc_dyaw += wz * live
        self._cmd_t = t

    # ── prediction ───────────────────────────────────────────────────
    def consume(self, now_sec: float, yaw: float) -> Tuple[float, float, float,
                                                           float, float, float]:
        """Commanded motion since the last consume, as world-frame deltas.

        Returns ``(dx, dy, dyaw, raw_dx, raw_dy, raw_dyaw)``: the first three are
        effectiveness-scaled and step-capped — apply them to the pose prior; the
        raw three are the unscaled commanded motion — accumulate them for
        :meth:`observe`, which needs to know what was *asked* regardless of how
        much of it was believed.
        """
        self._integrate_to(now_sec)
        bx, by = self._acc_body
        raw_dyaw = self._acc_dyaw
        self._acc_body = [0.0, 0.0]
        self._acc_dyaw = 0.0

        c, s = cos(yaw), sin(yaw)
        raw_dx = c * bx - s * by
        raw_dy = s * bx + c * by

        gl = self._eff_lin * self.p.trust_max
        gy = self._eff_yaw * self.p.trust_max
        dx, dy, dyaw = raw_dx * gl, raw_dy * gl, raw_dyaw * gy

        # One step may never outrun what a measurement could still correct.
        norm = hypot(dx, dy)
        if norm > self.p.max_step_m:
            k = self.p.max_step_m / norm
            dx, dy = dx * k, dy * k
        if dyaw > self.p.max_step_rad:
            dyaw = self.p.max_step_rad
        elif dyaw < -self.p.max_step_rad:
            dyaw = -self.p.max_step_rad
        return dx, dy, dyaw, raw_dx, raw_dy, raw_dyaw

    # ── learning ─────────────────────────────────────────────────────
    def observe(self, meas_dx: float, meas_dy: float, meas_dyaw: float,
                cmd_dx: float, cmd_dy: float, cmd_dyaw: float,
                confidence: float) -> None:
        """Update effectiveness from one measured-vs-commanded motion pair.

        ``cmd_*`` is the RAW commanded motion over the same interval as the
        measured deltas (world frame). Axes learn independently, and each only
        when its own commanded motion was large enough to mean something.
        """
        if confidence < self.p.conf_floor:
            return
        a = self.p.eff_alpha

        cmd_norm = hypot(cmd_dx, cmd_dy)
        if cmd_norm >= self.p.min_learn_disp_m:
            achieved = (meas_dx * cmd_dx + meas_dy * cmd_dy) / cmd_norm
            ratio = max(0.0, min(1.0, achieved / cmd_norm))
            self._eff_lin += a * (ratio - self._eff_lin)

        if abs(cmd_dyaw) >= self.p.min_learn_dyaw_rad:
            ratio = max(0.0, min(1.0, meas_dyaw / cmd_dyaw))
            self._eff_yaw += a * (ratio - self._eff_yaw)

    # ── introspection ────────────────────────────────────────────────
    @property
    def effectiveness_lin(self) -> float:
        """Learned fraction of commanded TRANSLATION actually achieved, 0..1."""
        return self._eff_lin

    @property
    def effectiveness_yaw(self) -> float:
        """Learned fraction of commanded ROTATION actually achieved, 0..1."""
        return self._eff_yaw

    @property
    def enabled(self) -> bool:
        return self.p.trust_max > 0.0

    def reset(self) -> None:
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_t = None
        self._acc_body = [0.0, 0.0]
        self._acc_dyaw = 0.0
        self._eff_lin = self.p.eff_initial
        self._eff_yaw = self.p.eff_initial
