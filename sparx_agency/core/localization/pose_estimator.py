"""Windowed pose + velocity estimator (ROS-free, 3.8-safe).

A small, *interpretable* fuser that decouples a noisy ~10 Hz localization stream
(AprilTag PoseStamped) from the control loop. It keeps a short sliding window of
timestamped (x, y, yaw) readings plus the one-axis command (vx, wz) currently
being executed, and on ``estimate(now)`` does a centred least-squares line fit
over the window:

  * the fit's **value at the window centroid is the denoised pose** (exactly the
    window mean, so a single jumpy reading shifts it by only ~1/N), and
  * the fit's **slope is the measured velocity**, blended with the *commanded*
    rate by a feed-forward fraction that grows with how hard the axis is driven —
    so during rotation (the noisy regime) the estimate leans on the command and a
    lone AprilTag jump can never dominate.

This is the literal "fuse the last ~2 readings plus the command" the platform
needs, not a black-box filter. The platform is strictly one-axis-at-a-time, so
the command alone selects the regime:

  * **stopped**  (vx≈0, wz≈0, and the measured rate has settled): average the
    window → rejects hover drift; this generalises the follower's YAW_SETTLE
    ``circular_mean``.
  * **turning**  (wz commanded): denoise + propagate the heading (this is the
    pose the follower's mid-burst cut reads), and also expose the true yaw-rate
    (``wz``) for a controller that wants a rate-based reached check.
  * **forward**  (vx commanded): denoise position, dropping the perpendicular
    (lateral-drift) component geometrically.

``estimate`` is side-effect-free: it re-fits the pruned window every call, so it
cannot be corrupted by interleaving with ``add_measurement``. On a localization
dropout it dead-reckons from the command (``coast``) and then holds the last
reading (``hold``) with decaying confidence, so the adapter can gate a stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, sin
from typing import List, Optional, Tuple

from sparx_agency.core.common.types import Pose2D, circular_mean, normalize_angle


@dataclass(frozen=True)
class PoseEstimatorParams:
    """Tuning for :class:`WindowedPoseEstimator` (SI units; 10 Hz assumed).

    Attributes:
        window_s: Least-squares fit window length (s). ``window_n ≈ window_s*10``
            samples at 10 Hz — long enough to average jitter, short enough that a
            0.7 rad/s turn only spans ~0.4 rad across it (the yaw unwrap stays
            well below π).
        min_samples: Minimum readings to attempt a slope fit; below this (or a
            degenerate time spread) the slope is forced to 0 and the pose is the
            window centroid / last reading.
        max_buffer_s: Discard readings older than this (bounds the buffer). Kept
            ``>= max_coast_s`` so a dropout can still hold the last reading.
        wz_cmd_eps: |wz_cmd| (rad/s) at/above which the regime is TURNING.
        vx_cmd_eps: |vx_cmd| (m/s) at/above which the regime is FORWARD.
        settle_wz_eps: Measured |wz| (rad/s) below which a zero-command state is
            treated as truly STOPPED — waits out yaw coast after a stop command
            (mirrors the follower's ``yaw_settle_eps``). Load-bearing.
        settle_vx_eps: Measured |speed| (m/s) below which a zero-command state is
            treated as truly STOPPED — the symmetric guard that waits out forward
            coast after a stop command, so a glide is not frozen as a hover.
        wz_ff_ref: Commanded yaw rate (= follower ``yaw_rate``) mapping to the
            maximum feed-forward trust.
        vx_ff_ref: Commanded forward speed (= follower ``vel_x``) mapping to the
            maximum feed-forward trust.
        ff_blend_min: Feed-forward fraction when the axis is barely driven (trust
            the measured fit).
        ff_blend_max: Feed-forward fraction at/above the reference command (during
            rotation, lean on the commanded rate so one jumpy reading can't win).
        dropout_s: Measurement age (s) beyond which the estimate is flagged COAST
            and dead-reckons from the command.
        max_coast_s: Cap on the dead-reckon horizon (s); beyond it the pose is
            HELD at the last reading and confidence ≈ 0.
        fresh_tau_s: Confidence freshness decay constant: ``exp(-age/fresh_tau_s)``.
    """
    window_s: float = 0.6
    min_samples: int = 2
    max_buffer_s: float = 1.5
    wz_cmd_eps: float = 0.05
    vx_cmd_eps: float = 0.03
    settle_wz_eps: float = 0.05
    settle_vx_eps: float = 0.10   # measured |speed| below which a zero-cmd state is STOPPED
                                  # (generous: catches a sustained glide, tolerates a single
                                  #  edge jump so hover drift rejection is not lost)
    wz_ff_ref: float = 0.7
    vx_ff_ref: float = 0.3
    ff_blend_min: float = 0.15
    ff_blend_max: float = 0.6
    dropout_s: float = 0.25
    max_coast_s: float = 1.0
    fresh_tau_s: float = 0.3


@dataclass(frozen=True)
class PoseEstimate:
    """Output of :meth:`WindowedPoseEstimator.estimate`."""
    t: float            # the `now` it was evaluated at
    x: float            # smoothed planar position (m, path frame)
    y: float
    yaw: float          # denoised heading (rad, normalized)
    vx: float           # forward speed along heading (m/s)
    wz: float           # yaw rate (rad/s)
    confidence: float   # [0, 1] = freshness * window fill
    n_samples: int      # readings used in the fit
    age: float          # now - newest measurement time (s)
    mode: str           # "stopped" | "turning" | "forward" | "coast" | "hold" | "invalid"
    stopped: bool

    def as_pose2d(self) -> Pose2D:
        """The denoised pose as a :class:`Pose2D` for the follower."""
        return Pose2D(self.x, self.y, self.yaw)

    def wz_deg_s(self) -> float:
        """Estimated yaw rate in degrees/second (for the yaw controller)."""
        return self.wz * 57.29577951308232


def _fit_slope(ts: List[float], vals: List[float], t_bar: float,
               min_samples: int) -> Tuple[float, float]:
    """Centred least-squares slope of ``vals`` vs ``ts`` and their mean value.

    Returns ``(slope, mean)``. The fitted value at the centroid is exactly the
    mean (so the denoised channel is the window average). The slope is forced to
    0 when there are too few samples or the time spread is degenerate.
    """
    n = len(vals)
    v_bar = sum(vals) / n
    if n < min_samples:
        return 0.0, v_bar
    num = 0.0
    den = 0.0
    for t, v in zip(ts, vals):
        tau = t - t_bar
        num += tau * (v - v_bar)
        den += tau * tau
    if den < 1e-9:
        return 0.0, v_bar
    return num / den, v_bar


def _ff_blend(cmd_mag: float, ref: float, lo: float, hi: float) -> float:
    """Feed-forward fraction: ``lo`` idle → ``hi`` at/above the reference command."""
    if ref <= 0.0:
        return lo
    frac = cmd_mag / ref
    if frac > 1.0:
        frac = 1.0
    return lo + (hi - lo) * frac


class WindowedPoseEstimator:
    """Sliding-window LS pose/velocity fuser with command feed-forward."""

    def __init__(self, params: Optional[PoseEstimatorParams] = None) -> None:
        self.params = params or PoseEstimatorParams()
        self.reset()

    # ─── Public API ──────────────────────────────────────────────
    def reset(self) -> None:
        """Clear the measurement buffer and the last command."""
        self._buf: List[Tuple[float, float, float, float]] = []  # (t, x, y, yaw)
        self._vx_cmd = 0.0
        self._wz_cmd = 0.0

    def add_measurement(self, x: float, y: float, yaw: float, t: float) -> None:
        """Append one localization reading (drops non-increasing/duplicate ``t``)."""
        if self._buf and t <= self._buf[-1][0]:
            return                                   # jitter / duplicate guard
        self._buf.append((float(t), float(x), float(y), float(yaw)))
        cutoff = t - self.params.max_buffer_s
        if self._buf[0][0] < cutoff:
            self._buf = [r for r in self._buf if r[0] >= cutoff]

    def set_command(self, vx: float, wz: float, t: Optional[float] = None) -> None:
        """Store the one-axis command currently being executed (vx, wz)."""
        self._vx_cmd = float(vx)
        self._wz_cmd = float(wz)

    def estimate(self, now: float) -> PoseEstimate:
        """Re-fit the window and return the fused pose+velocity (pure read)."""
        p = self.params
        if not self._buf:
            return PoseEstimate(now, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, float("inf"),
                                "invalid", True)

        t_new = self._buf[-1][0]
        age = now - t_new
        x_last, y_last, yaw_last = self._buf[-1][1], self._buf[-1][2], self._buf[-1][3]

        # Long dropout: hold the last reading, confidence ~0 (adapter can stop).
        if age > p.max_coast_s:
            return PoseEstimate(now, x_last, y_last, normalize_angle(yaw_last),
                                0.0, 0.0, 0.0, len(self._buf), age, "hold", True)

        # Window for the LS fit (readings within window_s of now).
        win = [r for r in self._buf if now - r[0] <= p.window_s]
        if not win:
            win = [self._buf[-1]]
        ts = [r[0] for r in win]
        n = len(win)
        t_bar = sum(ts) / n
        dt_prop = now - t_bar
        if dt_prop < 0.0:
            dt_prop = 0.0
        elif dt_prop > p.max_coast_s:
            dt_prop = p.max_coast_s

        # Unwrap yaw around its circular mean before fitting a slope.
        yaw_ref = circular_mean([r[3] for r in win])
        u = [yaw_ref + normalize_angle(r[3] - yaw_ref) for r in win]
        wz_ls, u_bar = _fit_slope(ts, u, t_bar, p.min_samples)
        x_bar = sum(r[1] for r in win) / n
        y_bar = sum(r[2] for r in win) / n
        vx_w, _ = _fit_slope(ts, [r[1] for r in win], t_bar, p.min_samples)
        vy_w, _ = _fit_slope(ts, [r[2] for r in win], t_bar, p.min_samples)
        speed_ls = hypot(vx_w, vy_w)      # measured ground speed (heading-agnostic)

        vx_cmd, wz_cmd = self._vx_cmd, self._wz_cmd
        fill = min(1.0, float(n) / max(1.0, round(p.window_s * 10.0)))
        freshness = exp(-max(0.0, age) / p.fresh_tau_s) if p.fresh_tau_s > 0 else 1.0
        confidence = freshness * fill

        # Dropout (short): dead-reckon on the command rather than a stale pose.
        if age > p.dropout_s:
            yaw_est = normalize_angle(u_bar + wz_cmd * dt_prop)
            x_est = x_bar + vx_cmd * cos(yaw_est) * dt_prop
            y_est = y_bar + vx_cmd * sin(yaw_est) * dt_prop
            return PoseEstimate(now, x_est, y_est, yaw_est, vx_cmd, wz_cmd,
                                confidence, n, age, "coast", False)

        # STOPPED requires BOTH a zero command and a settled MEASURED motion on
        # each axis, so neither a yaw coast nor a forward glide after a stop
        # command is frozen as a hover (the symmetric settle guards).
        stopped = (abs(vx_cmd) < p.vx_cmd_eps and abs(wz_cmd) < p.wz_cmd_eps
                   and abs(wz_ls) < p.settle_wz_eps and speed_ls < p.settle_vx_eps)
        if stopped:
            # Average the window: rejects hover drift; slopes discarded.
            return PoseEstimate(now, x_bar, y_bar, normalize_angle(yaw_ref),
                                0.0, 0.0, confidence, n, age, "stopped", True)

        # Moving: blend the measured slope with the commanded rate (feed-forward).
        beta_wz = _ff_blend(abs(wz_cmd), p.wz_ff_ref, p.ff_blend_min, p.ff_blend_max)
        wz_est = (1.0 - beta_wz) * wz_ls + beta_wz * wz_cmd
        yaw_est = normalize_angle(u_bar + wz_est * dt_prop)

        vx_ls = vx_w * cos(yaw_est) + vy_w * sin(yaw_est)   # along-heading; drop lateral
        beta_vx = _ff_blend(abs(vx_cmd), p.vx_ff_ref, p.ff_blend_min, p.ff_blend_max)
        vx_est = (1.0 - beta_vx) * vx_ls + beta_vx * vx_cmd
        x_est = x_bar + vx_est * cos(yaw_est) * dt_prop
        y_est = y_bar + vx_est * sin(yaw_est) * dt_prop

        if abs(wz_cmd) >= p.wz_cmd_eps:
            mode = "turning"
        elif abs(vx_cmd) >= p.vx_cmd_eps:
            mode = "forward"
        else:
            mode = "coast"            # commanded stop, but still coasting (inertia)
        return PoseEstimate(now, x_est, y_est, yaw_est, vx_est, wz_est,
                            confidence, n, age, mode, False)
