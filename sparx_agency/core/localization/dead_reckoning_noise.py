"""Dead-reckoning localization noise model (pure numpy, ROS-free).

Emulates what a real IMU/VIO pipeline *without loop closure* outputs: each tick
integrates a noisy body-frame motion estimate into a self-propagating belief
pose. The belief is never re-anchored to ground truth, so error accumulates
exactly the way real dead-reckoning fails.

Given the true ground-truth pose at each tick, the model:

  1. Computes the true body-frame increment since the previous tick
     ``dT_body = T_gt_prev^-1 @ T_gt`` and reads its translation
     ``(dx, dy, dz)`` and yaw ``dyaw``.
  2. Perturbs each axis independently with three INDEPENDENT noise sources
     (every parameter defaults to ``0`` -> a clean pass-through run):

       a) SCALE-FACTOR DRIFT — error per unit of body-frame motion in that
          axis (IMU scale error). ``x/y/z`` are per-metre of body motion;
          ``yaw`` is per-radian of yaw rotation (so a yaw-in-place still
          drifts, unlike a per-metre model).
       b) TIME-BASED BIAS — drift per second, independent of motion
          (always-on gyro/accel bias). Makes the belief wander even while
          hovering perfectly still.
       c) PER-TICK JITTER — i.i.d. Gaussian applied to the *published* pose
          only; it does NOT enter the integrated belief (models downstream
          measurement noise after the filter has run).

  3. Rebuilds the noisy body-frame increment and propagates the belief
     ``T_belief <- T_belief @ dT_body_noisy``. A wrong yaw step bends every
     subsequent forward translation, so position error grows with distance
     flown — no extra logic needed.
  4. Optionally injects a rare one-shot body-frame jump (outlier).

Roll and pitch are passed through cleanly: a stabilised indoor drone observes
them tightly from gravity, so VIO keeps them accurate in practice. The noisy
increment therefore keeps the true ``dT_body`` rotation and only *pre-multiplies*
a yaw-error rotation ``Rz(err_yaw)`` onto it, which is mathematically identical
to perturbing the ZYX-yaw while preserving roll/pitch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

#: Perturbed axes. ``x/y/z`` are body-frame translations; ``yaw`` is the
#: ZYX yaw of the body-frame increment.
AXES = ("x", "y", "z", "yaw")


def _zero_axes() -> Dict[str, float]:
    return {a: 0.0 for a in AXES}


def _yaw_rotation_3x3(angle: float) -> np.ndarray:
    """Right-handed rotation about +z as a 3x3 matrix."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


@dataclass
class DeadReckoningNoiseParams:
    """Per-axis noise parameters for :class:`DeadReckoningNoiseModel`.

    Each dict is keyed by axis name (``"x"``, ``"y"``, ``"z"``, ``"yaw"``).
    All default to zero, which makes the model an exact pass-through of the
    ground-truth pose.

    Attributes:
        jitter_mean: Constant per-tick offset added to the published pose
            (m for x/y/z, rad for yaw). Memoryless; not integrated.
        jitter_std: Per-tick Gaussian std of the published-pose jitter.
        drift_mean_per_motion: Scale-factor bias. ``x/y/z`` are per metre of
            body motion in that axis; ``yaw`` is per radian of yaw rotation.
        drift_std_per_motion: Random-walk std of the scale-factor drift,
            scaled by ``sqrt(|motion|)`` each tick.
        bias_per_s_mean: Constant bias *rate* (per second) seeding the
            always-on bias state (m/s for x/y/z, rad/s for yaw).
        bias_per_s_std: Random-walk std (per ``sqrt(second)``) of the bias
            state, applied every tick regardless of motion.
        outlier_rate_hz: Expected rate of one-shot body-frame jumps (Hz).
        outlier_pos_std: Std of the x/y/z jump when an outlier fires (m).
        outlier_yaw_std: Std of the yaw jump when an outlier fires (rad).
    """

    jitter_mean: Dict[str, float] = field(default_factory=_zero_axes)
    jitter_std: Dict[str, float] = field(default_factory=_zero_axes)
    drift_mean_per_motion: Dict[str, float] = field(default_factory=_zero_axes)
    drift_std_per_motion: Dict[str, float] = field(default_factory=_zero_axes)
    bias_per_s_mean: Dict[str, float] = field(default_factory=_zero_axes)
    bias_per_s_std: Dict[str, float] = field(default_factory=_zero_axes)
    outlier_rate_hz: float = 0.0
    outlier_pos_std: float = 0.0
    outlier_yaw_std: float = 0.0

    def any_jitter(self) -> bool:
        return any(v != 0 for v in (*self.jitter_mean.values(),
                                    *self.jitter_std.values()))

    def any_drift(self) -> bool:
        return any(v != 0 for v in (*self.drift_mean_per_motion.values(),
                                    *self.drift_std_per_motion.values()))

    def any_bias(self) -> bool:
        return any(v != 0 for v in (*self.bias_per_s_mean.values(),
                                    *self.bias_per_s_std.values()))

    def any_outlier(self) -> bool:
        return (self.outlier_rate_hz > 0.0 and
                (self.outlier_pos_std > 0.0 or self.outlier_yaw_std > 0.0))

    def enabled(self) -> bool:
        """True if any source would perturb the pose (else it is a no-op)."""
        return (self.any_jitter() or self.any_drift()
                or self.any_bias() or self.any_outlier())


class DeadReckoningNoiseModel:
    """Self-propagating dead-reckoning belief driven by ground-truth ticks.

    Stateful: it remembers the integrated belief ``T_belief``, the previous
    ground-truth pose and the slowly-walking bias state. Feed it the *true*
    4x4 pose and the elapsed time each tick; it returns the 4x4 pose to
    publish (belief + per-tick jitter).

    Args:
        params: Per-axis noise configuration.
        rng: A seeded ``numpy.random.RandomState`` (or compatible) for
            reproducibility. The caller owns seeding.
    """

    def __init__(self, params: DeadReckoningNoiseParams,
                 rng: Optional[np.random.RandomState] = None) -> None:
        self.params = params
        self.rng = rng if rng is not None else np.random.RandomState()
        self._T_belief: Optional[np.ndarray] = None
        self._T_gt_prev: Optional[np.ndarray] = None
        # Bias state walks each tick; it is seeded with the constant rate.
        self._bias_state: Dict[str, float] = dict(params.bias_per_s_mean)

    @property
    def belief(self) -> Optional[np.ndarray]:
        """The current integrated belief (no jitter), or None before init."""
        return self._T_belief

    def reset(self) -> None:
        """Forget the belief so the next :meth:`step` re-anchors to truth."""
        self._T_belief = None
        self._T_gt_prev = None
        self._bias_state = dict(self.params.bias_per_s_mean)

    def step(self, T_gt: np.ndarray, dt: float) -> np.ndarray:
        """Advance the belief by one tick and return the pose to publish.

        Args:
            T_gt: The true 4x4 world->body pose this tick.
            dt: Wall-clock seconds since the previous tick (>= 0). Ignored on
                the first call.

        Returns:
            The 4x4 world->body pose to publish: the integrated belief with
            per-tick jitter applied on top. On the first call this equals
            ``T_gt`` (the belief is initialised to truth).
        """
        T_gt = np.asarray(T_gt, dtype=float)

        # First call: belief == truth, nothing to integrate yet.
        if self._T_belief is None:
            self._T_belief = T_gt.copy()
            self._T_gt_prev = T_gt.copy()
            return T_gt.copy()

        dt = max(float(dt), 0.0)
        p = self.params

        # 1. Bias-state random walk (always-on, motion-independent).
        for a in AXES:
            walk = p.bias_per_s_std[a]
            if walk > 0.0 and dt > 0.0:
                self._bias_state[a] += self.rng.normal(0.0, walk * math.sqrt(dt))

        # 2. True body-frame increment from ground truth.
        dT_body = np.linalg.inv(self._T_gt_prev) @ T_gt
        dR_body = dT_body[:3, :3]
        true_vals = {
            "x": float(dT_body[0, 3]),
            "y": float(dT_body[1, 3]),
            "z": float(dT_body[2, 3]),
            # ZYX yaw of the increment (exact for the stabilised regime this
            # models, where roll/pitch of an increment are ~0).
            "yaw": math.atan2(float(dR_body[1, 0]), float(dR_body[0, 0])),
        }
        self._T_gt_prev = T_gt.copy()

        # 3. Per-axis error: scale-factor drift + time-based bias.
        err = {}
        for a in AXES:
            true_d = true_vals[a]
            mag = abs(true_d)
            e = p.drift_mean_per_motion[a] * true_d        # signed scale error
            std_rate = p.drift_std_per_motion[a]
            if std_rate > 0.0 and mag > 0.0:
                e += self.rng.normal(0.0, std_rate * math.sqrt(mag))
            e += self._bias_state[a] * dt                  # motion-independent
            err[a] = e

        # 4. Optional one-shot outlier (rare body-frame jump).
        if (p.outlier_rate_hz > 0.0 and dt > 0.0
                and self.rng.random_sample() < p.outlier_rate_hz * dt):
            if p.outlier_pos_std > 0.0:
                err["x"] += self.rng.normal(0.0, p.outlier_pos_std)
                err["y"] += self.rng.normal(0.0, p.outlier_pos_std)
                err["z"] += self.rng.normal(0.0, p.outlier_pos_std)
            if p.outlier_yaw_std > 0.0:
                err["yaw"] += self.rng.normal(0.0, p.outlier_yaw_std)

        # 5. Build the noisy increment. Pre-multiplying Rz(err_yaw) onto the
        # true increment rotation preserves roll/pitch exactly and adds only
        # the yaw error (identical to perturbing the ZYX yaw).
        dT_noisy = np.eye(4)
        dT_noisy[:3, :3] = _yaw_rotation_3x3(err["yaw"]) @ dR_body
        dT_noisy[0, 3] = true_vals["x"] + err["x"]
        dT_noisy[1, 3] = true_vals["y"] + err["y"]
        dT_noisy[2, 3] = true_vals["z"] + err["z"]

        # 6. Propagate the belief — the dead-reckoning step.
        self._T_belief = self._T_belief @ dT_noisy

        # 7. Per-tick jitter (memoryless; published pose only).
        return self._apply_jitter(self._T_belief)

    def _apply_jitter(self, T: np.ndarray) -> np.ndarray:
        p = self.params
        if not p.any_jitter():
            return T.copy()
        j = {a: p.jitter_mean[a] for a in AXES}
        for a in AXES:
            if p.jitter_std[a] > 0.0:
                j[a] += self.rng.normal(0.0, p.jitter_std[a])
        J = np.eye(4)
        J[:3, :3] = _yaw_rotation_3x3(j["yaw"])
        J[0, 3], J[1, 3], J[2, 3] = j["x"], j["y"], j["z"]
        return T @ J
