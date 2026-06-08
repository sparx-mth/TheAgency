"""Parameter types for the dead-reckoning localization noise model (ROS-free).

The stateful model lives in
:mod:`sparx_agency.core.localization.dead_reckoning_noise`; this module holds
only the configuration vocabulary it consumes, so the type is importable
without pulling in the integration logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

#: Perturbed axes. ``x/y/z`` are body-frame translations; ``yaw`` is the
#: ZYX yaw of the body-frame increment.
AXES = ("x", "y", "z", "yaw")


def _zero_axes() -> Dict[str, float]:
    return {a: 0.0 for a in AXES}


@dataclass
class DeadReckoningNoiseParams:
    """Per-axis noise parameters for the dead-reckoning noise model.

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
