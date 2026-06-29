"""
Lightweight signal filters — no ROS, no external dependencies.
"""
from __future__ import annotations


class ExponentialMovingAverage:
    """
    Scalar or array EMA:  value = alpha * new + (1 - alpha) * value

    alpha=1.0  → no smoothing (pass-through)
    alpha→0    → heavy smoothing (slow to track changes)

    First call to update() sets the initial value (no warm-up lag).
    Pass initial= to seed with a known value instead.
    """

    def __init__(self, alpha: float, initial=None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._value = initial

    @property
    def value(self):
        return self._value

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def update(self, new_value):
        """Update the filter with a new sample and return the current smoothed value."""
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.alpha * new_value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self, value=None) -> None:
        """Reset to a given value (or None to re-initialise on next update)."""
        self._value = value