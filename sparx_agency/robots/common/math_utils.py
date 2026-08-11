# math_utils.py
"""Dependency-free scalar clamping helpers.

Kept separate from helpers.py (which pulls in cv2/sensor_msgs for camera
utilities) so drone control code can use these without needing OpenCV
installed — e.g. the ROBOTICAN Rooster container's Python 3.8 env.
"""


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def clamp_symmetric(value: float, limit: float) -> float:
    """Clamp value to [-limit, limit]."""
    return max(-float(limit), min(float(limit), float(value)))


def clamp_axis(value: float, limit: float = 1000.0) -> int:
    """Clamp to [-limit, limit] and cast to int (for drone controller axis values)."""
    return int(clamp_symmetric(value, limit))


def slew(target: float, current: float, max_step: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_step`` (rate limit)."""
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target
