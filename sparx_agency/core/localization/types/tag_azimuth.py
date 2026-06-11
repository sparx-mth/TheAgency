"""Data types for AprilTag azimuth estimation (ROS-free)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagBearingObservation:
    """A single tag observation reduced to a camera-frame bearing.

    Only the translation components needed for azimuth are kept. Assumes the
    optical convention: Z forward, X right (typical camera optical frame).

    Attributes:
        tag_id: Detected tag identifier.
        tx: Tag translation along camera X (meters, right-positive).
        tz: Tag translation along camera Z (meters, forward-positive).
    """

    tag_id: int
    tx: float
    tz: float
