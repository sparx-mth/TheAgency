"""Localization abstraction: shared internal types and provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation


@dataclass(frozen=True)
class LocalizationEstimate:
    """
    Platform-agnostic localization result.
    No ROS2 types — convert at the node boundary.
    """
    pose: Pose3D
    source: str          # e.g. "apriltag", "optical_flow", "amcl"
    confidence: float    # 0.0 – 1.0
    stamp_sec: float
    pos_std_m: float = 0.05
    yaw_std_rad: float = 0.05
    #: How much of the COMMANDED motion the drone is actually achieving, 0..1,
    #: or None for providers that do not watch the command stream. Low while
    #: commands are being sent is the direct signature of a drone pressed
    #: against an obstacle: it is told to move and the world says it did not.
    cmd_effectiveness: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.pos_std_m <= 0:
            raise ValueError(f"pos_std_m must be > 0, got {self.pos_std_m}")
        if self.yaw_std_rad <= 0:
            raise ValueError(f"yaw_std_rad must be > 0, got {self.yaw_std_rad}")


class BaseLocalizationProvider(ABC):
    """
    Abstract localization provider.

    Implement update() with your sensor-specific algorithm.
    All ROS2 I/O lives in the node layer — providers are pure Python.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier used in /xtend/localization_source and logs."""
        ...

    @abstractmethod
    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        """
        Process one sensor observation.
        Returns an estimate when the algorithm produces a result, None otherwise.
        obs.rgb, obs.depth, obs.cloud are all optional — use what your method needs.
        """
        ...

    def is_healthy(self) -> bool:
        """Return False when the provider has entered a failed state."""
        return True

    def reset(self) -> None:
        """Optional: reset accumulated state (e.g. integrated odometry drift)."""