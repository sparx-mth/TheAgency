"""Pure Pursuit tracker package (2D and 3D)."""
from sparx_agency.core.planning.interfaces.tracker import (
    TrackerRequest,
    TrackerResult,
    BaseTracker,
)
from .params import PurePursuitParams, PurePursuitParams3D
from .tracker import PurePursuitTracker, PurePursuitTracker3D

__all__ = [
    # 2D (original)
    "PurePursuitParams",
    "PurePursuitTracker",
    # 3D (new)
    "PurePursuitParams3D",
    "PurePursuitTracker3D",
    # Shared
    "TrackerRequest",
    "TrackerResult",
    "BaseTracker",
]