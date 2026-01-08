"""
Smoother interface.

A Smoother converts a geometric path (Path2D) into a time-parameterized
trajectory (Trajectory) suitable for tracking/control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from sparx_agency.core.common.types import Path2D, Trajectory, KinematicLimits


@dataclass(frozen=True)
class SmootherRequest:
    """
    Input to trajectory smoothers.

    Attributes:
        path: Geometric path to smooth.
        limits: Optional kinematic constraints. If None, smoother uses defaults.
        options: Algorithm-specific parameters (e.g., continuity order, speed).
    """
    path: Path2D
    limits: Optional[KinematicLimits] = None
    options: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseSmoother(Protocol):
    """
    Smoother protocol.

    Converts Path2D → Trajectory with time parameterization.
    """
    name: str

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """
        Generate trajectory from path.

        Args:
            request: Path and constraints.
            world: Optional environment context (e.g., for obstacle-aware smoothing).

        Returns:
            Time-parameterized trajectory.
        """
        ...
