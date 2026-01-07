"""
Smoother interface.

A Smoother converts a geometric path into a time-parameterized trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from core.common.types import Path2D, Trajectory, DynamicLimits


@dataclass(frozen=True)
class SmootherRequest:
    """
    Inputs to the trajectory smoother.

    - `limits` is optional: if not provided, the smoother may use conservative defaults.
    - `options` can include algorithm-specific parameters (degree, continuity, etc.).
    """
    path: Path2D
    limits: Optional[DynamicLimits] = None
    options: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseSmoother(Protocol):
    """Smoother contract."""
    name: str

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """
        Produce a time-parameterized trajectory from a path.

        Args:
            request: SmootherRequest with path + optional limits/options
            world: optional environment context (e.g., for altitude rules, obstacle-aware smoothing)

        Returns:
            Trajectory object implementing core.common.types.Trajectory
        """
        ...
