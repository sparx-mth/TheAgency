"""Cubic Hermite spline trajectory smoother (2D and 3D)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from sparx_agency.core.common.types import Path2D, Trajectory, KinematicLimits

# Try importing Path3D
try:
    from sparx_agency.core.common.types import Path3D
except ImportError:
    from typing import Tuple
    from sparx_agency.core.common.types import Pose3D

    @dataclass(frozen=True)
    class Path3D:
        points: Tuple[Pose3D, ...]
        frame_id: str = "map"
        metadata: Dict[str, Any] = field(default_factory=dict)

# Import DiscreteTrajectory adapter
try:
    from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory
except ImportError:
    # Minimal fallback implementation
    from sparx_agency.core.common.types import TrajectoryPoint
    from typing import List, Tuple as TypingTuple

    class DiscreteTrajectory:
        """Minimal trajectory wrapper."""
        def __init__(self, points: TypingTuple[TrajectoryPoint, ...]):
            self._points = points

        @property
        def total_time(self) -> float:
            return self._points[-1].t if self._points else 0.0

        @property
        def start(self) -> TypingTuple[float, float, float]:
            p = self._points[0]
            return (p.x, p.y, p.z)

        @property
        def end(self) -> TypingTuple[float, float, float]:
            p = self._points[-1]
            return (p.x, p.y, p.z)

        def sample(self, t: float) -> TrajectoryPoint:
            if t <= 0:
                return self._points[0]
            if t >= self._points[-1].t:
                return self._points[-1]
            for i, pt in enumerate(self._points[:-1]):
                if self._points[i+1].t >= t:
                    return pt
            return self._points[-1]

        def sample_by_time(self, dt: float) -> List[TrajectoryPoint]:
            return list(self._points)

from . import algorithm
from .params import HermiteParams, HermiteParams3D


# =============================================================================
# Request types
# =============================================================================

@dataclass(frozen=True)
class SmootherRequest:
    """Input to 2D trajectory smoothers."""
    path: Path2D
    limits: Optional[KinematicLimits] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmootherRequest3D:
    """Input to 3D trajectory smoothers."""
    path: Path3D
    limits: Optional[KinematicLimits] = None
    options: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class BaseSmoother(Protocol):
    """Smoother protocol (2D): Path2D → Trajectory."""
    name: str

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        ...


@runtime_checkable
class BaseSmoother3D(Protocol):
    """Smoother protocol (3D): Path3D → Trajectory."""
    name: str

    def smooth(self, request: SmootherRequest3D, world: Any = None) -> Trajectory:
        ...


# =============================================================================
# Implementations
# =============================================================================

class HermiteSmoother:
    """
    G1-continuous trajectory smoother using cubic Hermite splines (2D).

    Example:
        >>> smoother = HermiteSmoother()
        >>> trajectory = smoother.smooth(SmootherRequest(path=my_path))
    """
    name: str = "hermite"

    def __init__(self, params: Optional[HermiteParams] = None) -> None:
        self.params = params or HermiteParams()

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """Smooth 2D path into time-parameterized trajectory."""
        params = self._merge_options(request.options)

        solution = algorithm.solve(
            path=request.path,
            params=params,
            limits=request.limits,
        )

        return DiscreteTrajectory(points=solution.samples)

    def _merge_options(self, options: Dict[str, Any]) -> HermiteParams:
        """Override params with request options."""
        if not options:
            return self.params

        overrides = {k: v for k, v in options.items() if hasattr(self.params, k)}
        if not overrides:
            return self.params

        return HermiteParams(**{**self.params.__dict__, **overrides})


class HermiteSmoother3D:
    """
    G1-continuous trajectory smoother using cubic Hermite splines (3D).

    Example:
        >>> smoother = HermiteSmoother3D()
        >>> trajectory = smoother.smooth(SmootherRequest3D(path=my_path_3d))
    """
    name: str = "hermite_3d"

    def __init__(self, params: Optional[HermiteParams3D] = None) -> None:
        self.params = params or HermiteParams3D()

    def smooth(self, request: SmootherRequest3D, world: Any = None) -> Trajectory:
        """Smooth 3D path into time-parameterized trajectory."""
        params = self._merge_options(request.options)

        solution = algorithm.solve_3d(
            path=request.path,
            params=params,
            limits=request.limits,
        )

        return DiscreteTrajectory(points=solution.samples)

    def _merge_options(self, options: Dict[str, Any]) -> HermiteParams3D:
        """Override params with request options."""
        if not options:
            return self.params

        overrides = {k: v for k, v in options.items() if hasattr(self.params, k)}
        if not overrides:
            return self.params

        return HermiteParams3D(**{**self.params.__dict__, **overrides})