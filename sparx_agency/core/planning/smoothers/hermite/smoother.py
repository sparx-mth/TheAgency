"""Cubic Hermite spline trajectory smoother."""
from __future__ import annotations

from typing import Any, Dict, Optional

from sparx_agency.core.common.types import Trajectory
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory

from . import algorithm
from .params import HermiteParams


class HermiteSmoother:
    """
    G1-continuous trajectory smoother using cubic Hermite splines.

    Produces smooth trajectories that pass through all waypoints with
    tangent continuity. Suitable for ground robots and slow-moving drones.

    Example:
        >>> smoother = HermiteSmoother()
        >>> trajectory = smoother.smooth(SmootherRequest(path=my_path))
    """
    name: str = "hermite"

    def __init__(self, params: Optional[HermiteParams] = None) -> None:
        self.params = params or HermiteParams()

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """Smooth path into time-parameterized trajectory."""
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

        overrides = {
            k: v for k, v in options.items()
            if hasattr(self.params, k)
        }

        if not overrides:
            return self.params

        return HermiteParams(**{**self.params.__dict__, **overrides})