"""Cubic Hermite spline trajectory smoother (2D and 3D)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from sparx_agency.core.common.types import Trajectory
from sparx_agency.core.planning.interfaces.smoother import (
    SmootherRequest,
    SmootherRequest3D,
)
from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory

from . import algorithm
from .params import HermiteParams, HermiteParams3D


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