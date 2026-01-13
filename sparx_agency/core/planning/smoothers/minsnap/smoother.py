"""Minimum-snap trajectory smoother."""
from __future__ import annotations

from typing import Any, Dict, Optional

from sparx_agency.core.common.types import Trajectory
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory

from . import algorithm
from .params import MinSnapParams


class MinSnapSmoother:
    """
    Minimum-snap trajectory smoother.

    Generates polynomial trajectories that minimize snap (4th derivative),
    producing smooth accelerations ideal for quadrotors and agile robots.

    Implements BaseSmoother protocol.

    Requires: pip install minsnap-trajectories

    Example:
        >>> smoother = MinSnapSmoother()
        >>> trajectory = smoother.smooth(SmootherRequest(path=my_path))
    """
    name: str = "minsnap"

    def __init__(self, params: Optional[MinSnapParams] = None) -> None:
        """
        Initialize smoother.

        Args:
            params: Algorithm configuration. Uses defaults if None.
        """
        self.params = params or MinSnapParams()

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """
        Smooth path into minimum-snap trajectory.

        Args:
            request: Path and optional constraints.
            world: Optional environment context (unused).

        Returns:
            Discrete trajectory implementing Trajectory protocol.
        """
        params = self._merge_options(request.options)

        solution = algorithm.solve(
            path=request.path,
            params=params,
            limits=request.limits,
        )

        return DiscreteTrajectory(points=solution.samples)

    def _merge_options(self, options: Dict[str, Any]) -> MinSnapParams:
        """Override params with request options."""
        if not options:
            return self.params

        overrides = {
            k: v for k, v in options.items()
            if hasattr(self.params, k)
        }

        if not overrides:
            return self.params

        return MinSnapParams(**{**self.params.__dict__, **overrides})