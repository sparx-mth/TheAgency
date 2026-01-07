"""
MinSnap smoother.

This module provides the `BaseSmoother` implementation that:
- receives `SmootherRequest` (path + optional limits/options)
- delegates trajectory generation to `MinSnapAlgorithm`
- returns a `Trajectory` implementation via `DiscreteTrajectory`
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.common.types.planning import Trajectory
from core.planning.interfaces.smoother import BaseSmoother, SmootherRequest

from core.planning.smoothers.adapter import DiscreteTrajectory
from .algorithm import MinSnapAlgorithm
from .params import MinSnapParams


class MinSnapSmoother(BaseSmoother):
    """Generate a minimum-snap trajectory from a geometric path."""
    name: str = "minsnap"

    def __init__(self, *, params: Optional[MinSnapParams] = None) -> None:
        self.params = params or MinSnapParams()
        self.params.validate()
        self._algo = MinSnapAlgorithm(params=self.params)

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        """Return a time-parameterized trajectory for the requested path."""
        options: Dict[str, Any] = dict(request.options or {})
        raw = self._algo.solve(
            path=request.path,
            limits=request.limits,
            options=options,
            world=world,
        )
        return DiscreteTrajectory(points=raw.samples)
