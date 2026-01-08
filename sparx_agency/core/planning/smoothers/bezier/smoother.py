from __future__ import annotations

from typing import Any, Dict, Optional

from sparx_agency.core.common.types.planning import Trajectory
from sparx_agency.core.planning.interfaces.smoother import BaseSmoother, SmootherRequest

from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory
from .algorithm import BezierAlgorithm
from .params import BezierParams


class BezierSmoother(BaseSmoother):
    """Heading-aware Hermite/Bezier-like trajectory smoother."""
    name: str = "bezier"

    def __init__(self, *, params: Optional[BezierParams] = None) -> None:
        self.params = params or BezierParams()
        self.params.validate()
        self._algo = BezierAlgorithm(params=self.params)

    def smooth(self, request: SmootherRequest, world: Any = None) -> Trajectory:
        options: Dict[str, Any] = dict(request.options or {})
        sol = self._algo.solve(
            path=request.path,
            limits=request.limits,   # expects KinematicLimits-like (max_speed_xy)
            options=options,
            world=world,
        )
        return DiscreteTrajectory(points=sol.samples)
