from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from sparx_agency.core.common.types import Observation
from sparx_agency.core.mapping.interfaces.costmap import GridSpec
from sparx_agency.core.mapping.multi_robot.robot_context import RobotContext


class MultiRobotManager:
    """
    Holds multiple RobotContext objects.
    One costmap per robot by default.
    """

    def __init__(self):
        self._robots: Dict[str, RobotContext] = {}

    def add_robot(self, ctx: RobotContext) -> None:
        self._robots[ctx.robot_id] = ctx

    def has_robot(self, robot_id: str) -> bool:
        return robot_id in self._robots

    def step(self, robot_id: str, obs: Observation) -> None:
        if robot_id not in self._robots:
            raise KeyError(f"Robot '{robot_id}' not registered in MultiRobotManager")
        self._robots[robot_id].step(obs)

    def get_costmap(self, robot_id: str) -> Tuple[GridSpec, np.ndarray]:
        if robot_id not in self._robots:
            raise KeyError(f"Robot '{robot_id}' not registered")
        return self._robots[robot_id].pipeline.costmap.get_grid()

    def robots(self):
        return list(self._robots.keys())
