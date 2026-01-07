from __future__ import annotations

"""
Convenience façade for your higher-level app:
- create pipelines per robot
- feed observations
- pull costmaps
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .map import Observation
from .multi_robot.manager import MultiRobotManager
from .multi_robot.robot_context import RobotContext
from .pipeline.mapping_pipeline import MappingPipeline, MappingPipelineConfig
from .costmap.probabilistic_grid import ProbabilisticGridCostmap, ProbabilisticGridConfig
from .costmap.log_odds_grid import LogOddsGridCostmap, LogOddsGridConfig
from .interfaces.depth_model import DepthModel


@dataclass
class RobotPipelineDefaults:
    use_log_odds: bool = True
    resolution_m: float = 0.3
    size_m: float = 40.0


def build_pipeline_for_robot(
    robot_id: str,
    depth_model: Optional[DepthModel],
    defaults: RobotPipelineDefaults,
) -> RobotContext:
    if defaults.use_log_odds:
        costmap = LogOddsGridCostmap(
            LogOddsGridConfig(resolution_m=defaults.resolution_m, size_m=defaults.size_m, frame_id="map")
        )
    else:
        costmap = ProbabilisticGridCostmap(
            ProbabilisticGridConfig(resolution_m=defaults.resolution_m, size_m=defaults.size_m, frame_id="map")
        )

    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=None,
        cfg=MappingPipelineConfig(stride=2),
    )

    return RobotContext(robot_id=robot_id, pipeline=pipeline)


class MultiRobotMapping:
    def __init__(self):
        self.manager = MultiRobotManager()

    def add_robot(self, ctx: RobotContext) -> None:
        self.manager.add_robot(ctx)

    def step(self, robot_id: str, obs: Observation) -> None:
        self.manager.step(robot_id, obs)

    def get_costmap(self, robot_id: str):
        return self.manager.get_costmap(robot_id)
