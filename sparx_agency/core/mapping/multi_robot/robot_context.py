from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline
from sparx_agency.core.common.types import Observation


@dataclass
class RobotContext:
    robot_id: str
    pipeline: MappingPipeline
    last_obs: Optional[Observation] = None
    last_update_sec: Optional[float] = None

    def step(self, obs: Observation) -> None:
        self.pipeline.step(obs)
        self.last_obs = obs
        # best-effort stamp
        if obs.cloud is not None:
            self.last_update_sec = obs.cloud.stamp_sec
        elif obs.depth is not None:
            self.last_update_sec = obs.depth.stamp_sec
        elif obs.rgb is not None:
            self.last_update_sec = obs.rgb.stamp_sec
