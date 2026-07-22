"""
Obstacle map for drone simulation.
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams


@dataclass
class ObstacleMap:
    """Simple obstacle map with rectangles and circles."""
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.02

    rectangles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)

    def add_rectangle(self, x: float, y: float, w: float, h: float):
        self.rectangles.append((x, y, w, h))

    def add_circle(self, cx: float, cy: float, r: float):
        self.circles.append((cx, cy, r))

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        for rx, ry, rw, rh in self.rectangles:
            closest_x = max(rx, min(x, rx + rw))
            closest_y = max(ry, min(y, ry + rh))
            if math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2) <= margin:
                return True
        for cx, cy, r in self.circles:
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r + margin:
                return True
        return False

    def to_costmap(self, inflate_radius: float = 0.1) -> Costmap2D:
        """Convert to Costmap2D for planners."""
        nx, ny = int(self.width / self.resolution), int(self.height / self.resolution)
        occupancy = np.zeros((ny, nx), dtype=np.uint8)

        for iy in range(ny):
            for ix in range(nx):
                x = self.origin_x + ix * self.resolution
                y = self.origin_y + iy * self.resolution
                if self.is_occupied(x, y):
                    occupancy[iy, ix] = 255

        if inflate_radius > 0:
            occupancy = inflate_occupancy(occupancy, resolution=self.resolution, params=InflationParams(radius_m=inflate_radius))

        clearance = compute_clearance_field(occupancy, resolution=self.resolution, params=DistanceFieldParams())
        params = CostmapParams(resolution=self.resolution, origin_x=self.origin_x, origin_y=self.origin_y, frame_id="map")
        return Costmap2D(occupancy, params, clearance=clearance)