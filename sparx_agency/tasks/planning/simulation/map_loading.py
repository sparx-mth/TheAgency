"""
Map loading utilities for drone simulation.

Provides PGM map loading and ObstacleMap class for defining environments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import yaml
import numpy as np
import matplotlib.pyplot as plt

# Environment / Costmap (from core)
from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.occupancy import occupancy_from_grayscale, OccupancyThresholds
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams


def load_pgm_map(pgm_path: str, yaml_path: str, inflate_radius: float = 0.1) -> Costmap2D:
    """
    Load PGM map with YAML metadata (same as run_pipeline.py).

    Args:
        pgm_path: Path to .pgm file
        yaml_path: Path to .yaml metadata file
        inflate_radius: Obstacle inflation radius (meters)

    Returns:
        Costmap2D ready for planning
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    resolution = config['resolution']
    origin = config['origin']

    img = plt.imread(pgm_path)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    thresholds = OccupancyThresholds(
        occupied_if_below=249,
        free_if_above=250,
        unknown_as_occupied=True
    )
    occupancy = occupancy_from_grayscale(img, thresholds)

    if inflate_radius > 0:
        occupancy = inflate_occupancy(
            occupancy,
            resolution=resolution,
            params=InflationParams(radius_m=inflate_radius)
        )

    clearance = compute_clearance_field(
        occupancy,
        resolution=resolution,
        params=DistanceFieldParams()
    )

    params = CostmapParams(
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        frame_id="map"
    )
    return Costmap2D(occupancy, params, clearance=clearance)


@dataclass
class ObstacleMap:
    """Simple obstacle map for defining the environment."""
    width: float = 10.0
    height: float = 10.0
    origin_x: float = -5.0
    origin_y: float = -5.0
    resolution: float = 0.02  # Resolution for RRT costmap (collision detection is exact)

    rectangles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)

    def add_rectangle(self, x: float, y: float, w: float, h: float):
        self.rectangles.append((x, y, w, h))

    def add_circle(self, cx: float, cy: float, r: float):
        self.circles.append((cx, cy, r))

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        for rx, ry, rw, rh in self.rectangles:
            # Find closest point on rectangle to (x, y)
            closest_x = max(rx, min(x, rx + rw))
            closest_y = max(ry, min(y, ry + rh))
            # Check distance to closest point (handles corners correctly)
            dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
            if dist <= margin:
                return True
        for cx, cy, r in self.circles:
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r + margin:
                return True
        return False

    def to_costmap(self, inflate_radius: float = 0.1) -> Costmap2D:
        """Convert obstacle map to Costmap2D for core planners."""
        # Create occupancy grid
        nx = int(self.width / self.resolution)
        ny = int(self.height / self.resolution)
        occupancy = np.zeros((ny, nx), dtype=np.uint8)

        # Fill occupancy grid
        for iy in range(ny):
            for ix in range(nx):
                x = self.origin_x + ix * self.resolution
                y = self.origin_y + iy * self.resolution
                if self.is_occupied(x, y):
                    occupancy[iy, ix] = 255  # Occupied

        # Inflate obstacles
        if inflate_radius > 0:
            occupancy = inflate_occupancy(
                occupancy,
                resolution=self.resolution,
                params=InflationParams(radius_m=inflate_radius)
            )

        # Compute clearance field
        clearance = compute_clearance_field(
            occupancy,
            resolution=self.resolution,
            params=DistanceFieldParams()
        )

        # Create costmap
        params = CostmapParams(
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id="map"
        )

        return Costmap2D(occupancy, params, clearance=clearance)