"""
Obstacle map with static obstacles and click-to-place obstacles.

SIMPLIFIED: Removed all dynamic/moving obstacle logic.
Click-placed obstacles are static - they stay where you put them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams


@dataclass
class PlacedObstacle:
    """A click-placed obstacle (static circle)."""
    cx: float
    cy: float
    r: float
    id: int = 0


@dataclass
class ObstacleMap:
    """
    Obstacle map with:
    - Static rectangles and circles (from config)
    - Click-placed obstacles (added at runtime, but static)
    """
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.02

    # Static obstacles from config
    rectangles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)

    # Click-placed obstacles (runtime)
    placed_obstacles: List[PlacedObstacle] = field(default_factory=list)
    _next_id: int = 1

    def add_rectangle(self, x: float, y: float, w: float, h: float) -> None:
        """Add a static rectangle obstacle."""
        self.rectangles.append((x, y, w, h))

    def add_circle(self, cx: float, cy: float, r: float) -> None:
        """Add a static circle obstacle."""
        self.circles.append((cx, cy, r))

    def place_obstacle(self, cx: float, cy: float, r: float) -> Optional[int]:
        """
        Place a new obstacle at runtime (click-to-place).
        Returns the obstacle ID, or None if max count reached.
        """
        if len(self.placed_obstacles) >= 50:  # Max limit
            return None

        obs = PlacedObstacle(
            cx=float(cx),
            cy=float(cy),
            r=float(r),
            id=self._next_id,
        )
        self._next_id += 1
        self.placed_obstacles.append(obs)
        return obs.id

    def remove_obstacle_by_id(self, obs_id: int) -> bool:
        """Remove a placed obstacle by ID."""
        for i, o in enumerate(self.placed_obstacles):
            if o.id == obs_id:
                self.placed_obstacles.pop(i)
                return True
        return False

    def clear_placed_obstacles(self) -> None:
        """Remove all click-placed obstacles."""
        self.placed_obstacles.clear()

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is occupied (within margin of any obstacle)."""
        # Static rectangles
        for rx, ry, rw, rh in self.rectangles:
            closest_x = max(rx, min(x, rx + rw))
            closest_y = max(ry, min(y, ry + rh))
            if math.hypot(x - closest_x, y - closest_y) <= margin:
                return True

        # Static circles
        for cx, cy, r in self.circles:
            if math.hypot(x - cx, y - cy) <= r + margin:
                return True

        # Placed obstacles
        for o in self.placed_obstacles:
            if math.hypot(x - o.cx, y - o.cy) <= o.r + margin:
                return True

        return False

    def get_obstacle_at(self, x: float, y: float) -> Optional[int]:
        """
        Get the ID of a placed obstacle at the given point.
        Returns None if no placed obstacle is at that point.
        """
        for o in self.placed_obstacles:
            if math.hypot(x - o.cx, y - o.cy) <= o.r:
                return o.id
        return None

    def to_costmap(self, inflate_radius: float = 0.1, include_placed: bool = True) -> Costmap2D:
        """
        Convert to a Costmap2D for planning.

        Args:
            inflate_radius: Inflation radius for obstacles
            include_placed: Whether to include click-placed obstacles
        """
        nx = int(self.width / self.resolution)
        ny = int(self.height / self.resolution)
        occupancy = np.zeros((ny, nx), dtype=np.uint8)

        for iy in range(ny):
            for ix in range(nx):
                x = self.origin_x + ix * self.resolution
                y = self.origin_y + iy * self.resolution
                if self._is_occupied_for_costmap(x, y, include_placed=include_placed):
                    occupancy[iy, ix] = 255

        if inflate_radius > 0.0:
            occupancy = inflate_occupancy(
                occupancy,
                resolution=self.resolution,
                params=InflationParams(radius_m=inflate_radius),
            )

        clearance = compute_clearance_field(
            occupancy,
            resolution=self.resolution,
            params=DistanceFieldParams(),
        )

        params = CostmapParams(
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id="map",
        )
        return Costmap2D(occupancy, params, clearance=clearance)

    def _is_occupied_for_costmap(self, x: float, y: float, include_placed: bool) -> bool:
        """Check occupancy for costmap generation."""
        # Static rectangles
        for rx, ry, rw, rh in self.rectangles:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True

        # Static circles
        for cx, cy, r in self.circles:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True

        # Placed obstacles (optional)
        if include_placed:
            for o in self.placed_obstacles:
                if (x - o.cx) ** 2 + (y - o.cy) ** 2 <= o.r ** 2:
                    return True

        return False