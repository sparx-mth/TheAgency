"""
Map representation adapters and converters.

This package contains utilities that convert between different map
representations (e.g., OccupancyGrid2D -> Costmap2D) or expose a compatible
view required by planners or behaviors.

No collision or safety logic should live here.
"""

from .occupancy_to_costmap2d import costmap_from_occupancy_grid

__all__ = [
    "costmap_from_occupancy_grid",
]
