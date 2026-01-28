"""
Behavior utility functions.

This subpackage contains shared utilities for behavior implementations:
- path_utils: Path manipulation helpers (trimming, subgoal selection)
- world_adapters: World representation converters (OccupancyGrid2D -> Costmap2D)
"""

from .path_utils import pick_subgoal_along_path, trim_path_prefix
from sparx_agency.core.planning.integrations.maps.occupancy_to_costmap2d import costmap_from_occupancy_grid

__all__ = [
    "costmap_from_occupancy_grid",
    "pick_subgoal_along_path",
    "trim_path_prefix",
]