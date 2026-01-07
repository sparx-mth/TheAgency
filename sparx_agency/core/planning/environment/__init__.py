"""
Planning environment.

This package provides map/costmap representations and collision checking helpers
used by planners and smoothers.
"""

from .costmap2d import Costmap2D, CostmapParams
from .collision import (
    is_state_valid,
    is_segment_collision_free,
    segment_collision_ratio,
)
