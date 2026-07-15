"""Route-aware replanning support (ROS-free, 3.8-compatible).

Pure primitives a planner node composes to decide *when* to replan a global A*
route and *whether* to adopt the new one, so a slow stop-and-turn platform is not
whipsawed by a route that changes every map frame:

  * :mod:`path_raster` -- rasterize the current path and build its corridor mask.
  * :mod:`map_change` -- count newly-observed cells that fall in that corridor
    (the "I turned and discovered a lot of relevant area" trigger).
  * :mod:`path_metrics` -- remaining-route length + candidate length for the
    adopt/keep hysteresis (only swap for a meaningfully shorter route).

There is deliberately no policy *object* here: the ROS node already owns the
committed path, the commit time and the collision streak, so it orchestrates
these functions directly rather than duplicating that state.
"""
from .map_change import (
    count_new_known_in_corridor,
    known_mask,
    newly_known_mask,
)
from .obstacle_confidence import route_obstacle_confidence
from .path_metrics import polyline_length, remaining_polyline
from .path_raster import corridor_mask, rasterize_path
from .route_difficulty import (
    RouteDifficulty,
    assess_route_difficulty,
    forward_window_2d,
    net_turn_deg,
    passage_free_width_2d,
    passage_widths_2d,
    windowed_turn_deg,
)

__all__ = [
    "RouteDifficulty",
    "assess_route_difficulty",
    "corridor_mask",
    "count_new_known_in_corridor",
    "forward_window_2d",
    "known_mask",
    "net_turn_deg",
    "newly_known_mask",
    "passage_free_width_2d",
    "passage_widths_2d",
    "polyline_length",
    "rasterize_path",
    "remaining_polyline",
    "route_obstacle_confidence",
    "windowed_turn_deg",
]
