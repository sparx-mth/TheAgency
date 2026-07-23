"""NavDP point-goal navigation policy: ROS-free geometry + HTTP client.

NavDP (`InternRobotics/NavDP <https://github.com/InternRobotics/NavDP>`_) is a
point-goal navigation diffusion policy. Given an RGB-D frame and a 2D goal in the
robot body frame it returns a trajectory of relative waypoints in that same
frame. This package owns the two ROS-free pieces our stack needs around it:

* :mod:`~sparx_agency.core.planning.vlas.navdp.geometry` -- the pure camera/body
  geometry (clicked pixel -> body-frame point-goal, body-frame trajectory ->
  world path, body-frame trajectory -> image pixels for the overlay).
* :mod:`~sparx_agency.core.planning.vlas.navdp.client` -- the HTTP wire contract with
  the NavDP server (the single integration seam; no ROS, no geometry).

The ROS node that drives them (camera window, click handling, publishing the
world path for the waypoint follower) lives in the FALCON task adapter, mirroring
how ``astar``/``waypoint_follower`` keep their algorithm in core and their thin
ROS shell in ``tasks/planning/falcon``.
"""
from .client import DEPTH_SCALE, NavDPError, NavDPPointgoalClient
from .geometry import (
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    anchor_trajectory_to_world,
    body_point_to_pixel,
    patch_median_depth,
    pixel_to_pointgoal,
    point_to_pointgoal,
    project_trajectory_to_pixels,
    world_to_body_2d,
)
from .local_goal import (
    LocalGoal,
    point_visible,
    select_farthest_visible_waypoint,
)

__all__ = [
    "NAVDP_MAX_FWD_M",
    "NAVDP_MAX_LAT_M",
    "patch_median_depth",
    "pixel_to_pointgoal",
    "point_to_pointgoal",
    "world_to_body_2d",
    "anchor_trajectory_to_world",
    "body_point_to_pixel",
    "project_trajectory_to_pixels",
    "LocalGoal",
    "point_visible",
    "select_farthest_visible_waypoint",
    "DEPTH_SCALE",
    "NavDPError",
    "NavDPPointgoalClient",
]
