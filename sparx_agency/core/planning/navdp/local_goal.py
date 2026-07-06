"""Pick the farthest line-of-sight-visible A* waypoint as a NavDP point-goal.

This is the bridge between a global A* route and the local NavDP point-goal
policy, used by the FALCON *combination* navigation mode: A* plans a
collision-free route to the mission goal, and at each step the drone hands NavDP
a LOCAL goal -- the farthest A* waypoint actually visible in the current camera
frame -- so NavDP returns a locally-grounded trajectory toward it. Flying NavDP
to (about) the midpoint and re-selecting keeps NavDP in its accurate near field
while the A* route supplies the global direction.

An A* waypoint is a *ground* location, so it is projected onto the floor plane
``cam_height_m`` below the camera (the live altitude) -- the test is therefore
ground-plane line-of-sight: where the waypoint's footprint appears in the image,
and whether the floor is clear out to it. "Visible", for a waypoint in the drone
body frame (FLU; see :mod:`~sparx_agency.core.planning.navdp.geometry`), means:

* it is in front of the camera (forward range ``>= min_fwd_m``);
* its ground-plane projection lands inside the image; and
* (optionally) the measured depth along that pixel ray is no nearer than the
  waypoint -- i.e. no wall stands between the camera and it. This FAILS CLOSED:
  a pixel with no valid depth (beyond range, or a no-return surface) is treated
  as NOT clear, so a goal is never placed past an unmeasured near occluder.

Each waypoint is classified ``visible`` / ``near`` / ``blocked``. Selection walks
the route outward, skipping the ``near`` prefix the drone is already on top of
(waypoints that project below the frame), and keeps the farthest waypoint of the
first UNBROKEN run of ``visible`` waypoints. A ``blocked`` waypoint (off to the
side past a turn, above, or occluded) ENDS the run -- and, if hit before any
waypoint is visible, stops the scan rather than leaping past it -- so the local
goal always sits on the clear stretch of corridor directly ahead.

ROS-free, numpy-consuming, and Python 3.8 compatible (the FALCON Noetic adapter
imports ``core`` under 3.8): no PEP 604 unions, no ``match``/``case``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sparx_agency.core.common.types import Intrinsics

from .geometry import (
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    body_point_to_pixel,
    patch_median_depth,
    point_to_pointgoal,
    world_to_body_2d,
)


# Visibility classes for a body-frame waypoint.
_NEAR = "near"          # below the frame / behind the camera -> drone is on top of it
_VISIBLE = "visible"    # in front, in-frame, and (if required) unoccluded
_BLOCKED = "blocked"    # off to the side/above the frame, or occluded -> view ends here


@dataclass(frozen=True)
class LocalGoal:
    """A selected A* waypoint, expressed for NavDP.

    Attributes:
        index: Index of the chosen waypoint in the input sequence.
        world: Waypoint in the world frame ``(x, y)``.
        body: Waypoint in the drone body frame ``(forward, left)`` (m, FLU).
        goal: NavDP point-goal ``(gx, gy)`` -- ``body`` scaled into NavDP's input
            range with the bearing preserved. Pass straight to ``pointgoal_step``.
    """

    index: int
    world: Tuple[float, float]
    body: Tuple[float, float]
    goal: Tuple[float, float]


def _classify_waypoint(fwd, left, depth, intr, cam_height_m, require_unoccluded,
                       depth_tol_m, depth_patch_half, min_fwd_m):
    """Classify a body-frame ground point as ``near`` / ``visible`` / ``blocked``.

    Ground-plane projection (point on the floor ``cam_height_m`` below the camera,
    optical axis along body ``+x`` -- the same model as :func:`body_point_to_pixel`).
    ``near`` = the drone is on top of it (behind the camera or below the frame);
    ``blocked`` = off to the side/above the frame, or occluded by a nearer surface;
    ``visible`` = in front, in-frame, and (when required) with clear line of sight.
    The occlusion test fails closed: an invalid depth reading at the pixel counts
    as ``blocked`` (a goal is never placed past an unmeasured near occluder).
    """
    if fwd < min_fwd_m:
        return _NEAR                          # at/behind the camera plane
    px = body_point_to_pixel(fwd, left, intr, cam_height_m, min_fwd_m=min_fwd_m)
    if px is None:
        return _NEAR
    u, v = px
    if v >= intr.height:
        return _NEAR                          # below the frame: the drone is above it
    if u < 0 or u >= intr.width or v < 0:
        return _BLOCKED                       # off to the side / above: the view ends here
    if require_unoccluded:
        d = patch_median_depth(depth, u, v, half=depth_patch_half)
        if d is None or fwd > d + depth_tol_m:
            return _BLOCKED                    # unmeasurable or occluded -> not clear
    return _VISIBLE


def point_visible(fwd, left, depth, intr, cam_height_m, require_unoccluded=True,
                  depth_tol_m=0.5, depth_patch_half=6, min_fwd_m=0.2):
    """Is a body-frame ground point visible in the current camera frame?

    A convenience boolean wrapper over the ``near``/``visible``/``blocked``
    classification: ``True`` only for ``visible`` (in front of the camera,
    projecting inside the image, and -- when ``require_unoccluded`` -- with a
    clear, *measured* line of sight). See the module docstring for the model.

    Args:
        fwd, left: body-frame point ``(forward, left)`` in meters (FLU).
        depth: HxW float metric depth (optical Z), aligned to the camera.
        intr: camera :class:`Intrinsics` matching ``depth``.
        cam_height_m: camera height above the ground plane the point lies on.
        require_unoccluded: also require clear line of sight (fails closed on an
            invalid depth reading). When False only the in-frame test applies.
        depth_tol_m: slack (m) on the occlusion comparison (depth noise + the
            ground-plane approximation).
        depth_patch_half: half-size (px) of the depth patch median at the pixel.
        min_fwd_m: points closer than this in forward range are never visible.
    """
    return _classify_waypoint(fwd, left, depth, intr, cam_height_m,
                              require_unoccluded, depth_tol_m, depth_patch_half,
                              min_fwd_m) == _VISIBLE


def select_farthest_visible_waypoint(
        waypoints, ref_x, ref_y, ref_yaw, depth, intr, cam_height_m,
        require_unoccluded=True, depth_tol_m=0.5, depth_patch_half=6,
        min_fwd_m=0.2, max_fwd_m=NAVDP_MAX_FWD_M, max_lat_m=NAVDP_MAX_LAT_M):
    """Farthest route waypoint reachable through an unbroken run of visible ones.

    Walks ``waypoints`` in route order (drone -> goal). The ``near`` prefix the
    drone is on top of (waypoints projecting below the frame) is skipped; the
    first ``visible`` waypoint starts a run that continues only while each
    successive waypoint stays ``visible``; a ``blocked`` waypoint ends it. A
    ``blocked`` waypoint hit BEFORE any is visible (the route turns out of view or
    a wall fills the frame) stops the scan, so the goal never leaps past a turn or
    an occluder. The farthest waypoint of the run is returned.

    Args:
        waypoints: ordered world-frame route as a sequence of ``(x, y)`` (the A*
            ``nav_msgs/Path`` points; the first ~= the drone).
        ref_x, ref_y, ref_yaw: drone world pose (the body-frame origin/heading).
        depth: HxW float metric depth (optical Z), aligned to the camera.
        intr: camera :class:`Intrinsics` matching ``depth``.
        cam_height_m: camera height above the ground plane (the live altitude).
        require_unoccluded, depth_tol_m, depth_patch_half, min_fwd_m: visibility
            knobs -- see :func:`point_visible`.
        max_fwd_m, max_lat_m: NavDP input-range bounds for the returned goal.

    Returns:
        The chosen :class:`LocalGoal`, or ``None`` if no waypoint is visible
        (the drone faces a near wall or the route turns out of view) -- the caller
        should then fall back to the A* path.
    """
    chosen = None       # type: Optional[LocalGoal]
    started = False
    for i, wp in enumerate(waypoints):
        wx, wy = float(wp[0]), float(wp[1])
        fwd, left = world_to_body_2d(wx, wy, ref_x, ref_y, ref_yaw)
        cls = _classify_waypoint(fwd, left, depth, intr, cam_height_m,
                                 require_unoccluded, depth_tol_m,
                                 depth_patch_half, min_fwd_m)
        if cls == _VISIBLE:
            gx, gy = point_to_pointgoal(fwd, left, max_fwd_m, max_lat_m)
            chosen = LocalGoal(index=i, world=(wx, wy), body=(fwd, left),
                               goal=(gx, gy))
            started = True
        elif started:
            break                  # the visible run ended -> keep its farthest point
        elif cls == _BLOCKED:
            break                  # view ends before any visible waypoint -> no leap
        # else: cls == _NEAR and not started -> skip the near prefix, scan outward
    return chosen
