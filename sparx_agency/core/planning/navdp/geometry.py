"""Camera/body geometry for the NavDP point-goal policy (ROS-free, numpy-only).

This module owns the *pure geometry* that surrounds NavDP and contains neither
networking nor ROS:

* :func:`pixel_to_pointgoal` -- a clicked image pixel + its depth -> the
  body-frame point-goal NavDP consumes, scaled into NavDP's input range while
  preserving the click bearing.
* :func:`anchor_trajectory_to_world` -- NavDP's body-frame trajectory anchored at
  the drone's world pose, ready to publish as a world-frame path (the BEV map /
  waypoint-follower frame).
* :func:`project_trajectory_to_pixels` / :func:`body_point_to_pixel` -- body-frame
  waypoints projected back onto the image for the operator overlay.

Frame conventions (NavDP / drone body, FLU)
-------------------------------------------
body
    ``+x`` forward, ``+y`` left, ``+z`` up; the drone is the origin.
camera
    the forward camera's optical axis is taken parallel to body ``+x``, so a pixel
    ``(u, v)`` at metric depth ``d`` (optical Z) back-projects to::

        forward = d
        left    = -(u - cx) * d / fx
        up      = -(v - cy) * d / fy

    The small camera mount offset is neglected (matching the reference NavDP
    integration). The click goal and the returned trajectory therefore share one
    camera-centric frame and stay mutually consistent: anchoring both at the drone
    body introduces the same constant offset and cancels out in the relative
    geometry the policy and follower care about.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8): no
PEP 604 unions, no ``match``/``case``. numpy-only at import time.
"""
from __future__ import annotations

from math import cos, sin

import numpy as np

from sparx_agency.core.common.types import Intrinsics

# NavDP input range: forward in ``[0, NAVDP_MAX_FWD_M]`` m, lateral in
# ``[-NAVDP_MAX_LAT_M, NAVDP_MAX_LAT_M]`` m. Goals beyond this are scaled as a
# whole (not clipped per-axis) so the bearing to the click is preserved.
NAVDP_MAX_FWD_M = 10.0
NAVDP_MAX_LAT_M = 10.0


def patch_median_depth(depth, px, py, half=10, min_valid=0.1, max_valid=50.0):
    """Median of the valid depth in a ``2*half`` box around ``(px, py)``.

    Args:
        depth: HxW array of metric depth (optical Z), NaN/0 allowed.
        px, py: pixel column/row at the box centre.
        half: half box size in pixels.
        min_valid, max_valid: a reading counts only if ``min_valid < d < max_valid``.

    Returns:
        The median valid depth (float), or ``None`` if the patch holds no valid
        reading -- the caller decides the fallback.
    """
    h, w = depth.shape
    patch = depth[max(0, int(py) - half):min(h, int(py) + half),
                  max(0, int(px) - half):min(w, int(px) + half)]
    valid = patch[np.isfinite(patch) & (patch > min_valid) & (patch < max_valid)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_pointgoal(px, py, depth, intr, fallback_depth_m=3.0,
                       max_fwd_m=NAVDP_MAX_FWD_M, max_lat_m=NAVDP_MAX_LAT_M):
    """Convert a clicked pixel + depth into NavDP's body-frame point-goal.

    Args:
        px, py: clicked pixel ``(column, row)`` on the RGB image.
        depth: HxW float metric depth (optical Z), aligned to the RGB image.
        intr: camera :class:`Intrinsics` matching the RGB/depth stream.
        fallback_depth_m: depth assumed when the patch has no valid reading.
        max_fwd_m, max_lat_m: NavDP input-range bounds.

    Returns:
        ``(gx, gy, d, bz)``:

        * ``gx, gy`` -- body-frame point-goal ``(forward, left)`` scaled uniformly
          so ``gx <= max_fwd_m`` and ``|gy| <= max_lat_m`` while preserving
          ``atan2(gy, gx)`` -- the bearing the drone would steer. Per-axis clipping
          is deliberately avoided: a click past ``max_fwd_m`` would otherwise keep
          its full lateral offset and look far more sideways than intended.
        * ``d`` -- raw depth at the click (m).
        * ``bz`` -- vertical body offset of the click (``+`` up). Never sent to
          NavDP (a 2D ground-plane policy); returned only for the operator readout.
    """
    d = patch_median_depth(depth, px, py)
    if d is None:
        d = float(fallback_depth_m)

    bx_raw = d
    by_raw = -(px - intr.cx) * d / intr.fx
    bz = -(py - intr.cy) * d / intr.fy

    if bx_raw < 0.1:                       # click on/behind the camera plane
        return 0.1, 0.0, d, bz
    scale_fwd = max_fwd_m / bx_raw if bx_raw > max_fwd_m else 1.0
    scale_lat = max_lat_m / abs(by_raw) if abs(by_raw) > max_lat_m else 1.0
    scale = min(scale_fwd, scale_lat)
    return float(bx_raw * scale), float(by_raw * scale), d, bz


def anchor_trajectory_to_world(traj_xy, ref_x, ref_y, ref_yaw):
    """Anchor NavDP's body-frame trajectory at a world pose (SE(2) rigid map).

    NavDP returns waypoints relative to the drone *at inference time* (the drone is
    the origin, facing ``+x``). Rotating by ``ref_yaw`` and translating by
    ``(ref_x, ref_y)`` expresses them in the world / BEV frame::

        world_x = ref_x + forward * cos(ref_yaw) - left * sin(ref_yaw)
        world_y = ref_y + forward * sin(ref_yaw) + left * cos(ref_yaw)

    Anchor with the pose captured co-temporally with the RGB-D frame that produced
    the trajectory, so localization drift after inference does not corrupt it.

    Args:
        traj_xy: ``(T, >=2)`` array-like of ``(forward, left)`` body waypoints
            (extra columns such as yaw are ignored).
        ref_x, ref_y, ref_yaw: drone world pose at inference time.

    Returns:
        ``list[(world_x, world_y)]`` of length ``T``.
    """
    c, s = cos(ref_yaw), sin(ref_yaw)
    out = []
    for wp in traj_xy:
        fwd, left = float(wp[0]), float(wp[1])
        out.append((ref_x + fwd * c - left * s,
                    ref_y + fwd * s + left * c))
    return out


def body_point_to_pixel(x_fwd, y_left, intr, cam_height_m, min_fwd_m=0.05,
                        clamp=8000):
    """Project a body-frame ground point onto the image.

    Models the waypoint as lying on a ground plane ``cam_height_m`` below the
    camera, with the optical axis along body ``+x``::

        u = fx * (-y_left)    / x_fwd + cx
        v = fy *  cam_height_m / x_fwd + cy

    The line lands on the *true* floor when ``cam_height_m`` equals the camera's
    real height, so callers should pass the live altitude for a floor-accurate
    overlay. A smaller fixed value (e.g. NavDP's ~0.5 m training height) is an
    optional visualization trick that keeps more near waypoints in-frame at the
    cost of floor alignment -- a drone flying at ~1 m otherwise pushes the first
    several waypoints below the image.

    Args:
        x_fwd, y_left: body-frame waypoint (m).
        intr: camera :class:`Intrinsics`.
        cam_height_m: ground-plane height below the camera used for the render.
        min_fwd_m: points closer than this in forward range are dropped.
        clamp: pixel coordinates are clamped to ``+-clamp`` so ``cv2.line`` clips
            partial segments at the image edge instead of discarding them.

    Returns:
        ``(u, v)`` int pixel, or ``None`` if the point is at/behind the camera.
    """
    if x_fwd < min_fwd_m:
        return None
    u = intr.fx * (-y_left) / x_fwd + intr.cx
    v = intr.fy * cam_height_m / x_fwd + intr.cy
    return (int(np.clip(u, -clamp, clamp)), int(np.clip(v, -clamp, clamp)))


def project_trajectory_to_pixels(traj_xy, intr, cam_height_m):
    """Project every body-frame ``(forward, left)`` waypoint onto the image.

    Returns a list (same length as ``traj_xy``) of ``(u, v)`` pixels or ``None``
    for waypoints at/behind the camera. See :func:`body_point_to_pixel`.
    """
    return [body_point_to_pixel(float(p[0]), float(p[1]), intr, cam_height_m)
            for p in traj_xy]
