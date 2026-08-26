"""Brake on what the depth camera sees RIGHT NOW, before the map knows it.

The voxel map is the planner's memory; this is the reflex. A thin obstacle
(a person, a pole, a door edge) can be visible in the raw depth image for
seconds while the voxel map still flaps it between occupied and free --
raycast clearing erodes one-voxel-thick silhouettes -- and an aircraft
braking only on the map flies through that gap in its knowledge. Measured
in the hospital campaign: every remaining strike of run 009 was one thin
human, approached four times, visible in raw depth at 2.8 m each time.

This class answers one question per frame: how fast may the aircraft fly given
the closest depth return inside the corridor it is about to sweep? Everything
else -- mapped obstacles behind the aircraft -- belongs to the voxel gate; the
two compose.

**The corridor follows the direction of TRAVEL, not the nose.** On a holonomic
platform those are different, and the difference is not academic: a drone
crossing a 0.93 m doorway with its nose 20 deg off swings a nose-aligned
corridor onto the jamb and reads "blocked" while the path it is actually flying
is clear. Measured in the hospital: a route drawn straight through the middle of
an opening, refused at 0.40 m, repeatedly. :meth:`allowed_speed_along` takes the
travel bearing; :meth:`allowed_forward_speed` is that with a bearing of zero and
is kept because a one-axis platform only ever has one.

Pure numpy 1.17 / Python 3.8; imported inside the Noetic FALCON container.
"""
from __future__ import annotations

import math

import numpy as np


class DepthProximityBrakeConfig(object):
    """Tunables for :class:`DepthProximityBrake`.

    Attributes:
        fx, fy, cx, cy: Pinhole intrinsics of the depth image.
        corridor_halfwidth_m: Lateral half-extent of the protected corridor
            (airframe radius plus a little).
        corridor_halfheight_m: Vertical half-extent around the camera axis.
        nose_offset_m: Camera-to-airframe-nose distance along the axis; the
            depth return is to the CAMERA, the strike happens at the nose.
        min_valid_m: Returns closer than this are sensor noise / self-view
            and are ignored (the camera's near clip sits above it anyway).
        brake_decel: Deceleration the airframe delivers, m/s^2.
        react_s: Command-to-plant latency budget.
        margin_m: Clearance kept beyond the stopping point.
        stride: Pixel subsampling stride (both axes) for speed.
    """

    def __init__(self,
                 fx=390.6427, fy=390.6427, cx=300.5, cy=300.5,
                 corridor_halfwidth_m=0.35,
                 corridor_halfheight_m=0.35,
                 nose_offset_m=0.10,
                 min_valid_m=0.15,
                 brake_decel=0.8,
                 react_s=0.30,
                 margin_m=0.70,
                 hard_block_d_m=1.05,
                 stride=4):
        # type: (...) -> None
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.corridor_halfwidth_m = float(corridor_halfwidth_m)
        self.corridor_halfheight_m = float(corridor_halfheight_m)
        self.nose_offset_m = float(nose_offset_m)
        self.min_valid_m = float(min_valid_m)
        self.brake_decel = float(brake_decel)
        self.react_s = float(react_s)
        self.margin_m = float(margin_m)
        self.hard_block_d_m = float(hard_block_d_m)
        self.stride = int(stride)


class DepthProximityBrake(object):
    """Corridor-minimum depth -> allowed forward speed."""

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or DepthProximityBrakeConfig()
        self._u = None      # cached per-shape pixel coordinate grids
        self._v = None
        self._shape = None

    def _grids(self, shape):
        # type: (tuple) -> tuple
        if shape != self._shape:
            s = self._cfg.stride
            vs = np.arange(0, shape[0], s, dtype=np.float32)
            us = np.arange(0, shape[1], s, dtype=np.float32)
            self._u, self._v = np.meshgrid(us, vs)
            self._shape = shape
        return self._u, self._v

    def corridor_min_depth(self, depth_m, bearing_rad=0.0):
        # type: (np.ndarray, float) -> object
        """Closest valid return inside the swept corridor, or None.

        A pixel is in the corridor when the point it back-projects to lies
        within ``corridor_halfwidth_m`` of the **travel ray** and within
        ``corridor_halfheight_m`` of the camera plane vertically -- i.e. inside
        the cross-section the airframe sweeps flying along ``bearing_rad``. The
        distance returned is measured ALONG that ray, not along the optical
        axis, so it is the range the aircraft actually has to stop in.

        At ``bearing_rad = 0`` this is exactly the old optical-axis test: the
        along-track distance is the depth and the cross-track offset is the
        lateral offset.

        Args:
            depth_m: ``(H, W)`` metric depth.
            bearing_rad: direction of travel, radians CCW from the nose (body
                FLU, so positive is left).
        """
        cfg = self._cfg
        s = cfg.stride
        d = depth_m[::s, ::s]
        u, v = self._grids(depth_m.shape)
        cos_b, sin_b = math.cos(float(bearing_rad)), math.sin(float(bearing_rad))
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(d) & (d > cfg.min_valid_m)
            # Body FLU: forward is the depth, left is the negated lateral offset.
            left = -(u - cfg.cx) * d / cfg.fx
            vert = np.abs(v - cfg.cy) * d / cfg.fy
            along = d * cos_b + left * sin_b
            cross = np.abs(-d * sin_b + left * cos_b)
            mask = (valid & (along > 0.0) & (cross <= cfg.corridor_halfwidth_m)
                    & (vert <= cfg.corridor_halfheight_m))
        if mask.any():
            return float(along[mask].min())
        # Nothing valid in the corridor. Two very different situations share
        # that symptom, and returning None for both is what turns this brake
        # OFF at exactly the range it exists for:
        #
        #   * the corridor is genuinely EMPTY -- no returns at all, nothing to
        #     brake for;
        #   * every return is CLOSER than ``min_valid_m`` -- the aircraft is up
        #     against something. On the SJTU drone the front face is at +0.26 m
        #     and the camera lens at +0.20 m, so a nose-on contact puts the wall
        #     0.06 m from the lens: below ``min_valid_m`` AND below the sensor's
        #     own 0.1 m near clip. Every pixel goes invalid at the one moment
        #     the answer must be "stop".
        #
        # So separate them: a corridor full of too-close returns reports
        # ``min_valid_m``, which the caller turns into a full stop.
        with np.errstate(invalid="ignore"):
            present = np.isfinite(d) & (d > 0.0)
            lat = np.abs(u - cfg.cx) * d / cfg.fx
            vert = np.abs(v - cfg.cy) * d / cfg.fy
            near = (present & (d <= cfg.min_valid_m)
                    & (lat <= cfg.corridor_halfwidth_m)
                    & (vert <= cfg.corridor_halfheight_m))
        if near.any():
            return float(cfg.min_valid_m)
        return None

    def horizontal_half_fov(self):
        # type: () -> float
        """Half the horizontal field of view, radians, from the intrinsics.

        Assumes a centred principal point, which is what every camera in this
        stack has; a strongly off-centre one would make this optimistic on one
        side, so it is the smaller of the two half-angles that matters and
        callers should keep a guard band.
        """
        return math.atan2(self._cfg.cx, self._cfg.fx)

    def sees_bearing(self, bearing_rad, guard_rad=0.17):
        # type: (float, float) -> bool
        """Can this frame say anything about travel along ``bearing_rad``?

        The corridor around a travel direction near the edge of the image is
        half outside it, and a corridor with nothing in it reports "clear" --
        so an uncertified bearing must never be answered with a speed. The
        guard band (10 deg by default) covers the corridor's own angular width.
        """
        return abs(float(bearing_rad)) <= self.horizontal_half_fov() - float(guard_rad)

    def allowed_speed_along(self, depth_m, bearing_rad=0.0):
        # type: (np.ndarray, float) -> tuple
        """``(v_allow, d_min, certified)``: max safe speed along a travel bearing.

        The general form of :meth:`allowed_forward_speed`. ``certified`` is
        False when ``bearing_rad`` falls outside what the camera can see, in
        which case ``v_allow`` is the answer for the **nose** instead -- a
        conservative stand-in, since a corridor the camera cannot observe must
        not come back "clear". A caller that gets ``certified=False`` should cap
        the command on its own account as well.

        Args:
            depth_m: ``(H, W)`` metric depth.
            bearing_rad: direction of travel, radians CCW from the nose.

        Returns:
            ``(v_allow, d_min, certified)``.
        """
        if not self.sees_bearing(bearing_rad):
            v, d = self._speed_for(self.corridor_min_depth(depth_m, 0.0))
            return v, d, False
        v, d = self._speed_for(self.corridor_min_depth(depth_m, bearing_rad))
        return v, d, True

    def allowed_forward_speed(self, depth_m):
        # type: (np.ndarray) -> tuple
        """``(v_allow, d_min)``: max safe forward speed given this frame.

        ``v_allow`` is inf when the corridor is clear (nothing to brake for);
        0.0 when something already sits at the nose.
        """
        return self._speed_for(self.corridor_min_depth(depth_m, 0.0))

    def _speed_for(self, d_min):
        # type: (object) -> tuple
        """Turn a corridor range into an allowed speed. The braking arithmetic."""
        cfg = self._cfg
        if d_min is None:
            return float("inf"), None
        # The camera cannot see inside its ~0.95 m near clip, and inside it
        # the corridor minimum JUMPS to the background -- the brake would
        # release at exactly the closest range. So anything at or near the
        # clip is a full stop while it is still visible: the aircraft never
        # buys a look it cannot afford. (First deployment had margin 0.15,
        # which put the brake's whole authority below the clip: it could not
        # slow a 0.6 m/s cruise at any range it could observe.)
        if d_min <= cfg.hard_block_d_m:
            return 0.0, d_min
        avail = d_min - cfg.nose_offset_m - cfg.margin_m
        if avail <= 0.0:
            return 0.0, d_min
        a, t = cfg.brake_decel, cfg.react_s
        v = a * (-t + math.sqrt(t * t + 2.0 * avail / a))
        return max(0.0, v), d_min


def freer_side(depth_m, config=None):
    # type: (np.ndarray, object) -> float
    """Which way to turn when the corridor ahead is shut: ``+1`` left, ``-1`` right.

    The brake above says *stop*; it says nothing about where to go instead, and
    a policy that cannot see it is blocked will happily ask to fly forward
    again. This is the smallest honest answer to "which way is more open" --
    compare the two halves of the frame inside the same vertical band the
    corridor uses, and name the roomier one.

    It is a **reflex, not a plan**. It has no memory, no goal and no map, and it
    is only ever right about the first few metres. A caller should use it to
    break a deadlock and then hand the decision straight back to the policy.

    Sign convention: the image's ``u`` grows to the right, and the body frame is
    FLU, so columns left of ``cx`` are the aircraft's **left** and a positive
    return means "rotate counter-clockwise". Getting that backwards turns an
    escape into a second attempt at the same wall.

    Args:
        depth_m: ``(H, W)`` metric depth, metres. Non-finite samples are treated
            as "as far as this sensor can see", which is what a Gazebo depth
            camera's misses (the sky, a window) actually mean.
        config: a :class:`DepthProximityBrakeConfig` for the intrinsics and the
            vertical band; the default is this platform's.

    Returns:
        ``+1.0`` to rotate left, ``-1.0`` to rotate right. Ties break left,
        deterministically, because an escape that alternates on noise never
        leaves.
    """
    cfg = config or DepthProximityBrakeConfig()
    s = max(1, int(cfg.stride))
    d = np.asarray(depth_m, dtype=np.float32)[::s, ::s]
    if d.size == 0:
        return 1.0
    rows = np.arange(0, np.asarray(depth_m).shape[0], s, dtype=np.float32)
    cols = np.arange(0, np.asarray(depth_m).shape[1], s, dtype=np.float32)
    u, v = np.meshgrid(cols, rows)
    with np.errstate(invalid="ignore"):
        # An invalid or absent return is open space, not a wall. Scoring it as
        # zero would make a doorway (which the depth camera sees straight
        # through) read as the most blocked direction in the frame.
        reach = np.where(np.isfinite(d) & (d > cfg.min_valid_m), d, 10.0)
        band = np.abs(v - cfg.cy) * reach / cfg.fy <= cfg.corridor_halfheight_m
    if not band.any():
        band = np.ones_like(reach, dtype=bool)
    left = band & (u < cfg.cx)
    right = band & (u > cfg.cx)
    left_reach = float(reach[left].mean()) if left.any() else 0.0
    right_reach = float(reach[right].mean()) if right.any() else 0.0
    # An explicit tie band, not `>=`. The two halves rarely hold the same NUMBER
    # of samples (an odd width leaves the centre column in neither), so a
    # genuinely symmetric scene -- a flat wall dead ahead, which is exactly when
    # this is asked -- differs in the last float32 ulp and the "deterministic"
    # tie-break lands wherever the rounding does. One millimetre of asymmetry is
    # not a reason to prefer a side.
    if abs(left_reach - right_reach) <= 1e-3:
        return 1.0
    return 1.0 if left_reach > right_reach else -1.0
