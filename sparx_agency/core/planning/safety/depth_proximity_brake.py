"""Brake on what the depth camera sees RIGHT NOW, before the map knows it.

The voxel map is the planner's memory; this is the reflex. A thin obstacle
(a person, a pole, a door edge) can be visible in the raw depth image for
seconds while the voxel map still flaps it between occupied and free --
raycast clearing erodes one-voxel-thick silhouettes -- and an aircraft
braking only on the map flies through that gap in its knowledge. Measured
in the hospital campaign: every remaining strike of run 009 was one thin
human, approached four times, visible in raw depth at 2.8 m each time.

This class answers one question per frame: how fast may the aircraft fly
FORWARD (along the camera axis) given the closest depth return inside the
flight corridor? Everything else -- lateral motion, mapped obstacles behind
the aircraft -- belongs to the voxel gate; the two compose.

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

    def corridor_min_depth(self, depth_m):
        # type: (np.ndarray) -> object
        """Closest valid return inside the flight corridor, or None.

        A pixel is in the corridor when its back-projected lateral offset is
        within ``corridor_halfwidth_m`` and its vertical offset within
        ``corridor_halfheight_m`` -- i.e. the cross-section the airframe
        sweeps if it flies straight ahead.
        """
        cfg = self._cfg
        s = cfg.stride
        d = depth_m[::s, ::s]
        u, v = self._grids(depth_m.shape)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(d) & (d > cfg.min_valid_m)
            lat = np.abs(u - cfg.cx) * d / cfg.fx
            vert = np.abs(v - cfg.cy) * d / cfg.fy
            mask = (valid & (lat <= cfg.corridor_halfwidth_m)
                    & (vert <= cfg.corridor_halfheight_m))
        if mask.any():
            return float(d[mask].min())
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

    def allowed_forward_speed(self, depth_m):
        # type: (np.ndarray) -> tuple
        """``(v_allow, d_min)``: max safe forward speed given this frame.

        ``v_allow`` is inf when the corridor is clear (nothing to brake for);
        0.0 when something already sits at the nose.
        """
        cfg = self._cfg
        d_min = self.corridor_min_depth(depth_m)
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
