"""Pure-Python optical-flow + depth localization provider.

Two-stage queued architecture (mirrors flow_depth_velocity_node_separated):
  1. process_rgb()   → computes LK flow immediately, stores in pending queue.
  2. process_depth() → finds closest queued flow, computes metric velocity,
                       integrates position, returns LocalizationEstimate.

Yaw (Option A):
  - set_yaw() locks initial_yaw on first call (bearing reference frame).
  - set_turning(True) freezes velocity to zero (no drift during turns).
  - World-frame integration uses yaw_offset = initial_yaw.
"""
from __future__ import annotations

import math
from collections import deque, namedtuple
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import yaml

from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker

_Stamp = namedtuple("_Stamp", ["sec", "nanosec"])


def _float_to_stamp(t: float) -> _Stamp:
    sec = int(t)
    return _Stamp(sec=sec, nanosec=int((t - sec) * 1e9))


def _load_flow_intrinsics(path: str) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from YAML (direct keys, projection_matrix, or camera_matrix)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera calib not found: {path}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    pm = cfg.get("projection_matrix", {})
    if pm and "data" in pm:
        P = pm["data"]
        if len(P) >= 12 and P[0] != 0.0:
            return float(P[0]), float(P[5]), float(P[2]), float(P[6])

    if all(k in cfg for k in ("fx", "fy", "cx", "cy")):
        return float(cfg["fx"]), float(cfg["fy"]), float(cfg["cx"]), float(cfg["cy"])

    cm = cfg.get("camera_matrix", {})
    if cm and "data" in cm:
        K = cm["data"]
        if len(K) >= 9 and K[0] != 0.0:
            return float(K[0]), float(K[4]), float(K[2]), float(K[5])

    raise ValueError(f"No usable intrinsics (fx/fy/cx/cy) in: {path}")


def _inlier_mask(
    du: np.ndarray, dv: np.ndarray, Z: np.ndarray,
    fx: float, fy: float, threshold: float = 0.3,
) -> np.ndarray:
    vx_est = -(du * Z) / fx
    vy_est = -(dv * Z) / fy
    return (
        (np.abs(vx_est - np.median(vx_est)) < threshold)
        & (np.abs(vy_est - np.median(vy_est)) < threshold)
    )


def _flow_velocity(
    good_old: np.ndarray, good_new: np.ndarray,
    depth_map: np.ndarray, dt: float,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float, max_depth: float, depth_scale: float,
) -> Tuple[float, float, float, int]:
    """Metric velocity from optical flow + depth. Returns (vx, vy, vz, n_inliers)."""
    if dt <= 0.0:
        return 0.0, 0.0, 0.0, 0

    H, W = depth_map.shape[:2]
    du = (good_new[:, 0] - good_old[:, 0]) / dt
    dv = (good_new[:, 1] - good_old[:, 1]) / dt

    u_int = np.rint(good_new[:, 0]).astype(np.int32)
    v_int = np.rint(good_new[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0

    Z = np.zeros(len(du), dtype=np.float32)
    Z[valid] = depth_map[v_int[valid], u_int[valid]] * depth_scale
    valid &= np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
    if int(np.sum(valid)) < 8:
        return 0.0, 0.0, 0.0, 0

    valid &= _inlier_mask(du, dv, Z, fx, fy)
    nv = int(np.sum(valid))
    if nv < 8:
        return 0.0, 0.0, 0.0, 0

    u_c = good_old[valid, 0].astype(np.float64) - cx
    v_c = good_old[valid, 1].astype(np.float64) - cy
    Zv = Z[valid].astype(np.float64)
    du_v = du[valid].astype(np.float64)
    dv_v = dv[valid].astype(np.float64)

    A = np.zeros((2 * nv, 3), dtype=np.float64)
    B = np.zeros(2 * nv, dtype=np.float64)
    A[0::2, 0] = -fx;  A[0::2, 2] = u_c;  B[0::2] = du_v * Zv
    A[1::2, 1] = -fy;  A[1::2, 2] = v_c;  B[1::2] = dv_v * Zv

    w = np.exp(-(u_c ** 2 + v_c ** 2) / max(0.05 * (cx ** 2 + cy ** 2), 1e-9))
    sw = np.sqrt(np.repeat(w, 2))
    try:
        vel, *_ = np.linalg.lstsq(A * sw[:, None], B * sw, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, nv

    # Axis convention from separated node: return (Vz, -Vx, -Vy)
    return float(vel[2]), float(-vel[0]), float(-vel[1]), nv


class OpticalFlowLocalizationProvider(BaseLocalizationProvider):
    """
    Optical-flow + metric-depth localization.

    Call process_rgb() on each RGB frame and process_depth() on each depth frame.
    The two-stage queue handles the depth arriving later than RGB.
    """

    source_name = "optical_flow"

    def __init__(
        self,
        camera_calib_path: str,
        min_depth: float = 0.05,
        max_depth: float = 10.0,
        min_corners: int = 70,
        max_corners: int = 300,
        lk_win: int = 21,
        lk_levels: int = 3,
        depth_ema_alpha: float = 0.05,
        vel_alpha: float = 0.2,
        max_wait_for_depth_sec: float = 0.1,
        allow_depth_before_flow_sec: float = 0.03,
        flow_queue_size: int = 200,
        depth_scale: float = 1.0,
    ) -> None:
        self._fx, self._fy, self._cx, self._cy = _load_flow_intrinsics(camera_calib_path)
        self._tracker = OpticalFlowTracker(
            max_corners=max_corners, min_corners=min_corners,
            lk_win=lk_win, lk_levels=lk_levels,
        )
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._depth_scale = depth_scale
        self._ema_alpha = depth_ema_alpha
        self._vel_alpha = vel_alpha
        self._max_wait = max_wait_for_depth_sec
        self._allow_before = allow_depth_before_flow_sec
        self._pending: Deque[Dict[str, Any]] = deque(maxlen=flow_queue_size)
        self._last_depth: Optional[np.ndarray] = None
        self._prev_vx = self._prev_vy = self._prev_vz = 0.0
        self._x = self._y = self._z = 0.0
        self._initial_yaw: Optional[float] = None
        self._current_yaw: float = 0.0
        self._is_turning: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_yaw(self, yaw_rad: float) -> None:
        """Lock initial heading on first call; track current yaw thereafter."""
        if self._initial_yaw is None:
            self._initial_yaw = yaw_rad
        self._current_yaw = yaw_rad

    def set_turning(self, turning: bool) -> None:
        """Freeze velocity to zero during turns to prevent integration drift."""
        if turning and not self._is_turning:
            self._prev_vx = self._prev_vy = self._prev_vz = 0.0
        self._is_turning = turning

    def process_rgb(self, frame: np.ndarray, stamp_sec: float) -> None:
        """Compute LK flow and enqueue result for later depth matching."""
        flow = self._tracker.process(frame, _float_to_stamp(stamp_sec))
        if flow is None or flow.dt <= 0.0:
            return
        self._pending.append({
            "stamp_sec": stamp_sec,
            "good_old": flow.good_old.copy(),
            "good_new": flow.good_new.copy(),
            "dt": float(flow.dt),
        })

    def process_depth(
        self, depth_map: np.ndarray, stamp_sec: float
    ) -> Optional[LocalizationEstimate]:
        """Match closest queued flow, compute velocity, integrate position."""
        flow = self._find_best_flow(stamp_sec)
        if flow is None:
            return None

        depth_map = self._smooth_depth(depth_map.astype(np.float32))

        if self._is_turning:
            self._prev_vx = self._prev_vy = self._prev_vz = 0.0
            return None

        raw_vx, raw_vy, raw_vz, n = _flow_velocity(
            flow["good_old"], flow["good_new"], depth_map, flow["dt"],
            self._fx, self._fy, self._cx, self._cy,
            self._min_depth, self._max_depth, self._depth_scale,
        )
        if n == 0:
            return None

        vx = self._vel_alpha * raw_vx + (1.0 - self._vel_alpha) * self._prev_vx
        vy = self._vel_alpha * raw_vy + (1.0 - self._vel_alpha) * self._prev_vy
        vz = self._vel_alpha * raw_vz + (1.0 - self._vel_alpha) * self._prev_vz
        self._prev_vx, self._prev_vy, self._prev_vz = vx, vy, vz

        if abs(vx) < 0.02: vx = 0.0
        if abs(vy) < 0.20: vy = 0.0
        if abs(vz) < 0.20: vz = 0.0

        yaw_off = self._initial_yaw if self._initial_yaw is not None else 0.0
        dt = flow["dt"]
        self._x += (vx * math.cos(yaw_off) + vy * math.sin(yaw_off)) * dt
        self._y += (-vx * math.sin(yaw_off) + vy * math.cos(yaw_off)) * dt
        self._z += vz * dt

        confidence = float(min(1.0, n / 100.0))
        return LocalizationEstimate(
            pose=Pose3D(x=self._x, y=self._y, z=self._z, yaw=self._current_yaw),
            source=self.source_name,
            confidence=confidence,
            stamp_sec=stamp_sec,
            pos_std_m=max(0.05, 0.1 * (1.0 - confidence)),
            yaw_std_rad=0.1,
        )

    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        if obs.rgb is not None:
            self.process_rgb(obs.rgb.image, obs.rgb.stamp_sec)
        if obs.depth is not None:
            return self.process_depth(obs.depth.depth_m, obs.depth.stamp_sec)
        return None

    def reset(self) -> None:
        self._tracker.reset()
        self._pending.clear()
        self._last_depth = None
        self._prev_vx = self._prev_vy = self._prev_vz = 0.0
        self._x = self._y = self._z = 0.0
        self._initial_yaw = None
        self._current_yaw = 0.0
        self._is_turning = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_depth(self, raw: np.ndarray) -> np.ndarray:
        if self._last_depth is None or self._last_depth.shape != raw.shape:
            self._last_depth = raw
        else:
            self._last_depth = (
                self._ema_alpha * raw + (1.0 - self._ema_alpha) * self._last_depth
            )
        return self._last_depth.astype(np.float32)

    def _find_best_flow(self, depth_sec: float) -> Optional[Dict[str, Any]]:
        """Drop stale flows; return and consume the closest match within the time window."""
        while self._pending:
            if depth_sec - self._pending[0]["stamp_sec"] > self._max_wait:
                self._pending.popleft()
            else:
                break

        candidates = [
            (item, depth_sec - item["stamp_sec"])
            for item in self._pending
            if -self._allow_before <= (depth_sec - item["stamp_sec"]) <= self._max_wait
        ]
        if not candidates:
            return None

        best, _ = min(candidates, key=lambda p: abs(p[1]))
        while self._pending and self._pending[0] is not best:
            self._pending.popleft()
        if self._pending:
            self._pending.popleft()
        return best