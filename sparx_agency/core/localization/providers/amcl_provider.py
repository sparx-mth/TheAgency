"""AMCL localization provider (Option A — single node, optical flow internal).

Architecture:
  - OpticalFlowTracker runs internally for the motion model (velocity only).
  - Position state is owned exclusively by AMCL; optical flow position is never used.
  - At each depth frame:
      1. Extract optical flow velocity from the pending RGB queue.
      2. motion_predict(last_AMCL_pos, v, dt) → predicted grid position.
      3. Slice a local window around the prediction from the precomputed LUT.
      4. Sample depth → horizontal range measurements z[].
      5. amcl_estimator(lut_window, z) → MAP estimate → new AMCL position.
  - LUT is loaded from amcl_lut.npy if cached; built in a background thread otherwise.
    Run precompute_amcl_lut.py offline to avoid first-run delays.
"""
from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.tasks.localization.amcl import (
    amcl_estimator,
    extract_local_window,
    motion_predict,
    ray_cast_lut_vectorized,
)
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker
from sparx_agency.core.localization.providers.optical_flow_provider import (
    _float_to_stamp,
    _flow_velocity,
    _load_flow_intrinsics,
)


def _depth_to_horizontal_ranges(
    depth_map: np.ndarray,
    beam_angles: np.ndarray,
    fx: float,
    cx: float,
    max_range_m: float,
    row_fraction: float = 0.5,
) -> np.ndarray:
    """Sample depth at each beam angle along the image horizon row."""
    row = int(depth_map.shape[0] * row_fraction)
    z = np.full(len(beam_angles), max_range_m, dtype=np.float32)
    for i, angle in enumerate(beam_angles):
        u = int(round(cx + fx * math.tan(angle)))
        if 0 <= u < depth_map.shape[1]:
            d = float(depth_map[row, u])
            if np.isfinite(d) and d > 0.0:
                z[i] = min(d, max_range_m)
    return z


class AmclLocalizationProvider(BaseLocalizationProvider):
    """
    Grid-based AMCL fused with optical flow motion model.

    Single node — optical flow runs internally for velocity only.
    AMCL corrects accumulated drift via map-matching at each depth frame.
    """

    source_name = "amcl"

    def __init__(
        self,
        map_dir: str,
        camera_calib_path: str,
        num_orientations: int = 32,
        num_beams: int = 64,
        max_range_m: float = 8.0,
        window_m: float = 5.0,
        initial_loc_m_json: str = "",
        initial_orientation_rad: float = 0.0,
        prediction_uncertainty_cells: int = 5,
        max_wait_for_depth_sec: float = 0.1,
        allow_depth_before_flow_sec: float = 0.03,
        flow_queue_size: int = 200,
        min_corners: int = 70,
        max_corners: int = 300,
    ) -> None:
        self._world, self._m_per_cell, self._origin = _load_map(map_dir)
        self._orientations = np.linspace(0, 2 * np.pi, num_orientations, endpoint=False)
        self._beam_angles = np.linspace(-np.pi / 2, np.pi / 2, num_beams)
        self._max_range_m = max_range_m
        self._max_range_cells = max_range_m / self._m_per_cell
        self._window_cells = max(1, int(window_m / self._m_per_cell))
        self._uncertainty = prediction_uncertainty_cells
        self._fx, self._fy, self._cx, self._cy = _load_flow_intrinsics(camera_calib_path)
        self._tracker = OpticalFlowTracker(max_corners=max_corners, min_corners=min_corners)
        self._pending: Deque[Dict[str, Any]] = deque(maxlen=flow_queue_size)
        self._max_wait = max_wait_for_depth_sec
        self._allow_before = allow_depth_before_flow_sec
        self._last_loc_grid = _parse_initial_loc(
            initial_loc_m_json, self._origin, self._m_per_cell, self._world.shape
        )
        self._last_orientation = float(initial_orientation_rad)
        self._last_stamp: float = 0.0
        self._lut: Optional[np.ndarray] = None
        self._lut_ready = threading.Event()
        self._start_lut(Path(map_dir) / "amcl_lut.npy")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_rgb(self, frame: np.ndarray, stamp_sec: float) -> None:
        """Track optical flow and queue the result for later depth matching."""
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
        """Predict via optical flow velocity, correct via map, return AMCL estimate."""
        if not self._lut_ready.is_set():
            return None
        flow = self._find_best_flow(stamp_sec)
        if flow is None:
            return None
        depth_map = depth_map.astype(np.float32)
        vx, vy, n = self._compute_velocity(flow, depth_map)
        pred_loc, pred_yaw = motion_predict(
            self._last_loc_grid, self._last_orientation, vx, vy, flow["dt"], self._m_per_cell
        )
        pred_loc = np.clip(pred_loc, [0, 0],
                           [self._world.shape[0] - 1, self._world.shape[1] - 1])
        return self._map_update(pred_loc, pred_yaw, depth_map, n, flow["stamp_sec"])

    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        if obs.rgb is not None:
            self.process_rgb(obs.rgb.image, obs.rgb.stamp_sec)
        if obs.depth is not None:
            return self.process_depth(obs.depth.depth_m, obs.depth.stamp_sec)
        return None

    def set_yaw(self, yaw_rad: float) -> None:
        """Inject bearing correction into AMCL orientation state."""
        self._last_orientation = float(yaw_rad)

    def reset(self) -> None:
        self._tracker.reset()
        self._pending.clear()
        self._last_stamp = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_velocity(self, flow, depth_map):
        raw_vx, raw_vy, _, n = _flow_velocity(
            flow["good_old"], flow["good_new"], depth_map, flow["dt"],
            self._fx, self._fy, self._cx, self._cy,
            0.1, self._max_range_m, 1.0,
        )
        return (raw_vx, raw_vy, n) if n > 0 else (0.0, 0.0, 0)

    def _map_update(self, pred_loc, pred_yaw, depth_map, n_inliers, stamp):
        lut_w, origin = extract_local_window(self._lut, pred_loc, self._window_cells)
        world_w, _ = extract_local_window(self._world, pred_loc, self._window_cells)
        if lut_w.shape[0] == 0 or lut_w.shape[1] == 0:
            return None
        z = _depth_to_horizontal_ranges(
            depth_map, self._beam_angles, self._fx, self._cx, self._max_range_m
        )
        local_pred = pred_loc - origin
        loc_in_win, orientation = amcl_estimator(
            lut_w, self._orientations, local_pred, pred_yaw,
            world_w, z, (self._uncertainty, self._uncertainty),
        )
        self._last_loc_grid = (origin + loc_in_win).astype(np.float64)
        self._last_orientation = float(orientation)
        self._last_stamp = stamp
        x_m = self._last_loc_grid[1] * self._m_per_cell + self._origin[0]
        y_m = self._last_loc_grid[0] * self._m_per_cell + self._origin[1]
        confidence = min(1.0, n_inliers / 100.0)
        return LocalizationEstimate(
            pose=Pose3D(x=x_m, y=y_m, z=0.0, yaw=self._last_orientation),
            source=self.source_name,
            confidence=confidence,
            stamp_sec=stamp,
            pos_std_m=max(0.05, 0.1 * (1.0 - confidence)),
            yaw_std_rad=0.1,
        )

    def _find_best_flow(self, depth_sec: float) -> Optional[Dict[str, Any]]:
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

    def _start_lut(self, lut_path: Path) -> None:
        if lut_path.exists():
            self._lut = np.load(str(lut_path), mmap_mode='r')
            self._lut_ready.set()
            print(f"[amcl] LUT loaded: shape={self._lut.shape}, {self._lut.nbytes / 1e6:.0f} MB")
        else:
            print(f"[amcl] amcl_lut.npy not found — building in background thread")
            print(f"[amcl] Run precompute_amcl_lut.py offline to avoid this delay")
            threading.Thread(target=self._build_and_cache_lut,
                             args=(lut_path,), daemon=True).start()

    def _build_and_cache_lut(self, lut_path: Path) -> None:
        n_or, n_b = len(self._orientations), len(self._beam_angles)
        print(f"[amcl] Building LUT: {self._world.shape} × {n_or}or × {n_b}b …", flush=True)
        lut = ray_cast_lut_vectorized(
            self._world, self._orientations, self._beam_angles,
            self._max_range_cells, step=1.0,
        )
        np.save(str(lut_path), lut)
        self._lut = np.load(str(lut_path), mmap_mode='r')
        self._lut_ready.set()
        print(f"[amcl] LUT ready — {lut_path} ({lut.nbytes / 1e6:.0f} MB)", flush=True)


# ------------------------------------------------------------------
# Module-level helpers (pure functions, no provider state)
# ------------------------------------------------------------------

def _load_map(map_dir: str):
    p = Path(map_dir)
    grid_path = p / "occ_grid_int8.npy"
    meta_path = p / "occ_metadata.json"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Occupancy grid not found: {grid_path}\n"
            "Build it first:\n"
            "  python3 -m sparx_agency.tasks.mapping.map_from_image_clean --out-dir <map_dir> ..."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Map metadata not found: {meta_path}")
    raw = np.load(str(grid_path))
    with open(meta_path) as f:
        meta = json.load(f)
    m_per_cell = float(meta["resolution_m_per_cell"])
    origin = np.array([float(meta.get("origin_x_m", 0.0)),
                       float(meta.get("origin_y_m", 0.0))])
    world = (raw == 100).astype(np.float32)
    return world, m_per_cell, origin


def _parse_initial_loc(loc_m_json: str, origin: np.ndarray,
                        m_per_cell: float, shape: tuple) -> np.ndarray:
    if loc_m_json.strip():
        loc = json.loads(loc_m_json)
        col = (float(loc[0]) - origin[0]) / m_per_cell
        row = (float(loc[1]) - origin[1]) / m_per_cell
        return np.array([row, col], dtype=np.float64)
    return np.array([shape[0] / 2.0, shape[1] / 2.0], dtype=np.float64)