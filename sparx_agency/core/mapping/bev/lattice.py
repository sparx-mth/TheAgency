"""
BEV lattice: the fixed 2D grid plus vertical (z) discretisation, and the
voxelisation that places FALCON's world-frame voxel centres into that grid.

Pure numpy; no ROS. A BevLattice is derived once from a BevConfig and reused
every frame: it knows the grid dimensions, the per-z-layer height weights and
the door-inspection band, and turns (N,3) world-point arrays into the dense
3D volume / 2D column counts the projector reasons over.

Conventions (shared with the rest of the costmap stack):
  - points are in WORLD/MAP coordinates (x east, y north, z up)
  - grids are (H, W) indexed [gy, gx]; origin is the bottom-left corner
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.mapping.interfaces.costmap import GridSpec
from .config import BevConfig


def _height_weights(z_centres: np.ndarray, cfg: BevConfig) -> np.ndarray:
    """Per-layer weight in [0,1]; 0 outside [z_floor,z_ceil], peak 1 at z_peak."""
    z = z_centres
    w = np.zeros_like(z, np.float32)
    inb = (z >= cfg.z_floor) & (z <= cfg.z_ceil)
    if cfg.weight_profile == "flat":
        w[inb] = 1.0
    elif cfg.weight_profile == "gaussian":
        s = max(1e-3, cfg.weight_sigma)
        w[inb] = np.exp(-0.5 * ((z[inb] - cfg.z_peak) / s) ** 2)
    else:  # triangular
        up = inb & (z <= cfg.z_peak)
        dn = inb & (z > cfg.z_peak)
        w[up] = ((z[up] - cfg.z_floor) / (cfg.z_peak - cfg.z_floor)
                 if cfg.z_peak > cfg.z_floor else 1.0)
        w[dn] = ((cfg.z_ceil - z[dn]) / (cfg.z_ceil - cfg.z_peak)
                 if cfg.z_ceil > cfg.z_peak else 1.0)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


class BevLattice:
    """Fixed grid + z-layers derived from a BevConfig."""

    def __init__(self, cfg: BevConfig):
        self.cfg = cfg
        self.res = float(cfg.resolution_m)
        self.x_min, self.y_min = float(cfg.x_min), float(cfg.y_min)
        self.vz = float(cfg.voxel_size_m)

        self.W = int(round((cfg.x_max - cfg.x_min) / self.res))
        self.H = int(round((cfg.y_max - cfg.y_min) / self.res))
        self.nz = max(1, int(round((cfg.z_ceil - cfg.z_floor) / self.vz)))

        z_centres = cfg.z_floor + (np.arange(self.nz) + 0.5) * self.vz
        self.z_weights = _height_weights(z_centres, cfg)              # (nz,)
        self.band_idx = np.where(
            (z_centres >= cfg.z_peak - 0.5 * cfg.door_band_m) &
            (z_centres <= cfg.z_peak + 0.5 * cfg.door_band_m))[0]

    def spec(self) -> GridSpec:
        return GridSpec(self.res, self.W, self.H,
                        self.x_min, self.y_min, self.cfg.frame_id)

    def world_to_cell(self, xy: np.ndarray):
        """(N,>=2) world -> (cx, cy, in_bounds) int32 / bool cell indices."""
        cx = ((xy[:, 0] - self.x_min) / self.res).astype(np.int32)
        cy = ((xy[:, 1] - self.y_min) / self.res).astype(np.int32)
        ok = (cx >= 0) & (cx < self.W) & (cy >= 0) & (cy < self.H)
        return cx, cy, ok

    def occupied_volume(self, occ_xyz: np.ndarray) -> np.ndarray:
        """Sparse 3D occupied volume (nz,H,W) bool from voxel centres."""
        vol = np.zeros((self.nz, self.H, self.W), bool)
        if occ_xyz.shape[0]:
            cx, cy, ok = self.world_to_cell(occ_xyz)
            lz = np.floor((occ_xyz[:, 2] - self.cfg.z_floor) / self.vz).astype(np.int32)
            ok &= (lz >= 0) & (lz < self.nz)
            if ok.any():
                flat = (lz[ok] * self.H + cy[ok]) * self.W + cx[ok]
                vol.reshape(-1)[flat] = True
        return vol

    def column_count(self, xyz: np.ndarray, zlo=None, zhi=None) -> np.ndarray:
        """(H,W) int32 count of points per cell; optionally only z in [zlo,zhi]."""
        out = np.zeros((self.H, self.W), np.int32)
        if xyz.shape[0] == 0:
            return out
        cx, cy, ok = self.world_to_cell(xyz)
        if zlo is not None:
            ok &= (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi)
        if ok.any():
            out += np.bincount(cy[ok] * self.W + cx[ok],
                               minlength=self.H * self.W).reshape(self.H, self.W)
        return out