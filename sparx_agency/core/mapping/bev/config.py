"""
Configuration for projecting FALCON's 3D voxel map onto a 2D BEV grid.

FALCON publishes its (already temporally fused) voxel map as two point
clouds of voxel centres in the world frame -- occupied and known-free. This
config holds every spatial parameter the projector needs to collapse those
voxels into a clean 2D OccupancyGrid-valued grid. It carries no ROS state;
the task/adapter layer fills it from rosparams.

Defaults are the values validated on the DA3 (monocular, ~3 Hz) indoor
setup. Set a gate to its disabling value to isolate a single stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.common.types.primitives import _assert_finite

_PROFILES = ("triangular", "gaussian", "flat")
_WALL_MODES = ("off", "directional", "count")
_CONN = (6, 18, 26)


@dataclass
class BevConfig:
    """
    Spatial parameters for voxel -> 2D BEV projection.

    Geometry / IO:
        resolution_m: Metres per cell. Match FALCON's voxel size.
        x_min, x_max, y_min, y_max: BEV bounds in world metres.
        frame_id: Frame the grid is expressed in (FALCON: "world").
        occ_dilate_cells: Inflate OCC by N cells (0 = off).

    Vertical column / height weighting (stage 1):
        z_floor, z_ceil: Column z-range considered (metres).
        z_peak: Flight altitude; per-voxel weight peaks here.
        weight_profile: "triangular" | "gaussian" | "flat".
        weight_sigma: Gaussian width (used iff profile == "gaussian").
        voxel_size_m: Z-layer thickness; defaults to resolution_m.

    Occupancy decision (stage 1):
        occ_weight_thresh: Min weighted column mass to be an OCC *candidate*.
        min_occ_voxels: Min raw occupied-voxel count for an OCC candidate.
        min_free_voxels: Min free-voxel count for a cell to read FREE.
        occ_conf_full: Weighted column mass at which a single frame counts as a
            full-confidence (conf == 1) occupied observation. Per-frame
            confidence is occ_w / occ_conf_full clipped to [0, 1]; it scales how
            fast a candidate accrues temporal evidence (stage 5). A marginal
            column (mass just over occ_weight_thresh) -- the typical monocular
            speckle in a corridor opening -- accrues slowly and needs many
            frames, while a solid wall accrues ~t_inc/frame and confirms in
            ~t_on/t_inc frames. Raise it to be stricter (slower to trust),
            lower it (toward occ_weight_thresh) to trust faster.

    3D neighbour confirm (stage 2):
        confirm_3d: Drop occupied voxels with too few occupied neighbours.
        neighbors_3d: Connectivity 6 | 18 | 26.
        min_occ_neighbors_3d: Min occupied neighbours to survive.

    Door / window protection (stage 3):
        protect_openings: Never wall a cell that is open at flight height.
        door_band_m: Z-band around z_peak inspected for openness.
        door_free_voxels: Free voxels in the band => it is an opening.
        door_occ_tol: Max occupied voxels tolerated in the band.

    Wall completion (stage 4):
        wall_fill_mode: "off" | "directional" | "count".
        wall_fill_neighbors: (count mode) occupied 8-neighbours to fill.
        wall_fill_iters: Max bridge width; keep small (1-2).

    Temporal confirmation (stage 5, STATEFUL, ON by default):
        temporal_filter: Require a cell to be observed occupied with enough
            confidence across MULTIPLE frames before it is published OCCUPIED.
            FALCON fuses in time, but its monocular-depth speckle still leaks
            single-frame false positives into corridor openings the camera is
            not pointed straight at; this stage rejects them. When on (the
            default) BevProjector is stateful (holds a per-cell evidence map).
            Set it off for a pure single-frame projection -- e.g. a unit test
            that asserts on the result of one project() call.
        t_inc: Per-frame evidence scale. The increment on a candidate cell is
            t_inc * confidence (see occ_conf_full), so weak/marginal columns add
            little and solid walls add ~t_inc.
        t_dec: Per-frame evidence removed from a cell observed FREE -- lets a
            wrongly-filled opening recover within a frame or two once it is
            actually seen to be open.
        t_max: Evidence saturation ceiling.
        t_on, t_off: Schmitt thresholds. A cell turns OCC once evidence reaches
            >= t_on (roughly t_on / t_inc confident frames) and only clears once
            it falls <= t_off. Keep t_on <= t_max or cells can never confirm.
    """

    # geometry / IO
    resolution_m: float = 0.15
    x_min: float = -12.0
    x_max: float = 12.0
    y_min: float = -12.0
    y_max: float = 12.0
    frame_id: str = "world"
    occ_dilate_cells: int = 0

    # vertical column / height weighting
    z_floor: float = 0.30
    z_ceil: float = 2.20
    z_peak: float = 1.00
    weight_profile: str = "triangular"
    weight_sigma: float = 0.50
    voxel_size_m: Optional[float] = None

    # occupancy decision
    occ_weight_thresh: float = 1.2
    min_occ_voxels: int = 2
    min_free_voxels: int = 1
    occ_conf_full: float = 3.0

    # 3D neighbour confirm
    confirm_3d: bool = True
    neighbors_3d: int = 6
    min_occ_neighbors_3d: int = 1

    # door / window protection
    protect_openings: bool = True
    door_band_m: float = 0.60
    door_free_voxels: int = 2
    door_occ_tol: int = 0

    # wall completion
    wall_fill_mode: str = "directional"
    wall_fill_neighbors: int = 5
    wall_fill_iters: int = 1

    # temporal confirmation (multi-frame; stateful)
    temporal_filter: bool = True
    t_inc: float = 1.0
    t_dec: float = 1.0
    t_max: float = 5.0
    t_on: float = 2.0
    t_off: float = 0.5

    def __post_init__(self) -> None:
        for name in ("resolution_m", "x_min", "x_max", "y_min", "y_max",
                     "z_floor", "z_ceil", "z_peak", "weight_sigma",
                     "occ_weight_thresh", "occ_conf_full", "door_band_m",
                     "t_inc", "t_dec", "t_max", "t_on", "t_off"):
            _assert_finite(f"BevConfig.{name}", float(getattr(self, name)))

        if self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be > 0, got {self.resolution_m}")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"invalid bounds x=[{self.x_min},{self.x_max}] "
                             f"y=[{self.y_min},{self.y_max}]")
        if self.z_ceil <= self.z_floor:
            raise ValueError(f"z_ceil must be > z_floor, got "
                             f"[{self.z_floor},{self.z_ceil}]")
        if self.weight_profile not in _PROFILES:
            raise ValueError(f"weight_profile must be one of {_PROFILES}")
        if self.wall_fill_mode not in _WALL_MODES:
            raise ValueError(f"wall_fill_mode must be one of {_WALL_MODES}")
        if self.neighbors_3d not in _CONN:
            raise ValueError(f"neighbors_3d must be one of {_CONN}")
        if self.occ_dilate_cells < 0:
            raise ValueError("occ_dilate_cells must be >= 0")
        if self.occ_conf_full <= self.occ_weight_thresh:
            raise ValueError(
                f"occ_conf_full must be > occ_weight_thresh, got "
                f"occ_conf_full={self.occ_conf_full}, "
                f"occ_weight_thresh={self.occ_weight_thresh}")

        if self.voxel_size_m is None:
            self.voxel_size_m = self.resolution_m
        elif self.voxel_size_m <= 0:
            raise ValueError(f"voxel_size_m must be > 0, got {self.voxel_size_m}")

        if self.temporal_filter:
            if self.t_max <= 0:
                raise ValueError(f"t_max must be > 0, got {self.t_max}")
            if self.t_inc <= 0:
                raise ValueError(f"t_inc must be > 0, got {self.t_inc}")
            if self.t_on < self.t_off:
                raise ValueError(f"t_on must be >= t_off, got "
                                 f"[{self.t_off},{self.t_on}]")
            if self.t_on > self.t_max:
                raise ValueError(f"t_on must be <= t_max or no cell can ever "
                                 f"confirm, got t_on={self.t_on}, "
                                 f"t_max={self.t_max}")