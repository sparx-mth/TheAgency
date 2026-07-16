"""
BevProjector -- collapse FALCON's 3D voxel map into a clean 2D BEV grid.

FALCON publishes an already temporally-fused voxel map as two point clouds of
voxel centres in the world frame: occupied and known-free. The job here is
therefore purely SPATIAL (no raytracing, no log-odds, no sensor origin): turn
those voxels into an OccupancyGrid-valued 2D grid the planner can use, while
rejecting the monocular-depth speckle that FALCON's threshold lets through.

Pipeline (each stage is gated by BevConfig; disable one to isolate it):
  1. column projection    height-weighted occupied mass per (x,y) column
  2. 3D neighbour confirm  drop isolated occupied voxels (floaters)
  3. door protection       keep openings free where the flight band is clear
  4. temporal confirm      require a candidate to be seen occupied, with enough
                           confidence, over MULTIPLE frames before it is OCC
                           (ON by default; this rejects the view-dependent
                           monocular speckle that leaks into corridor openings)
  5. wall completion       bridge one-cell gaps in CONFIRMED walls
  5b. speck removal        drop OCC components that are not wall-like -- no
                           straight run of >=min_wall_run consecutive cells (and
                           an optional raw-area gate). A spatial, view-independent
                           cull of phantoms (a stuck voxel/clump in a turn opening
                           the drone can never re-observe free to clear); runs
                           AFTER wall completion so a bridged gapped wall survives
  6. compose               OCC > FREE > UNK, then stamp caller `force_occ` cells
  7. dilate                optional safety inflation (force_occ cells seed it too)

Output matches the costmap convention: (GridSpec, int8 (H,W)) with values
{UNKNOWN:-1, FREE:0, OCCUPIED:100}. With cfg.temporal_filter (the default) the
projector is STATEFUL: it holds a small per-cell evidence accumulator so a cell
must be seen occupied, with enough confidence, over several frames before it is
published OCCUPIED -- FALCON's in-time fusion is not enough to stop its
monocular speckle from filling openings the camera isn't aimed at. Set
cfg.temporal_filter=False for a pure single-frame projection (each project()
call then stands alone).

`force_occ` lets the caller stamp hard, env-specific obstacles (manual walls,
a virtual back-wall from /map_config) as OCCUPIED after compose and before
dilation, exactly as the legacy node did -- so those cells inflate with the
rest. Keeping it a generic mask keeps map-specific knowledge out of core.

NB: this deliberately does NOT implement the Costmap ABC. That interface
models incremental single-sensor integration
(update_from_cloud(cloud, sensor_origin)); BEV consumes a pre-fused
occupied+free voxel pair in one shot, so the contract genuinely differs.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sparx_agency.core.mapping.interfaces.costmap import GridSpec
from .config import BevConfig
from .lattice import BevLattice
from . import morphology as morph

UNKNOWN, FREE, OCCUPIED = -1, 0, 100


class BevProjector:
    """3D-voxel -> 2D-occupancy projector (stateful when temporal_filter)."""

    def __init__(self, cfg: BevConfig):
        self.cfg = cfg
        self.lattice = BevLattice(cfg)
        self.last_stats: Dict[str, int] = {}
        # Per-cell OCCUPIED confidence in [0, 1] from the last project() call, or
        # None when the temporal filter is off (no graded evidence exists then).
        # It is the temporal evidence / t_max for observed cells, and forced to 1.0
        # for caller-forced CERTAIN obstacles (``force_occ``), which carry no
        # evidence but are ground truth. A downstream planner reads this to gate an
        # obstacle reroute on a *confident* obstacle instead of a single
        # low-confidence depth speckle -- see route_obstacle_confidence.
        self.last_confidence: Optional[np.ndarray] = None
        # temporal-hysteresis state (used only when cfg.temporal_filter)
        self._ev = np.zeros((self.lattice.H, self.lattice.W), np.float32)
        self._occ_state = np.zeros((self.lattice.H, self.lattice.W), bool)

    def project(self, occupied_xyz: np.ndarray, free_xyz: np.ndarray,
                force_occ: Optional[np.ndarray] = None
                ) -> Tuple[GridSpec, np.ndarray]:
        """
        Args:
            occupied_xyz: (N,3) occupied voxel centres in the world frame.
            free_xyz:     (M,3) known-free voxel centres in the world frame.
            force_occ:    optional (H,W) bool mask of cells to force OCCUPIED
                          after compose and before dilation (e.g. manual walls).
        Returns:
            (spec, grid_int8) with grid shape (H,W) and values {-1, 0, 100}.
        """
        cfg, lat = self.cfg, self.lattice
        occ_xyz, free_xyz = _as_xyz(occupied_xyz), _as_xyz(free_xyz)

        # 1) 3D occupied volume (+ optional neighbour confirm to kill floaters)
        vol = lat.occupied_volume(occ_xyz)
        n_raw = int(vol.sum())
        if cfg.confirm_3d and n_raw:
            vol &= morph.count_neighbors_3d(vol, cfg.neighbors_3d) >= cfg.min_occ_neighbors_3d
        n_conf = int(vol.sum())

        # 2) height-weighted column projection
        occ_w = np.tensordot(lat.z_weights, vol.astype(np.float32), axes=([0], [0]))
        occ_c = vol.sum(axis=0).astype(np.int32)
        occ_band = (vol[lat.band_idx].sum(axis=0).astype(np.int32)
                    if lat.band_idx.size else np.zeros_like(occ_c))
        base_occ = (occ_w >= cfg.occ_weight_thresh) & (occ_c >= cfg.min_occ_voxels)

        # 3) free evidence (whole column + flight band)
        free_c = lat.column_count(free_xyz)
        free_band = lat.column_count(free_xyz,
                                     cfg.z_peak - 0.5 * cfg.door_band_m,
                                     cfg.z_peak + 0.5 * cfg.door_band_m)
        observed_free = free_c >= cfg.min_free_voxels

        # 4) door / window protection: open at flight height => force FREE
        protected = np.zeros_like(base_occ)
        if cfg.protect_openings:
            protected = (base_occ & (free_band >= cfg.door_free_voxels)
                         & (occ_band <= cfg.door_occ_tol))
            base_occ &= ~protected

        # 5) temporal confirmation (stateful, ON by default). A candidate cell
        #    only becomes OCCUPIED after it has been observed occupied, with
        #    enough confidence, across MULTIPLE frames -- this is what rejects
        #    the monocular-depth speckle FALCON leaks into corridor openings the
        #    camera is not aimed straight at. Per-frame confidence is
        #    occ_w / occ_conf_full in [0,1]: a marginal column (mass just over
        #    occ_weight_thresh) adds little evidence and needs many frames, a
        #    solid wall adds ~t_inc and confirms in ~t_on/t_inc frames. A Schmitt
        #    trigger (t_on/t_off) stops flicker; cells seen FREE bleed evidence
        #    (t_dec) so a wrongly-filled opening recovers once it's actually
        #    observed open. With temporal_filter off this is a no-op passthrough.
        if cfg.temporal_filter:
            conf = np.clip(occ_w / cfg.occ_conf_full, 0.0, 1.0) * base_occ
            self._ev += cfg.t_inc * conf - cfg.t_dec * (observed_free & ~base_occ)
            np.clip(self._ev, 0.0, cfg.t_max, out=self._ev)
            self._occ_state = ((self._occ_state & (self._ev > cfg.t_off))
                               | (self._ev >= cfg.t_on))
            confirmed = self._occ_state.copy()
            # Publish the accumulated evidence as a [0, 1] confidence so a planner
            # can distinguish a barely-latched speckle (near t_on/t_max) from a
            # solid, repeatedly-seen wall (near 1.0).
            self.last_confidence = (self._ev / cfg.t_max).astype(np.float32)
        else:
            confirmed = base_occ
            self.last_confidence = None
        n_pending = int((base_occ & ~confirmed).sum())

        # 6) wall completion: bridge one-cell gaps in CONFIRMED walls only,
        #    never over observed-free cells or protected openings.
        occ, n_fill = morph.bridge_fill(
            confirmed, observed_free | protected,
            mode=cfg.wall_fill_mode, n_neighbors=cfg.wall_fill_neighbors,
            iters=cfg.wall_fill_iters)

        # 6b) speck removal. A phantom FALCON drops into a turn opening blocks the
        #    planner but can never be re-observed free to clear it (the drone
        #    can't route to look at it), so it deadlocks. Cull it SPATIALLY --
        #    view-independently, no re-observation needed. Two gates, either may
        #    fire: a linear-run test (a real wall is >=min_wall_run cells in a
        #    line; a 2x2 clump or L-tromino is not) and a coarser raw-area test.
        #    Runs AFTER wall completion so a real wall with a one-cell gap is
        #    bridged into one component and survives, while a genuinely isolated
        #    phantom (never bridged) stays small/clumpy and is removed.
        n_speck = 0
        for enabled, fn, arg in (
                (cfg.min_wall_run > 1, morph.remove_non_wall_components,
                 cfg.min_wall_run),
                (cfg.min_component_cells > 1, morph.remove_small_components,
                 cfg.min_component_cells)):
            if enabled and occ.any():
                kept, n = fn(occ, arg, cfg.component_connectivity)
                if n and self.last_confidence is not None:
                    self.last_confidence[occ & ~kept] = 0.0
                occ = kept
                n_speck += n

        # 7) compose label grid (OCC > FREE > UNK), keep openings free,
        #    then stamp caller-forced obstacles (manual/back walls) as OCC
        grid = np.full((lat.H, lat.W), UNKNOWN, np.int8)
        grid[observed_free] = FREE
        grid[occ] = OCCUPIED
        grid[protected] = FREE
        if force_occ is not None and force_occ.any():
            grid[force_occ] = OCCUPIED
            # Caller-forced obstacles are CERTAIN ground truth (manual walls, the
            # virtual back-wall), not temporally-accrued evidence -- stamp them to
            # full confidence so a downstream confidence gate never treats a known
            # wall as a low-confidence speckle to "keep looking at". Their evidence
            # (_ev) is 0, so without this they would read confidence 0 despite
            # being OCCUPIED, the exact inverse of the truth.
            if self.last_confidence is not None:
                self.last_confidence[force_occ] = 1.0

        # 8) optional safety dilation (never seal a protected opening)
        if cfg.occ_dilate_cells > 0:
            occ_all = grid == OCCUPIED
            new = morph.dilate4(occ_all, cfg.occ_dilate_cells) & ~occ_all & ~protected
            grid[new] = OCCUPIED

        self.last_stats = dict(
            raw=n_raw, confirmed=n_conf, fill=n_fill, pending=n_pending,
            speck=n_speck,
            occ=int((grid == OCCUPIED).sum()),
            free=int((grid == FREE).sum()),
            unknown=int((grid == UNKNOWN).sum()),
            openings=int(protected.sum()))
        return lat.spec(), grid


def _as_xyz(pts: np.ndarray) -> np.ndarray:
    """Coerce to finite (N,3) float32; empty-safe."""
    if pts is None or len(pts) == 0:
        return np.empty((0, 3), np.float32)
    a = np.asarray(pts, np.float32).reshape(-1, 3)
    return a[np.isfinite(a).all(axis=1)]