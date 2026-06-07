# `core/mapping/bev`

Project FALCON's fused **3D voxel map** down to a clean **2D bird's-eye-view occupancy grid**.
Pure numpy, no ROS, stateless (FALCON owns the temporal fusion).

**In:** two `(N,3)` world-frame voxel-centre clouds — occupied and known-free.
**Out:** `(GridSpec, int8 (H,W))` with values `{-1 unknown, 0 free, 100 occupied}`.

## Files

- **`projector.py`** — `BevProjector`. The pipeline and the only entry point:
  `project(occupied_xyz, free_xyz, force_occ=None) -> (GridSpec, grid)`. Exposes
  `.last_stats`. Stateless unless `temporal_filter` is on (then it holds the
  evidence accumulator). `force_occ` is an optional `(H,W)` bool mask the caller
  stamps as occupied after compose / before dilation (manual walls etc.).
- **`config.py`** — `BevConfig`. Every spatial parameter (below), validated on init.
- **`lattice.py`** — `BevLattice`. The fixed grid + z-layers + voxelisation (world points → dense 3D/2D arrays).
- **`morphology.py`** — pure-numpy operators: 3D neighbour count, 2D shift, wall-bridge, dilation.
- **`__init__.py`** — exports `BevConfig`, `BevProjector`, `UNKNOWN/FREE/OCCUPIED`.

The ROS1 node that wraps this inside FALCON's container lives in
`tasks/planning/falcon/adapter/scripts/bev_publisher_node.py`.

## Usage

```python
from sparx_agency.core.mapping.bev import BevConfig, BevProjector

proj = BevProjector(BevConfig(resolution_m=0.15, z_peak=1.0))
spec, grid = proj.project(occupied_xyz, free_xyz)   # (N,3) world-frame voxel centres
```

## Parameters (`BevConfig`)

**Geometry / IO**
- `resolution_m` (0.15) — metres per cell; match FALCON's voxel size.
- `x_min,x_max,y_min,y_max` (±12) — BEV bounds in world metres.
- `frame_id` ("world") — frame the grid is expressed in.
- `occ_dilate_cells` (0) — inflate occupied by N cells for safety margin; 0 = off.

**Height / column projection** (stage 1)
- `z_floor,z_ceil` (0.30, 2.20) — column z-range considered.
- `z_peak` (1.00) — flight altitude; per-voxel weight peaks here.
- `weight_profile` ("triangular") — `triangular` | `gaussian` | `flat`.
- `weight_sigma` (0.50) — gaussian width (only if profile = gaussian).
- `voxel_size_m` (None) — z-layer thickness; defaults to `resolution_m`.

**Occupancy decision** (stage 1)
- `occ_weight_thresh` (1.2) — min weighted column mass to call a cell occupied.
- `min_occ_voxels` (2) — min raw occupied voxels in the column for occupied.
- `min_free_voxels` (1) — min free voxels for a cell to read free.

**3D neighbour confirm** (stage 2 — kills monocular speckle)
- `confirm_3d` (True) — drop occupied voxels with too few occupied neighbours.
- `neighbors_3d` (6) — connectivity `6` | `18` | `26`.
- `min_occ_neighbors_3d` (1) — min occupied neighbours a voxel needs to survive.

**Door / opening protection** (stage 3)
- `protect_openings` (True) — never wall a cell that is open at flight height.
- `door_band_m` (0.60) — z-band around `z_peak` inspected for openness.
- `door_free_voxels` (2) — free voxels in the band ⇒ it is an opening.
- `door_occ_tol` (0) — max occupied voxels tolerated in the band.

**Wall completion** (stage 4)
- `wall_fill_mode` ("directional") — `off` | `directional` (close 1-cell gaps) | `count`.
- `wall_fill_neighbors` (5) — count mode: occupied 8-neighbours required to fill.
- `wall_fill_iters` (1) — max bridge width; keep small (1–2).

**Temporal hysteresis** (stage 5 — optional, makes the projector stateful)
- `temporal_filter` (False) — FALCON already fuses in time, so usually off.
- `t_inc`/`t_dec` (1.0/1.0) — per-frame evidence added for OCC / removed for FREE.
- `t_max` (5.0) — evidence ceiling; `t_on`/`t_off` (2.0/0.5) — Schmitt on/off thresholds.

To isolate one stage, set its gate to the disabling value (`confirm_3d=False`,
`protect_openings=False`, `wall_fill_mode="off"`, `temporal_filter=False`,
`occ_dilate_cells=0`).