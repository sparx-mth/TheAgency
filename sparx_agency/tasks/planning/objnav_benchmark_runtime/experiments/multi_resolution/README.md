# Multi-Resolution FALCON

`run_gt_falcon.py` runs the planar FALCON adapter without RPT, an LLM, or an object detector. HM3D goal viewpoints are evaluator-only; success requires both a distance threshold and heading tolerance. The same-floor policy uses navmesh stair connectors. It does not use target coordinates for navigation.

The occupancy resolution is selectable with `--map-resolution-m`; FALCON's region cell size remains separate. The default coarse profile uses 1.0 m occupancy cells. A 12-turn RGB-D bootstrap scan runs before planning and counts toward the action limit.

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=/home/daphnaa/GIT/TheAgency \
/home/daphnaa/miniconda3/envs/objnav-habitat/bin/python \
-m sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.run_gt_falcon \
  --episodes-dir /path/to/objectnav/hm3d/v2/val \
  --scene-release-root /path/to/versioned_data/hm3d-0.2 \
  --scene-directory hm3d \
  --scene 00800-TEEsavR23oF \
  --episode-id 00800-TEEsavR23oF/000001 \
  --map-resolution-m 0.2 \
  --bootstrap-turns 12 \
  --max-actions 100 \
  --output-dir /path/to/results/multires
```

Repeat `--scene` and `--episode-id` to run several houses/episodes. Each episode gets a directory at `<output-dir>/<scene>/<episode-index>/` containing:

- `run.json`: configuration, evaluation, transfer state, reference-frame status, and action trace.
- `sample.mp4`: RGB per action with target category, action, FALCON status, and evaluator-only POV error. On success the final frame is the traversed POV; on failure the last frame is labelled `GT POV REFERENCE - NOT TRAVERSED`.
- `occupancy_map.png` and `occupancy_map.npz`: final occupancy, executed path, route lifecycle styles, frontiers, candidate viewpoints, connectivity groups/edges, and raw grid/world origin.
- `trajectory.csv`: action, world pose, evaluator distance/heading, and planner status at each step.
- `routes.json`: planned route polylines and reached/invalidated status.
- `frontiers.json`: per-plan frontier clusters, candidate views, connectivity group samples, and graph edges in world coordinates.

GT viewpoint poses appear only in the evaluator and labelled video reference frame; they are excluded from the FALCON state and transfer bundle. Fine sessions must rebuild occupancy detail, frontiers, connectivity, and executable routes from fine-resolution observations. HM3D v2 ObjectNav supplies its own target viewpoints; it does not provide pen/scissors goals or test detector recognition.

## Coarse-to-fine transfer

`run_transfer.py` explores at a coarse resolution (no target, no detector, no LLM), writes the coarse map, seeds it sparsely into a finer grid (`core/planning/exploration/prior_map.py`: one fine cell per known coarse cell, `--representative center|random`), then explores again at the fine resolution.

```bash
PYTHONPATH=/path/to/TheAgency /path/to/objnav-habitat/bin/python \
-m sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.run_transfer \
  --episodes-dir <objectnav/hm3d/v2/train> --scene-release-root <scenes root> --scene-directory hm3d \
  --scene <scene folder> --coarse-profile prescan_0p3.json --fine-profile fine_0p1.json \
  --mode transfer --output-dir <results>
```

`--mode coarse` runs the prescan only; `--mode cold` runs the fine phase with no prior.

- Both profiles must share `map_size_m`, and the resolutions must nest on it (79.8 m for 0.3 -> 0.1).
- Each phase writes `occupancy_map.npz` (with the sensor-observed mask), a PNG, `trajectory.csv` and `phase.json` (stop reason, plan statistics, coverage and a coverage-at-N-actions curve).
- Coverage is measured per navigable lattice point against the navmesh by `transfer_eval.py`, which is evaluator-only: the policy never imports it.
- Profiles: `prescan_0p5.json` and `prescan_0p3.json` (coarse), `fine_0p25.json` and `fine_0p1.json` (fine).
- Findings so far: a 0.5 m prescan stops early on most scenes; 0.3 m is workable; the sparse one-in-nine seed made three of six fine runs hit `planning_deadline` on their first plan.
