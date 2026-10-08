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
