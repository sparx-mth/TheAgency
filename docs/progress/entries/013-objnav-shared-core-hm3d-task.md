# 013 - ObjectNav: one navigation core, GibsonTask and HM3DTask

**Branch:** `feat/objnav-habitat-hm3d-nadav` (merge of `feat/objnav-habitat-gibson-nadav`)
**Status:** done (code + docs); first HM3D smoke run with real scenes pending
**Roadmap item:** [ObjectNav robustness](../ROADMAP.md)

## Goal

Run the updated navigation algorithm from the Gibson branch -- RPT\* room order,
the room-search loop (011), the exploration fallback and ground-truth stairs
(012), YOLO-World as the default detector -- on HM3D ObjectNav v1/v2, without
forking it. Two task modules (`gibson/`, `hm3d/`) own their environments,
dataset loaders and protocols; one shared core (`methods/`, `habitat/`,
`evaluation.py`, `recording.py`) owns the algorithm, and neither task may be
imported by the core.

## Steps

- [x] Merge `feat/objnav-habitat-gibson-nadav` into the existing
      `feat/objnav-habitat-hm3d-nadav` (no new branch); take the shared core from
      Gibson verbatim, keep `hm3d/` intact.
- [x] `habitat/simulator.py`: union of both sides -- explicit `height_m` and the
      `navmesh="published"|"agent"` choice with `pathfinder`/`navmesh_provenance()`
      (HM3D), plus `scene_structure()`, `last_collision` and RGB/depth extrinsic
      checks (Gibson). Gibson callers pass `PROTOCOL.agent_height_m` (new field,
      0.88 = the camera height it used before, so behaviour is unchanged).
- [x] Move `gibson/stair_connectors.py` → `habitat/stair_connectors.py` so the
      core stops importing a task module; leave a re-export shim.
- [x] `HM3DEnv.reset` attaches the navmesh storeys/stair connectors to
      `ObjNavEpisode.metadata` exactly as `GibsonEnv` does (the policy's declared
      ground-truth stair source); a synthetic simulator still yields `{}`.
- [x] `hm3d/run.py`: `--explorer frontier|falcon`, `--detector-backend`
      (default `yolo_world`), `--detector-timeout-s`; embodiment from
      `HM3DProtocol`; all recorded in `method`.
- [x] Keep both boundary test suites; register both label mappers; benchmark-
      neutral dashboard/runner/smoke wording; HM3D's `evaluation_diagnostics`
      hook and encoder-failure handling retained in `recording.py`.
- [x] Rewrite `hm3d/README.md` as the execution guide: setup & downloads,
      single recorded run, large-scale campaign / shards / freeze.
- [x] Rewrite the runtime `README.md` around the core / task layout.
- [ ] First recorded HM3D smoke episode with real v0.2 scenes (needs the
      licensed meshes; `hm3d.assets --check`), then a 36-scene `--limit 1` sweep.

## Verification

`PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` over the
runtime, objnav core, harness, planners, exploration, detection and scene-graph
tests: 2676 passed. The remaining failures pre-date the merge on their source
branches: five `test_hm3d_campaign` tests read the real split at
`~/datasets/objectnav/hm3d/v2/val` (absent here) and
`test_viz_render::test_door_labels_are_dropped_when_the_view_is_too_zoomed_out`
fails identically on the Gibson branch.

`hm3d.run._method` was exercised with fake service identities for
`--explorer` unset / `frontier` / `falcon`: the policy is the shared
`RPTSearchPolicy`, embodiment equals `HM3DProtocol` (0.88 m / 0.18 m), and
`configuration()` reports `ground_truth_stairs: true`, the loop, the fallback
and the `yolo_world` backend.

## Notes

- The HM3D branch's `RPTSettings` had *required* embodiment fields; the Gibson
  core gives them defaults (so the two tasks share one dataclass) and each task's
  `run.py` overrides them from its protocol. The boundary test was changed from
  "`RPTSettings()` raises" to "clearance below body radius raises".
- Old HM3D dashboards claimed "NOT a full Gibson benchmark" from the Gibson-era
  `dashboard.py`; the merged dashboard is benchmark-neutral and `test_demo`
  asserts the neutral wording.
- A `git stash` during conflict resolution silently dropped `MERGE_HEAD`; it was
  restored by hand so the commit records both parents. Don't stash mid-merge.

