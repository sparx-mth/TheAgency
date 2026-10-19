# 033 - HM3D under the Gibson-validated method

**Branch:** `feat/objnav-habitat-hm3d-nadav` (merge of `feat/objnav-benchmark-infra-nadav` at `b3c2a5b3`)
**Status:** the merge, the launcher and every mesh-free check are done; the three recorded HM3D runs and the
launcher's first real scene are blocked on the licensed HM3D scenes, which are not on this laptop (below)
**Roadmap item:** the same object-search method on HM3D v1/v2, then MP3D, then AI2-THOR, under each benchmark's
own published protocol

## Goal
Run the method that produced the Gibson val result (entry 030: SR 0.792, SPL 0.464, DTG 0.94 m) on HM3D
ObjectNav v1 and v2 **without changing it** -- the algorithm is what the Gibson branch has evolved into, now
`core/planning/object_search/`; everything HM3D is an adapter (`objnav_benchmark_runtime/hm3d/`) under
habitat-lab's own protocol: the six discrete actions at 0.25 m / 30° / 30°, 640x480 at 79° HFOV from 0.88 m,
radius 0.18 m, depth 0.5-5 m, no sliding, 500 steps, success = STOP and the geodesic to the nearest goal view
point under 0.1 m, on the navmesh habitat-sim recomputes at the agent's dimensions.

## What was decided in the merge
- **`core/` is the infra branch's, byte for byte**, plus the HM3D label table and its registry line. Two edits
  the HM3D branch had made to the method were dropped on purpose: the `"goal_changed"` exemption of the
  no-progress watchdog in `route_memory.py` (a behaviour change the Gibson number was never measured with) and
  `target_closing_enabled` (a no-op knob whose only consumer, `hm3d/falcon_only.py`, injects a detector that
  reports no box, so the closing cannot activate anyway). If HM3D shows the watchdog disarming across goal
  changes, that commit (`ec88f935`) is the candidate -- and it would be a change to the algorithm, to be made
  on both benchmarks or neither.
- **The Habitat bridge is the HM3D branch's superset**: body height separate from the camera mount,
  `navmesh="published"|"agent"`, `pathfinder` for the evaluator's geodesic, `navmesh_provenance()` in the frozen
  configuration. The reset-time height reconciliation it added is now applied only under `"agent"`; under
  `"published"` the start is placed verbatim, exactly as every Gibson run did, so Gibson's `reset()` contract
  (start moved by more than 0.1 mm raises) still holds.
- **The shared runtime** keeps whichever side was benchmark-neutral: the HM3D branch's dashboard, recorder and
  `run_evaluation` seams; the infra branch's `policy_factory` (with the `falcon-yolo` baseline, now offered by
  `hm3d.run --agent` too), `detector_options`, `gpu_plan`, `progress`, `visualization` and `services/`.
- HM3D's label table was already the Gibson table's twin (`e808a1bd`): the same 30 detector prompts, the same
  LLM words (`couch`, `tv`, `potted plant`), the same pitch range and depth-hole rule. Its tests had not been
  brought up to that commit; they are now.

## What was verified (laptop `LP-Boston-17093`, 2026-10-19)
- Tests: 1,078 of the runtime rig and 1,609 of `core/planning/object_search` + `objnav` + the harness pass with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. The one failure, `test_monitor.py::test_finished_is_judged_by_the_pid...`,
  is a pre-existing timing flake (the kernel's coarse file-timestamp clock lags `time.time()` by ~1.5 ms here);
  it passes in a worktree of the infra branch by luck of timing and has nothing to do with the merge.
- `gibson/run_benchmark.sh --preflight` before and after it was made to source `benchmark_services.sh`:
  identical frozen configuration.
- `hm3d/run_benchmark.sh --preflight` (v2 campaign and v1 `--scene`) against the real detector and Ollama:
  the launcher placed the GPU, started a YOLO-World detector with the HM3D vocabulary on its own port, parsed
  the 1,000-episode v2 val split, built the policy and froze `hm3d-objectnav-v2-val/2` (habitat-sim 0.2.4,
  reference match, `navmesh: agent`). **The frozen method block differs from Gibson's in the detector's
  deployment (URL, device, class order) and in none of the 548 method keys.**
- `hm3d/run_benchmark.sh --smoke --scenes <one>` with a placeholder mesh: launcher -> `campaign.py` -> child
  `hm3d.run` with the exact arguments; the child failed only at `habitat_sim.Simulator` loading the placeholder,
  the campaign reported the scene without rows, the launcher recorded the exit and the resume hint.
- The bridge on a real mesh (Gibson's Darden, the only meshes on this laptop), both modes: `"published"` places
  the start verbatim (gap 0.0000 m); `"agent"` has habitat-sim recompute at 0.18 m / 0.88 m (recorded settings
  confirm it), places the start on that mesh, and four actions later the height has not moved.

## A finding on Gibson, not acted on
On Darden/000000 the published start is **not on the shipped navmesh** (built at habitat's legacy 0.10 m /
1.50 m): `snap_point` disagrees with it, and the first forward step under `navmesh="published"` jumps the agent
**+0.143 m** in height, where under `"agent"` the start is on the mesh and the height never moves. habitat-sim's
`Simulator._config_pathfinder` recomputes the navmesh whenever the agent's radius/height differ from the legacy
defaults, and SemExp's Gibson config sets `RADIUS 0.18 / HEIGHT 0.88` -- so the SemExp stack navigated a
recomputed mesh, and the episode starts were plainly generated on one. The Gibson adapter's `"published"` choice
is therefore a protocol question for the Gibson result, separate from this entry; it changes where the agent may
step, not the algorithm, and nothing here touches it.

## Blocked: the HM3D scenes
`~/datasets/objectnav/hm3d/{v1,v2}/{val,val_mini,train}` (the episodes) are here; `scene_datasets/hm3d*` (the
licensed meshes) are not -- no archive, no mount, no Matterport credentials in the environment. The HANDOFF's
"36/36 scenes present" was the office PC. Once `MP_API_ID` / `MP_API_SECRET` are available, section 1.2 of
`hm3d/README.md` downloads them (`habitat_sim.utils.datasets_download --uids hm3d_val_v0.2` and `hm3d_val_v0.1`,
plus the v2 renaming note), and then, from the repo root:

```bash
L=sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh
$L --preflight
$L --scene 00877-4ok3usBNeis --limit 1 --record       # the three recorded runs: three scenes, one episode each
$L --scene 00800-TEEsavR23oF --limit 1 --record
$L --scene 00808-y9hTuugGdiq --limit 1 --record
$L --smoke                                              # the whole split touched once, then
$L                                                      # the full published v2 val split
```

## Steps
- [x] Merge `feat/objnav-benchmark-infra-nadav` into `feat/objnav-habitat-hm3d-nadav`; `core/` from infra.
- [x] Re-point the HM3D adapter and its tests at `core/planning/object_search` and the shared runtime modules.
- [x] Gate the bridge's reset-time height reconciliation on `navmesh="agent"`.
- [x] `benchmark_services.sh` out of `gibson/run_benchmark.sh`; `hm3d/run_benchmark.sh` on it.
- [x] Preflight both launchers against the real services; compare the frozen method blocks.
- [ ] HM3D scenes on this machine (credentials).
- [ ] Three recorded HM3D episodes, then `--smoke`, then the v2 val split; v1 after it.
- [ ] MP3D, then AI2-THOR/RoboTHOR, the same way: adapter only.
