# The FALCON + YOLO + closing baseline (`--agent falcon-yolo`)

The exploration ablation of the Gibson ObjectNav method: **goal-free planar FALCON
until YOLO-World sees the target, then the full method's own closing.** No room
LLM, no RPT*, no scene graph, no room partition, no doors, openings, peeks,
glances, frontier sweep or probabilities. Everything that is not exploration is
shared with the full method verbatim, so a run of the two on the same episodes
isolates the exploration policy: observed map, A*, route memory, the YOLO
perception cycle, the `TargetClosing` takeover (verification, standoff, pitch
servo, inspection, STOP overshoot) with its legacy pursuit, the camera
controller, and the spawn-floor confinement.

Code: `core/planning/object_search/baselines/falcon_yolo.py` (`FalconYoloBaseline`, `GoalFreeFalcon`,
`BaselineSettings`); tests: `tests/test_falcon_yolo_baseline.py`; entry point:
`gibson/run.py --agent falcon-yolo`, forwarded by `five_scene.py`,
`run_benchmark.sh --agent falcon-yolo` and `RUN_ARGS="--agent falcon-yolo"
launch_detached.sh`. Origin: Daphna's `StandaloneFalcon`
(`experiments/multi_resolution/run_gt_falcon.py`, branch
`feat/multi_resolution_falcon_daphna`) ran the same planar FALCON core goal-free
on HM3D with a 4 m disc about the start; this takes that loop and runs it over a
whole Gibson storey, with the benchmark's perception and closing behind it.

## The loop, action by action

1. **Perception, then the closing's look** -- exactly the full method's first two
   lines: `PerceptionCycle.observe` (one YOLO-World frame), `TargetClosing.observe`.
   The moment a confident box of the target's class has a coherent 3-D projection
   the takeover is active and owns the episode until its STOP or its release;
   the exploration is not consulted while it does (`_plan`, as in `RPTSearchPolicy`).
2. **The map and the confinement** -- `ObservedMap.update`, then
   `SpawnFloorGuard.observe` writes every SEEN ground-truth staircase
   (`MultiFloorSearch`'s perfect detector and portals, unchanged) and every
   observed drop below the storey plane occupied on the world every decision
   plans on, and refuses any goal off the spawn plane. FALCON's frontiers, its
   routes and the swept-motion veto all read that confined world.
3. **The legacy pursuit** -- a confirmed target landmark the takeover did not start
   on is walked to and verified in place (`_approach`), as the full method does
   under its frontier explorer.
4. **Goal-free FALCON** (`GoalFreeFalcon.plan`):
   - a FALCON viewpoint in force is walked (the route memory keeps it through
     turns and map updates, drops it when the map obstructs it), and retired as
     observed on arrival facing its yaw;
   - else one bounded FALCON transaction (`core/planning/exploration/falcon`) in a
     disc of `falcon.scope_radius_m` (4 m) about the agent: frontier clusters,
     viewpoint sampling with the camera's visibility, connectivity zones, the
     CP/SOP order, the first viewpoint's free route. Its goal passes the guard,
     the route is adopted by the route memory;
   - else, with no useful frontier in the disc, a **relocation**: the frontier
     cells of the storey outside the disc, clustered, nearest first, and A* on
     the observed map to the first one with a route (up to
     `relocation_candidates` tried per action). The relocation ends on arrival,
     or early when no frontier cell is left within `relocation_done_m` of its
     goal; a goal without a route, or one the route memory calls `no_progress`,
     is retired for `relocation_retire_actions`;
   - else, with nothing anywhere, the agent turns in place -- a new frame can
     open a frontier -- and after `exhausted_turns` (12, a full look-around at
     30 degrees) STOPs (`stop_when_exhausted`; off keeps turning to the limit).
5. **Hooks** -- a blocked forward step rejects the FALCON viewpoint and marks the
   map (the closing's own hook when it is active); the swept-motion veto
   (`GroundMotion.permits` on the confined cost map, held to the disc while a
   FALCON route is walked) turns a forward step into TURN_LEFT; the camera
   controller writes the pitch as in the full method.

`BaselineSettings` (a nested `"baseline"` object in `--policy-config`):
`bootstrap_turns` (0; Daphna's `--bootstrap-turns`), `relocation_candidates` (8),
`relocation_retire_m` (1.0), `relocation_retire_actions` (60),
`relocation_done_m` (0.75), `exhausted_turns` (12), `stop_when_exhausted` (true).
Everything else is `RPTSettings` as for the full method (`falcon.scope_radius_m`,
`target_closing.*`, `allow_stair_traversal`, `step_down_m`, ...); a `frontier`
`local_exploration` is refused.

## What differs from the full method, by construction

- No room labels: `TargetClosing._suspect` runs the context check without a room
  type (height alone). No landmark nodes, no openings, no second pass.
- The frozen configuration carries `llm: null`, `room_llm: none`, `rpt_star: none`
  and `exploration: planar FALCON run goal-free ...`; the per-episode record has
  `exploration` (FALCON plans, viewpoints observed/rejected, relocations,
  exhaustion, safety vetoes; kept whole in `--lean` records) beside the shared
  `target_closing`, `spawn_floor_guard`, `building` and `perception` blocks.
- The recording's right-hand column shows the explorer's mode, FALCON's last
  verdict, the closing's phase and candidate, the stairs seen and the frame's
  boxes instead of the room-search panel; `steps.jsonl` rows carry
  `method.exploration`.

## Running it

```bash
# the plumbing: one recorded episode (the detector left running is reused)
gibson/run_benchmark.sh --agent falcon-yolo --scene Darden --limit 1 --record

# the review set: 3 recorded episodes per val scene (15), live dashboard
RUNS_ROOT=$REPO/runs gibson/run_benchmark.sh --agent falcon-yolo --five-scene 3

# the published split (1,000 episodes), detached and supervised as BENCHMARK.md 2.3
PROGRESS_ENTRY=docs/progress/entries/032-falcon-yolo-exploration-baseline.md \
RUN_ARGS="--agent falcon-yolo" gibson/launch_detached.sh runs/gibson_val_falcon-yolo_$(date -u +%Y%m%dT%H%M%SZ)
```

`--agent falcon-yolo` starts and checks no Ollama, pins the GPU plan's LLM to
the CPU (`GPU_PLAN_LLM=cpu`, `LLM_VRAM_MIB=0`), and names the output
`benchmark_falcon-yolo_...` / `gibson_val_falcon-yolo_...`. The supervisor
ensures the detector but no LLM on a resume. Scores land where they always do:
`summary.json` (`overall.success_rate`, `overall.spl`, `overall.distance_to_goal_m`,
per scene and per category), `episodes.jsonl`, `comparison.md`; for the
five-scene campaign `benchmark_results.{json,csv}` and `summary.txt`. Compare
against the full method's `gibson/results/gibson_val_20261016_small_llm/summary.json`
(SR 0.792, SPL 0.464, DTG 0.94 m) on the same 1,000 episodes and seed 17.

## The first fifteen (2026-10-18, laptop, 3 per scene in split order, seed 17)

`runs/benchmark_falcon-yolo_15episodes_20261018T152349Z/` on `03d640ff`: **SR 0.800, SPL 0.433, DTG 0.94 m**,
153 steps and 31 s an episode (~5 actions/s with YOLO on the GPU), against the
full method's SR 0.867, SPL 0.641, DTG 0.16 m on the same fifteen
(`results/gibson_val_20261016_small_llm/episodes.csv`). The table, the per-episode
rows and the failure anatomy are in progress entry 032; the videos are under
`<Scene>/recordings/<id>/video.mp4`. Two of the three failures are the shared
closing STOPping on a YOLO `bed` false positive (one of them the full method
failed too), the third the explorer's 4 m horizon never bringing the bed into
frame; the SPL gap is mostly FALCON's greedy local gain criss-crossing a big
room before the next doorway.
