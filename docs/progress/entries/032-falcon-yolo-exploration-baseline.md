# 032 - The FALCON + YOLO + closing exploration baseline

**Branch:** `feat/falcon-yolo-baseline-nadav` (off `feat/objnav-habitat-gibson-nadav` at `59783f62`)
**Status:** in progress -- code, tests and docs done; the fifteen recorded episodes run; the 1,000-episode run
waits for the review of the fifteen
**Roadmap item:** ObjectNav exploration ablation -- goal-free FALCON + YOLO + the same closing, scored on the
Gibson val split against the full method

## Goal
A clear, simple baseline for the ZSON method on the Gibson benchmark: explore without a goal (planar FALCON, the
2-D adaptation Daphna ran goal-free on HM3D), run the same YOLO-World perception beside it, and the moment the
target is detected hand the episode to the **exact** closing the full method uses. Everything else -- the room
LLM, the node oracle's probabilities, RPT*, the scene graph, the room partition, doors, openings, peeks, glances,
the frontier sweep, the fallback rungs -- is removed. Spawn-floor confinement is the full method's own code
(seen ground-truth stairs and observed drops masked, off-plane goals refused). Fifteen recorded episodes first,
for review; then the published 1,000-episode split, with SR, SPL and DTG beside the full method's.

## Why
The full method's result (entry 030: SR 0.792, SPL 0.464, DTG 0.94 m) cannot say how much of it is the
exploration policy and how much the perception and the closing. A baseline that shares every part but the
exploration, run on the same episodes with the same seed, measures exactly that.

## What was taken from where
- From Daphna's `feat/multi_resolution_falcon_daphna`: the shape of the goal-free loop
  (`experiments/multi_resolution/run_gt_falcon.py`, `StandaloneFalcon`): the planar `FalconPlanner` planned in a
  disc, its first viewpoint walked through the route memory, retired on arrival, the ground-motion safety boundary
  (`GroundMotion`) vetoing swept steps, the bootstrap-turn option. Her core changes to the FALCON planner are
  world-coordinate diagnostics for her occupancy figures and were not needed; the planar FALCON core itself was
  already in the Gibson branch (`core/planning/exploration/falcon`, `methods/falcon_motion.py`).
- From the Gibson branch, unchanged: `ObservedMap`, `WeightedAStarPlanner2D`, `CommittedRoute`, `PerceptionCycle`,
  `TargetClosing` + `TargetApproachPath` + the legacy `_approach`, `CameraController`, `MultiFloorSearch` (the
  perfect stair detector and the portals, in its forbidden-traversal mode), `SpawnFloorGuard`, the `SightLedger`'s
  walked trail (so the stair mask never covers where the agent stood).
- New: `methods/falcon_yolo_baseline.py` -- the host policy (`FalconYoloBaseline`) that gives those components the
  attribute names they read in `rpt_policy`, and the two-level explorer (`GoalFreeFalcon`): bounded FALCON
  transactions in a 4 m disc about the agent, a relocation by A* to the nearest reachable frontier of the storey
  when the disc is exhausted, a full look-around and a STOP when the storey is.

## Steps
- [x] Branch off the Gibson branch; read Daphna's standalone FALCON and the Gibson method's component contracts.
- [x] `methods/falcon_yolo_baseline.py` with `BaselineSettings` (bootstrap turns, relocation knobs, exhaustion).
- [x] `gibson/run.py --agent falcon-yolo` (no LLM constructed or health-checked; `--explorer frontier` refused;
      `"baseline"` object in `--policy-config`); `five_scene.py --agent`; `run_benchmark.sh --agent` (no Ollama,
      GPU plan without the LLM, output named `..._falcon-yolo_...`); `launch_detached.sh` `RUN_ARGS`; the
      supervisor ensures no LLM on a baseline resume.
- [x] The recorder without rooms: `floor_panels.capture` tolerates a policy without a floor-context bank and a room
      graph; `method_snapshot` carries the explorer's snapshot and names the state `FALCON_<mode>`; the right-hand
      column is the baseline's (explorer, closing, stairs, boxes) when there is no search panel; lean records keep
      `exploration`.
- [x] `tests/test_falcon_yolo_baseline.py` (19): no LLM and the configuration's claims; FALCON from the first frame
      and the committed route; bootstrap turns; the two-frame takeover and STOP without consulting the exploration;
      weak and depthless boxes start nothing; the actual converter closes a toilet from 2 m, looks down and STOPs
      with the navmesh projection; blocked steps; relocation to the nearest frontier and its early end; the
      look-around and STOP on exhaustion, and the no-STOP option; a decision failure answered with a turn; the seen
      staircase impassable on the planning world, FALCON's route clear of it, the guard refusing a goal on it, the
      swept-motion veto at its foot; an unplanned height departure starting no traversal; the snapshot, the floor
      panels, the dashboard column and the lean record. The runtime suite: 877 pass (one pre-existing
      `test_monitor` timing failure, also on the untouched base).
- [x] Smoke: Darden/000000 (bed), recorded, 500 actions in 104 s; the closing took over twice and released as
      `unverified`, as the full method's run of the same episode (STOP at 2.3 m) suggests it would.
- [x] The fifteen: `RUNS_ROOT=$REPO/runs gibson/run_benchmark.sh --agent falcon-yolo --five-scene 3` ->
      `runs/benchmark_falcon-yolo_15episodes_20261018T145952Z/` (3 per scene in split order, seed 17; the same
      fifteen the full method's 1,000-episode run scored). Result below.
- [ ] Review the fifteen recordings; adjust what the videos say needs adjusting (candidates: the turn-heavy
      viewpoint facing, the 4 m disc, bootstrap turns, the exhaustion STOP).
- [ ] The 1,000-episode run: `PROGRESS_ENTRY=docs/progress/entries/032-falcon-yolo-exploration-baseline.md
      RUN_ARGS="--agent falcon-yolo" gibson/launch_detached.sh runs/gibson_val_falcon-yolo_<stamp>`; record SR,
      SPL, DTG overall, per scene and per category here beside entry 030's, and publish `results/<run>/`.

## The fifteen

See `runs/benchmark_falcon-yolo_15episodes_20261018T145952Z/summary.txt` and the table appended below when the
campaign ends. The full method's rows for the same episodes (from `results/gibson_val_20261016_small_llm/episodes.csv`):
Collierville 000000 toilet 1/0.83, 000001 toilet 1/1.00, 000002 chair 1/1.00; Corozal 000000 bed 1/0.98, 000001 tv
1/0.71, 000002 couch 1/0.98; Darden 000000 bed 0/0.00, 000001 bed 1/1.00, 000002 chair 1/0.60; Markleeville 000000
bed 1/0.23, 000001 bed 1/0.16, 000002 chair 1/1.00; Wiconisco 000000 toilet 0/0.00, 000001 potted plant 1/0.12,
000002 couch 1/1.00 (SR/SPL).

## Open items
- The explorer turns a lot: every FALCON viewpoint is faced at its own yaw (the gain-maximising heading) before it
  is retired, and the converter's 30-degree turns quantise the route headings. 321 turns to 161 forward steps in
  the smoke episode. Whether that costs SPL against the full method's frontier walk is what the fifteen say.
- The 4 m disc (`falcon.scope_radius_m`, FALCON's bounded box as Daphna and the hybrid ran it) keeps every FALCON
  transaction under ~0.8 s; a bigger disc buys longer local plans at the planner's capacity caps.
- `exhausted_turns` STOPs a storey the explorer calls finished; a boxed-in agent (inflation on a narrow spot) would
  STOP too, and the recording's `exploration.stats` says which it was.
