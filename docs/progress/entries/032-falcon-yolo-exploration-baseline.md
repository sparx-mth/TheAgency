# 032 - The FALCON + YOLO + closing exploration baseline

**Branch:** `feat/falcon-yolo-baseline-nadav` (off `feat/objnav-habitat-gibson-nadav` at `59783f62`)
**Status:** done -- the fifteen recorded episodes and the 1,000-episode Gibson val run, both recorded below
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
- [x] `tests/test_falcon_yolo_baseline.py` (21): no LLM and the configuration's claims; FALCON from the first frame
      and the committed route; bootstrap turns; the two-frame takeover and STOP without consulting the exploration;
      weak and depthless boxes start nothing; the actual converter closes a toilet from 2 m, looks down and STOPs
      with the navmesh projection; blocked steps; relocation to the nearest frontier and its early end; the
      look-around and STOP on exhaustion, and the no-STOP option; a decision failure answered with a turn; the seen
      staircase impassable on the planning world, FALCON's route clear of it, the guard refusing a goal on it, the
      swept-motion veto at its foot; an unplanned height departure starting no traversal; the pursuit dropping the
      viewpoint without rejecting it and not held to the disc; an A* route not re-judged by the swept check while
      a FALCON route is; the snapshot, the floor panels, the dashboard column and the lean record. The runtime
      suite: 880 pass (one pre-existing `test_monitor` timing failure, also on the untouched base).
- [x] Smoke: Darden/000000 (bed), recorded, 500 actions in 104 s; the closing took over twice and released as
      `unverified`, as the full method's run of the same episode (STOP at 2.3 m) suggests it would.
- [x] The fifteen: `RUNS_ROOT=$REPO/runs gibson/run_benchmark.sh --agent falcon-yolo --five-scene 3` ->
      `runs/benchmark_falcon-yolo_15episodes_20261018T152349Z/` on `03d640ff` (3 per scene in split order, seed 17; the same
      fifteen the full method's 1,000-episode run scored). Two earlier campaigns of the day are parked beside it:
      `..._20261018T145952Z.partial_source_changed` (stopped at 10/15 when the source changed under it) and
      `..._20261018T150923Z.pre_doorway_fix` (15/15 on `5992840c`, SR 0.733 / SPL 0.420 / DTG 2.22 m, with
      Wiconisco/000000 lost in its first doorway -- the fix below). Result below.
- [x] The doorway fix (`03d640ff`): the swept-motion veto judges FALCON's own routes only. Wiconisco/000000 had spent
      179 vetoes oscillating in the doorway of its first room -- the A* relocation route squeezed through at the
      body radius, FALCON's stricter swept check refused every forward step, the route memory retired each
      relocation as `no_progress` after 31 actions and the next one chose the same doorway. The full method's
      frontier explorer never re-judges its A* routes; now neither does the baseline's relocation or pursuit. The
      episode then succeeds (320 steps, SPL 0.199) where the full method's run did not (500 steps, DTG 0.13 m).
- [x] The fifteen reviewed; no change before the full run.
- [x] The 1,000-episode run on `03d640ff`: `PROGRESS_ENTRY=... RUN_ARGS="--agent falcon-yolo" SUPERVISOR_ARGS=--dry-run
      gibson/launch_detached.sh runs/gibson_val_falcon-yolo_20261018T173302Z`, with a RAM/VRAM guard (stop above 80 %;
      peaks 56 % RAM, 52 % VRAM). Its frozen configuration matches the reference run's
      (`results/gibson_val_20261016_small_llm`, `f628a476`) in the 1,000 episode ids and their order, seed 17,
      protocol, kinematics, dataset, lean records, detector checkpoint and closing settings; no `.py` differs between
      `f628a476` and this branch's base. Published to `results/gibson_val_falcon-yolo_20261018T173302Z/`.
- [x] The run's three deaths (below) and the supervisor fix (`9e81d2c6`).

## The fifteen (`runs/benchmark_falcon-yolo_15episodes_20261018T152349Z`, `03d640ff`, 2026-10-18, laptop)

Baseline **SR 0.800, SPL 0.433, DTG 0.94 m** (mean steps 153, 31 s an episode at ~5 actions/s with YOLO on the
GPU) against the full method's **SR 0.867, SPL 0.641, DTG 0.16 m** on the same fifteen episodes
(`results/gibson_val_20261016_small_llm/episodes.csv`). Videos: `<Scene>/recordings/<id>/video.mp4`; the table is
`summary.txt`, the rows `metrics.csv` / `benchmark_results.json`.

| episode | target | baseline SR | SPL | DTG m | steps | full method SR | SPL | DTG m | steps |
|---|---|---|---|---|---|---|---|---|---|
| Collierville/000000 | toilet | 1 | 0.090 | 0.00 | 322 | 1 | 0.829 | 0.00 | 44 |
| Collierville/000001 | toilet | 1 | 1.000 | 0.00 | 13 | 1 | 1.000 | 0.00 | 14 |
| Collierville/000002 | chair | 1 | 0.245 | 0.00 | 96 | 1 | 1.000 | 0.00 | 26 |
| Corozal/000000 | bed | 1 | 0.829 | 0.00 | 87 | 1 | 0.976 | 0.00 | 41 |
| Corozal/000001 | tv | 1 | 0.735 | 0.00 | 47 | 1 | 0.706 | 0.00 | 51 |
| Corozal/000002 | couch | 1 | 0.765 | 0.00 | 56 | 1 | 0.983 | 0.00 | 42 |
| Darden/000000 | bed | 0 | 0.000 | 3.89 | 131 | 0 | 0.000 | 2.31 | 136 |
| Darden/000001 | bed | 1 | 0.402 | 0.00 | 164 | 1 | 1.000 | 0.00 | 83 |
| Darden/000002 | chair | 1 | 0.300 | 0.00 | 124 | 1 | 0.601 | 0.00 | 74 |
| Markleeville/000000 | bed | 0 | 0.000 | 4.99 | 302 | 1 | 0.234 | 0.00 | 253 |
| Markleeville/000001 | bed | 0 | 0.000 | 5.17 | 500 | 1 | 0.159 | 0.00 | 260 |
| Markleeville/000002 | chair | 1 | 0.929 | 0.00 | 24 | 1 | 1.000 | 0.00 | 18 |
| Wiconisco/000000 | toilet | 1 | 0.199 | 0.00 | 320 | 0 | 0.000 | 0.13 | 500 |
| Wiconisco/000001 | potted plant | 1 | 0.153 | 0.00 | 75 | 1 | 0.123 | 0.00 | 128 |
| Wiconisco/000002 | couch | 1 | 0.845 | 0.00 | 41 | 1 | 1.000 | 0.00 | 28 |
| **mean (15)** | | **0.800** | **0.433** | **0.94** | | **0.867** | **0.641** | **0.16** | |

What the three failures are (every one is the shared perception/closing or the explorer's horizon, none a dead loop):
- Darden/000000 (bed) and Markleeville/000000 (bed): the closing LOCKED and STOPped on a YOLO `bed` at 0.78 and 0.77
  confidence, 3.9 m and 5.0 m from the dataset's bed -- Markleeville's is the dining table (the frame shows it, with
  `desk 0.59` beside the `bed 0.64`). The full method failed Darden/000000 the same way (STOP at 2.3 m) and passed
  Markleeville/000000 by a long path (SPL 0.23); its room label is the one input the baseline's `_suspect` does not
  have to doubt a bed in a dining room.
- Markleeville/000001 (bed): 500 actions, 39 m walked, the bed never in frame: 53 FALCON plans, 19 viewpoints
  observed, one relocation; five takeovers released `unverified from a wider perspective`. The explorer's 4 m
  horizon kept it in the storey's near half.
- The successes that cost steps (Collierville/000000 toilet 322, Wiconisco/000000 toilet 320, Collierville/000002
  chair 96 against 26) are FALCON's greedy local gain criss-crossing a big room before the next doorway: SPL 0.433
  against 0.641 is mostly path length, not success.

## Open items
- The explorer turns a lot: every FALCON viewpoint is faced at its own yaw (the gain-maximising heading) before it
  is retired, and the converter's 30-degree turns quantise the route headings. 321 turns to 161 forward steps in
  the smoke episode. Whether that costs SPL against the full method's frontier walk is what the fifteen say.
- The 4 m disc (`falcon.scope_radius_m`, FALCON's bounded box as Daphna and the hybrid ran it) keeps every FALCON
  transaction under ~0.8 s; a bigger disc buys longer local plans at the planner's capacity caps.
- `exhausted_turns` STOPs a storey the explorer calls finished; a boxed-in agent (inflation on a narrow spot) would
  STOP too, and the recording's `exploration.stats` says which it was.

## The 1,000 episodes (`results/gibson_val_falcon-yolo_20261018T173302Z`)

**Baseline SR 0.665 (95 % CI 0.635-0.694), SPL 0.337 (0.315-0.358), DTG 1.75 m**, against the full method's
**SR 0.792, SPL 0.464, DTG 0.94 m** on the same 1,000 episodes. Terminations: 847 STOP, 149 step limit, 4 agent
errors (the shared closing's `target floor changed`, scored as failures as the reference's one was).

| group | n | baseline SR | SPL | DTG m | full method SR | SPL | DTG m |
|---|---|---|---|---|---|---|---|
| **overall** | 1000 | 0.665 | 0.337 | 1.75 | 0.792 | 0.464 | 0.94 |
| Collierville | 200 | 0.760 | 0.411 | 0.91 | 0.780 | 0.544 | 0.86 |
| Corozal | 200 | 0.740 | 0.371 | 1.67 | 0.800 | 0.479 | 1.26 |
| Darden | 200 | 0.580 | 0.257 | 2.13 | 0.815 | 0.416 | 0.72 |
| Markleeville | 200 | 0.600 | 0.279 | 2.06 | 0.750 | 0.367 | 1.16 |
| Wiconisco | 200 | 0.645 | 0.364 | 1.98 | 0.815 | 0.514 | 0.68 |
| bed | 123 | 0.301 | 0.117 | 4.12 | 0.528 | 0.256 | 2.76 |
| chair | 177 | 0.893 | 0.554 | 0.34 | 0.915 | 0.689 | 0.23 |
| couch | 229 | 0.729 | 0.445 | 0.85 | 0.786 | 0.579 | 0.73 |
| potted plant | 193 | 0.824 | 0.343 | 0.62 | 0.798 | 0.376 | 0.70 |
| toilet | 208 | 0.500 | 0.154 | 3.41 | 0.894 | 0.357 | 0.52 |
| tv | 70 | 0.571 | 0.341 | 2.27 | 0.643 | 0.445 | 2.07 |

The exploration policy accounts for 12.7 points of SR and 12.7 of SPL. The gap is largest where search matters:
toilet (0.500 vs 0.894) and bed (0.301 vs 0.528), the targets in their own rooms; chair, couch and potted plant,
usually in view from the common rooms, are within a few points (potted plant slightly above).

## The run's three deaths, and the supervisor fix
Three native segfaults in habitat-sim (0.2.4 under `--allow-sim-version-mismatch`), each at an episode's start:
Collierville/000114 (attempt 1), Wiconisco/000139 (attempt 2), and Wiconisco/000139 again on the resume (attempt 3).
The reference run died three times the same way (Darden/000173, Markleeville/000121, Wiconisco/000072). Wiconisco/000139
alone in a fresh process ran cleanly (SR 1, 228 steps): an intermittent fault of the simulator over a long process,
not the episode or the policy. The third crash left a zombie whose renderer threads never exited, and the supervisor --
waiting for the launcher process to end -- waited five hours. `supervise_benchmark.sh` (`9e81d2c6`) now kills a run
whose progress files have not changed for `STALL_MINUTES` (20) and counts it as a death, and calls an episode
deterministic after `SAME_EPISODE_DEATHS` (3) deaths on it rather than two in a row. The resume completed the
remaining 61 episodes, Wiconisco/000139 included. The supervisor's dry-run publish step then failed writing its
result files (it expects the full method's configuration); the files were published by hand.
