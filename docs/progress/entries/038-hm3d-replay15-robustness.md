# 038 - Fifteen replayed HM3D failures: the bugs in the recordings, the fixes, the replay

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** done -- analysis and fixes 2026-10-25; the fixed replay is `runs/replay15_hm3d_v1_20261025_fixed`
(the original is `runs/replay15_hm3d_v1_20261025_merged`, HEAD `1e517dd7`, clean tree)
**Roadmap item:** ObjectNav -- the Gibson-proven method on HM3D v1 val (entry 033), made robust to what HM3D does
differently rather than tuned to one house

## Goal
Read all fifteen recordings of the merged replay (12 scenes, 15 episodes, **0/15**: 7 STOPs on the wrong object,
8 action limits), name every bug once, fix each with the smallest change that generalises, and replay the same
fifteen episodes with the same services (`qwen2.5:3b-instruct` in both LLM roles, YOLO-World X on the card,
Habitat on the CPU, seed 17).

## What the recordings said (one line per bug; the episodes that showed it)

The per-action log (`steps.jsonl`), the map journal (`review_recording.py map N`) and `trajectory.csv`'s
evaluator-only DTG were read before any frame. Four of the eight timeouts never searched at all.

| # | component | bug | episodes |
|---|---|---|---|
| 1 | multifloor | **Crash on every action after an accidental descent.** With the benchmark default `allow_stair_traversal=False` and the observed stair source (default since 2026-10-21), a committed route whose end lay past the top tread walked the agent three risers down; the atlas called the height a departure, `MultiFloorSearch.traversing` turned true from the atlas alone, and `plan()` dereferenced the traversal it had refused to start: `AttributeError: 'NoneType' object has no attribute 'plan'` -- 417, 108 and 941 recorded decision failures, the fallback crashing on the same line, the agent frozen (`T_LEFT` x454). Ground-truth mode never had it: Gibson never showed it. | 00814/000009, 00873/000049, 00873/000002 |
| 2 | planner / embodiment | **A squeeze the simulator refuses.** HM3D moves the agent without sliding (`allow_sliding: false`; Gibson slides). The planner's floor rung was the body radius, 0.18 m, which on the 0.1 m grid admits a 0.4 m gap; a 0.36 m body on a 30-degree heading grid cannot thread one without a lateral touch, and a touch refuses the whole step. Partial steps of 0.10, 0.04, 0.01 m show the agent creeping into the gap between a bed and a dresser, then every heading refused. | 00824/000066 (460 actions wedged), 00835/000006 (boxed in a hallway from action 17) |
| 3 | observed map | **The agent fences itself in.** A refused step writes an occupied cell 0.25 m ahead, a second within 0.5 m a sticky bar; under no-sliding the refusal is a lateral touch, so the marks land on the free corridor the agent just walked (36 bumps, 38 sticky cells around one spot) and the map reads "completely surrounded". | 00824/000066, 00890/000074 (the ring of singletons round the sofa corner) |
| 4 | escape probe | **A wedge with no candidate.** Only headings with a thin band and floor beyond were probed; the open side of the pocket ran 0.6 m to a bed too thick to probe and was never tried (113-173 degrees never stepped in 90 probes). | 00824/000066, 00890/000074 (207 holds) |
| 5 | fallback ladder | **Relocation for ever.** Only the LAST vantage point was excluded, so the farthest cell flipped between the two ends of a closed strip beside a bed: 458 relocations, the footing and escape rungs never reached, the strip's one thin exit never probed -- with a chair 1.1 m away at action 24. | 00835/000064 |
| 6 | detector + HM3D annotations | **Wrong-object STOPs.** An armchair that IS an HM3D chair (goal 427 of 00802, 0.25 m away) read as `sofa` 0.94 (STOP at action 3); a counter stool (no HM3D goal) as `chair` 0.86; a real wall TV, a high TV and a small nightstand plant that the dataset does not annotate; an upholstered daybed (beside bed goal 250) as `sofa` 0.91. | 00802/000058, 00824/000061, 00853/000018, 00880/000029, 00813/000017, 00876/000078 |
| 7 | detector (HM3D meshes) | **The goal TV at 0.28.** HM3D TVs are black, holed meshes; the goal TV never scored above 0.28 (floor 0.30, takeover 0.50) while the agent stood inside the 0.1 m success radius for 86 actions and within 0.3 m for 314. | 00890/000074 |
| 8 | closing | **Near miss, target occluded at the STOP.** The approach-history stop fired behind a lamp table at the sofa's end, 0.17 m outside the view-point set (`target_visible: false`). | 00800/000010 |
| 9 | protocol | **Sub-storey goals.** The only TV's view points lie 0.6-0.8 m below the spawn plane (a sunken room); the guard masks the drop and the floor is exhausted at 2.7 m. The half-level adoption exists for this but needs the agent to settle there. | 00835/000006 |
| 10 | search order | A toilet 4.3 m north; the 3B oracle sent the agent west and south through 17 rooms, never north (DTG 3.8 -> 11.1). No defect found beyond the model's judgement. | 00843/000097 |

## Fixes (minimal, each with a regression test)
- **#1** `multifloor_policy.py`: `_retreat_to_storey` -- with traversal forbidden and the atlas in transition, walk
  the storey trail (`source_trail`) back to the plane; a one-pose trail is extended two steps past its last pose;
  the return is confirmed (`FloorAtlas.settle`, `source_platform_returned`) the action the body is back at the
  plane height, since the atlas's translated-plateau test may never fire at a flight's head; bounded by
  `multifloor.retreat_actions`, given up once if the plane is not regained; the terrain cannot veto a step onto
  ground the body stood on; a refused step aims at the next trail point back and -- with no traversal of ours --
  reaches the probe's memory and the map as on the storey (the first replay of 00873/000002 pushed into one wall
  473 times because `notify_blocked` short-circuited while "traversing"). Events `retreat_started`,
  `retreat_ended`, `retreat_given_up`.
- **#2** `RPTSettings.min_clearance_m` (0.25 m): the planner's floor rung (`inflate_floor_m`) and the "passable"
  map every count, entry and frontier is measured on (`navigation_cost`, the fallback, the multifloor selectors,
  the loop's confined cost, FALCON's topology cost); the stair capsule judges its corridors at the rung. Its
  cost, found in the first replay: a doorway the map draws 0.4 m wide closes a wing at 0.25 m (00835/000064
  held five frontier goals a squeeze away; 00890/000074 released its locked TV as "no safe path"). Hence the
  **squeeze rung**: `RPTSearchPolicy.squeeze_planner` (the planner with the body radius as its floor) serves the
  fallback's `_squeeze` -- the frontier goals reachable at the body radius alone, tried only when nothing is
  reachable at the rung, before relocation -- and the closing's standoff rings before the planner through the
  unknown. Comfortable routes first; the squeeze when the alternative is giving up.
- **#3** `RPTSearchPolicy._walked_passable`: cells the body has stood on this storey (the sight ledger's poses,
  one cell about each) are free in the planning world; sticky bars lie two to three cells ahead of any pose and
  survive unless the body later stands on them.
- **#4** `EscapeSettings.min_pocket_m` (0.5 m): a heading with two steps of free floor before anything occupied is
  a candidate whatever lies beyond (`rule: open_run`).
- **#5** `FallbackSettings.relocation_revisit_m` (1.0 m): no vantage point of the current run of fallback actions
  twice; the memory clears when the floor's search takes an action in between.
- **#6** `labels/datasets/hm3d.py`: `armchair` a chair prompt, `monitor` a tv prompt, `stool` / `bar stool`
  distractors (HM3D's `chair` excludes stools; Gibson's COCO chairs include them, so its table keeps none); the
  alias tables (`detection_dedup`, `room_priors`, `revisable_room_labels`, `SHADOWLESS_CLASSES`) know the words.
  Probed on the recorded frames the detector still says sofa 0.94 / chair 0.72: these two STOPs are detector errors
  the vocabulary does not reach, and the unannotated TVs and plant are the dataset's.
- **#7-#10** not changed: a lower takeover threshold trades the broken-mesh TV for more wrong-object STOPs; the
  near miss, the sunken room and the oracle's order are open items.

## Result
The same fifteen episodes on the final code (`runs/replay15_hm3d_v1_20261025_fixed2`, same services, seed 17; the
first replay, `..._fixed`, ran while the fixes were still being made and found the squeeze and `notify_blocked`
items above):

Before: SR 0.000 (0/15), SPL 0.000, SoftSPL 0.046, DTG 5.34 m, 1466 decision failures, 137 refused steps.  
After:  SR 0.133 (2/15), SPL 0.018, SoftSPL 0.070, DTG 4.63 m, 0 decision failures, 170 refused steps.

| episode | target | before: outcome / actions / DTG | after: outcome / actions / DTG / SPL | decision failures before -> after | refused steps before -> after |
|---|---|---|---|---|---|
| 00800-TEEsavR23oF/000010 | sofa | wrong STOP / 133 / 0.17 m | wrong STOP / 88 / 0.12 m / 0.00 | 0 -> 0 | 3 -> 0 |
| 00802-wcojb4TFT35/000058 | sofa | wrong STOP / 3 / 5.05 m | wrong STOP / 3 / 5.05 m / 0.00 | 0 -> 0 | 0 -> 0 |
| 00813-svBbv1Pavdk/000017 | plant | wrong STOP / 20 / 2.58 m | wrong STOP / 20 / 2.58 m / 0.00 | 0 -> 0 | 0 -> 0 |
| 00814-p53SfW6mjZe/000009 | bed | action limit / 500 / 8.17 m | action limit / 500 / 2.88 m / 0.00 | 417 -> 0 | 0 -> 5 |
| 00824-Dd4bFSTQ8gi/000061 | chair | wrong STOP / 32 / 4.46 m | wrong STOP / 32 / 4.46 m / 0.00 | 0 -> 0 | 0 -> 0 |
| 00824-Dd4bFSTQ8gi/000066 | toilet | action limit / 500 / 6.03 m | **success** / 447 / 0.05 m / 0.09 | 0 -> 0 | 36 -> 4 |
| 00835-q3zU7Yy5E5s/000006 | tv_monitor | action limit / 500 / 2.70 m | action limit / 500 / 6.16 m / 0.00 | 0 -> 0 | 22 -> 22 |
| 00835-q3zU7Yy5E5s/000064 | chair | action limit / 500 / 2.98 m | action limit / 500 / 2.38 m / 0.00 | 0 -> 0 | 5 -> 71 |
| 00843-DYehNKdT76V/000097 | toilet | action limit / 500 / 11.06 m | action limit / 500 / 7.33 m / 0.00 | 0 -> 0 | 26 -> 16 |
| 00853-5cdEh9F2hJL/000018 | tv_monitor | wrong STOP / 29 / 13.46 m | wrong STOP / 29 / 13.46 m / 0.00 | 0 -> 0 | 0 -> 0 |
| 00873-bxsVRursffK/000002 | bed | action limit / 500 / 4.27 m | action limit / 500 / 3.23 m / 0.00 | 941 -> 0 | 0 -> 23 |
| 00873-bxsVRursffK/000049 | chair | action limit / 500 / 5.04 m | action limit / 500 / 8.11 m / 0.00 | 108 -> 0 | 17 -> 26 |
| 00876-mv2HUxq3B53/000078 | sofa | wrong STOP / 168 / 5.84 m | wrong STOP / 149 / 5.68 m / 0.00 | 0 -> 0 | 8 -> 0 |
| 00880-Nfvxx8J5NCo/000029 | tv_monitor | wrong STOP / 46 / 7.97 m | wrong STOP / 46 / 7.97 m / 0.00 | 0 -> 0 | 0 -> 0 |
| 00890-6s7QHgap2fW/000074 | tv_monitor | action limit / 500 / 0.27 m | **success** / 187 / 0.01 m / 0.19 | 0 -> 0 | 20 -> 3 |

- The two wins are the two "stuck" episodes the navigation fixes were for: the wedge (00824/000066, 460 actions
  of hold -> a toilet STOP at 0.05 m) and the TV the agent had stood on top of without a path to its standoff
  ring (00890/000074, STOP at 0.01 m, SPL 0.19).
- The three crash episodes run to the limit searching instead of spinning (1,466 -> 0 recorded decision
  failures; 00814 closes from 8.2 m to 2.9 m). 00873/000002 is a landing between a flight up (occupied) and a
  flight down (masked) whose bed is upstairs -- unwinnable under the single-storey rule without the half-level
  adoption, which needs a plateau to settle on; 00873/000049 passes its chair at 0.26 m with the detector never
  above 0.09 on it.
- The seven wrong-object STOPs reproduce action for action: they are the detector's and the annotations'
  (bug #6), beyond the policy's reach without a different detector or a lower threshold that would cost more
  than it buys. 00800 stops 0.12 m from the view-point set (0.17 m before).
- 00835/000064's doorway is physically marginal: the squeeze rung now brings the agent to it 100 times and the
  simulator refuses every step through it (71 refused steps), as it refused the original's one pass. 00835/000006's
  TV is below the storey plane (bug #9). 00843 is the oracle's order (bug #10).
- Deterministic: every episode whose decision path the fixes do not touch reproduces to the action.

The regression rig: 2,822 passed; the one failure (`test_monitor.py::test_finished_is_judged_by_the_pid...`) fails
on the pristine tree too (a timing test, unrelated).
