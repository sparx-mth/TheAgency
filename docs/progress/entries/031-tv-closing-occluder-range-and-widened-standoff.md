# 031 - The tv closing: a rail read as the television's range, and a standoff ring inside the cabinet

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- code, tests and docs; the two recorded episodes verified; the toilet regression campaign run
**Roadmap item:** ObjectNav target closing -- the range read around a surface in front of the object, and a
standoff that reaches a big object

## Goal
Fix the two target-closing failures the three recorded Collierville tv episodes of entry 030 exposed, in the
planning and state-machine logic rather than in detection: (1) 991eeffec5a5 (000018) -- a STOP 3.3 m from the
television after a lock whose range was wrong by 2.4 m; (2) 873d8cf1d32d (000009) -- three correct locks released
from 3 m as "no safe path to the target", a fourth released from 0.8 m as "saw nothing". Verify on the recordings,
check the toilet yardstick did not move, and leave the ground-truth tools behind for the next trace.

## Why
Both failures were deterministic -- the same five Collierville tv episodes failed at the same actions in two
1,000-episode runs -- and tv (n=70, SR 0.557) is the smallest category with the most leverage per fix. The
2026-10-15 re-run (entry 030) was stopped at 43 episodes rather than spend 34 hours confirming them.

## Steps
- [x] Ground truth beside the recordings: `runs/_analysis/closing_vs_truth.py` places the dataset's goal cells
      (val_info.pbz2, Habitat->world `x=-z, y=-x`, 5 cm cells) against the agent's pose and the closing's 3-D memory
      per step. 000018: memory on the television's bearing (+92 vs +90 degrees) but 2.40 m short of it. 000009:
      locks at 5/380/463 with the memory 0.02-0.14 m from the television, released for want of a path; the lock
      at 232 correct and 0.75 m from the goal, released `unverified` after six sweep turns.
- [x] The depth itself: `runs/_analysis/replay_depth_dissection.py` replays an episode in the simulator and dissects
      the box. 000018, lock frame: 79 % of the box at 4.0-4.25 m spanning 100 % of its height and width, 20 % at
      1.25-1.50 m in the lower middle continuing under the box; `robust_bbox_depth` 1.33 m, support 0.36 against the
      0.35 gate, heights 0.88-0.95 m -- a rail at camera height. The two `television` boxes on the frame without the
      rail: 4.1 m, support 1.00. 000009's correct locks unimodal; its phantom at 188 a near surface filling 82 % of
      its box (a detector false positive, out of scope).
- [x] `methods/box_occluder.py`: `occluder_in_front` -- bimodal (gap >= max(0.5 m, 30 %)), the far surface one
      component spanning >= 80 % of the box's height and width, covering more of the box than the near one, the near
      one's top at least a quarter of the box below its top edge. `perception._supported_centroid` sets the near
      pixels aside and reads the box again; `occluder_depth_m` / `occluder_px` in the row. Mask path untouched.
      Nine tests in `tests/test_box_occluder.py`: the rail frame fires and reads 4.1 m; a post reaching the box's
      top (a chair before a wall), a far part spanning the width only (bed, couch), two parts 0.3 m apart (toilet),
      a near surface covering more than the far one, too little depth, the mask path -- none fires.
- [x] `target_path.TargetApproachPath`: the standoff ring first with every bearing and both planners; then wider
      rings in the front arc (0.825, 0.95 m) with the goal within the terminal radius (`widened`, `widenings`); and a
      goal a refinement made stale kept while it lies within terminal + tolerance when no ring point has a route
      (`stale_kept`). `_adopt` records how a standoff was found.
- [x] `TargetClosing`: `closest_approach_stop` (default on). From an ARRIVED widened standoff a fresh range within one
      forward step of the limit is the inspection's business (not a reason to drop the standoff for a closer ring
      that does not exist), the inspection's step rule steps, and when its steps are spent or a step was blocked
      (`note_blocked` -> `inspection_blocked`, the pending flag cleared on the next frame) the closing STOPs. Counters
      `widened_standoffs`, `stale_standoffs_kept`, `closest_approach_stops`, `standoff_widened` in the diagnostics.
      Six tests in `tests/test_target_closing.py` (the widening and its preference for the ring, the stale goal kept
      and not kept, the blocked and the spent STOP, the rule's narrowness and switch, the pending flag's lifetime).
- [x] Suites: the runtime package 860 pass under the Habitat interpreter (a throwaway `/tmp/habitat-test-venv` over
      `objnav-habitat` with pytest, since `.venv` lacks `skfmm`); the README suite under `.venv` 2493 pass with the
      13 known `skfmm` environment failures.
- [x] The three recorded episodes again (`runs/gibson_Collierville_tv_fix_20261015T064436Z`): 000018 SR 1, SPL 0.806,
      45 actions, STOP 0.37 m from the television (occluder set aside on 10 frames, lock at step 8 on the real
      television, one widened standoff); 000009 SR 1, SPL 1.000, 32 actions, STOP 0.38 m from it (one widened
      standoff, zero boxed releases; the approach-history STOP fired, the closest-approach STOP stayed in reserve);
      000014 unchanged (the portrait false positive, 6.8 m from the television).
- [x] The toilet yardstick: `run_benchmark.sh --five-scene 2 --category toilet --shuffle` (the seeded ten episodes
      of the 2026-10-14 verification runs, SR 0.90 / SPL 0.456 on `…170224Z`) -- result below.
- [x] README (the standoff paragraph and the terminal-range paragraph), CHANGELOG, LESSONS; entry 030 points here.

## Open questions
- The seventh takeover of 000009 (actions 232-239): at 1.3 m a 1.5 m television fills the frame, YOLO-World reads
  it 0.06 centred and 0.31 at the frame's edge, the edge frames are refused as clipped for an unlocked candidate,
  and the sweep releases a target 0.75 m away. A re-sighting rule accepting an edge-clipped box of the class on the
  anchor's bearing for an ACTIVE candidate (never to start one) is the likely follow-up; not done here.
- The occluder test's thresholds (80 % span, top quarter) are set from one scene's frames and the synthetic
  counter-examples; the full re-run (entry 030) is the population-level check, and `occluder_px` in the projection
  rows says how often it fires and on what.

## Notes
- The depth dissection is the tool to reach for first on a "range is wrong" closing failure: the per-step records
  hold the box and the memory, the simulator replays the frame, and the two together say whether the detector or
  the measurement was wrong before any code is touched.

## Result
The toilet yardstick (`runs/benchmark_toilet_10episodes_20261015T064732Z`, the seeded ten of
`--five-scene 2 --category toilet --shuffle`): **SR 0.90, SPL 0.530, DTG 0.23 m** against the 2026-10-14 reference
`…170224Z` at SR 0.90, SPL 0.456, DTG 0.59 m -- the same nine successes (Markleeville/000188 and Darden/000034 at
SPL 1.0), the same one failure (Corozal/000019, a 500-action timeout), and no episode cheaper by its SPL. The two
tv episodes of the title: 000018 SR 0 -> 1 (SPL 0.81), 000009 SR 0 -> 1 (SPL 1.00). The full re-run of entry 030
is the population-level check.


