# 015 - Irreversible target closing and explicit STOP

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented and CPU-tested; single live Habitat verification FAILED (target lost, no STOP)
**Roadmap item:** ObjectNav robustness (012/013)

## Goal
Freeze global exploration on a confident target candidate; verify across two
consecutive frames, latch target ownership, approach with RGB-D/A* and bbox
servoing, look down at nearby low targets, then explicitly STOP. Never return
to global exploration after takeover, including on detection/planning failures.

## Steps
- [x] Kill active evaluation launcher, simulator, detector, encoder and dedicated LLM workers.
- [x] Trace perception, camera ownership, action conversion and existing target rejection paths.
- [x] Implement configurable verification and irreversible target-closing state.
- [x] Bind local NavMesh projection without exposing evaluator goals or DTG.
- [x] Test takeover, failures, depth/pitch/yaw geometry and real discrete-action STOP.
- [x] Update runtime documentation and changelog.

## Constraints
Use existing coherent RGB-D projection and collision-qualified A*. Habitat turns
are quantized: half-turn yaw tolerance prevents servo oscillation. The simulator
NavMesh is privileged geometry and must be declared separately from observed-map
planning; no target annotations, success flag, or evaluator distance are consumed.
Do not restart the cancelled evaluation campaign. Partial recordings are retained.

## Result
640 tests passed across the runtime and discrete-action converter suites (22 new
target-closing cases). The synthetic low-toilet rollout uses the real converter,
RGB-D projection and A*, includes MOVE_FORWARD and LOOK_DOWN, and finishes with
STOP inside the threshold. CPU tests also cover two-frame deduplication, non-target
confidence, inconsistent depth, lost targets, planner exceptions, unreachable
NavMesh projections, yaw signs, and both frontier/FALCON takeover.

Habitat Python imports passed without creating a simulator. Final process checks
found no evaluation/detector/encoder/Ollama worker jobs and no GPU compute processes.
At that implementation checkpoint no new Habitat episode was started; this was control/geometry verification, not a
claim of improved SR/SPL. Existing non-target exploration fallback is preserved;
target closing errors instead terminate with an explicit agent error and retain
the latch until reset.

## 2026-10-01 - single toilet verification
Exactly one new episode was subsequently authorized and attempted: Ranchester/000000,
seed 17, frozen multistory manifest, target explicitly checked as `toilet`, all
decisions recorded with lock/range/runtime HUD. No retry or second episode.

- Takeover at step 169, two-frame confirmation at 170. Room LLM=11, classifier=8,
  RPT*=11 stayed unchanged through every remaining recorded action.
- Initial bbox yaw error improved from 23.90 degrees to 6.18 degrees; two A* plans
  and four forward steps reduced observed target range from 2.75 m to 1.76 m.
- Route following turned right away from the visible target at steps 176 and 178.
  Reacquisition then oscillated away from the last successful heading. `target lost`
  aborted at observation 203. No LOOK_DOWN (never below 1.2 m), no explicit STOP.
- `ObjNavInternalError` is a harness infrastructure error: the normal completion
  callback and benchmark row were NOT emitted. Separate post-hoc failure diagnostics
  report SR=0/SPL=0 for the failed attempt, last evaluator DTG=9.9854 m, approximate
  launch-to-traceback runtime=104.62 s; recorded episode time through the last action
  is 101.30 s. These are not a normally finalized benchmark result.
- The detected target was on the upper floor; the manifest scores reference-floor
  targets downstairs. Observed bbox range is not evaluator DTG.

Artifacts: `runs/toilet_closing_verify_20261001/verification.json` and
`episode/recordings/e7c4f2ad5402/{video.mp4,video_review.mp4}` under that run directory.
Original video has 203 frames; review copy has 24 additional offline failure-card
frames, explicitly labeled. Both are H.264, 1600x900, 6 FPS and decode successfully.
32 focused regressions passed before this run. Only HUD/trace instrumentation changed;
navigation settings/code were not tuned during or after the episode. Dedicated services stopped.

