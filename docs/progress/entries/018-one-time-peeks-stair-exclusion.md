# 018 - One-time room peeks and stair exclusion

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; core/runtime regressions and headless stair probes passed
**Roadmap item:** ObjectNav robustness; follows 017

## Diagnosis
The retained Sofa trace shows a room-7 `doorway_peek_approach`, started at 293,
not a committed stair traversal. Steps 297/299/301/302 move forward; step 303
turns to correct a 17.19-degree heading error. At 304 the converter requests
MOVE_FORWARD (error -12.81 degrees) but `floor_coverage_guard` vetoes it to
TURN_LEFT, cancels the peek and clears the route. Subsequent `transit/7` plans
hit the same veto and oscillate. Floor height remains 2.6394 m; transition is null.

## Goal
At most one peek attempt per episode-local room identity, consumed on start even
if interrupted. Skip scanned, initially classified (including weak/provisional)
and previously peeked rooms. Keep these exemptions consistent with floor eligibility.
Exclude seen stair connectors from peek goals/routes and all committed transitions
from peek ownership. Preserve collision checks and forward-priority angular deadband.

## Steps
- [x] Read exact trace around 297-303 and the first veto at 304.
- [x] Implement monotonic one-time peek/exemption evidence across room renumbering.
- [x] Exclude stair geometry and transitions from peek selection/routing.
- [x] Prevent the floor gate from requiring exempt peeks; preserve legitimate stair safety.
- [x] Validate forward-priority deadband and committed stairs with deterministic regressions.
- [x] Update documentation and log test/probe results.

## Verification scope
Initial implementation validation used CPU tests and headless NavMesh connector
probes only. A subsequent single recorded episode was explicitly authorized;
its separate result is recorded below. Existing videos/metrics remain unchanged.

## Result
`room_peeked` latches at the first attempt, including a failed or interrupted one.
An initial non-unknown room label is enough to exempt it regardless of confidence;
single confirmed-object clues are queried before peek selection. `done` still
means a completed measured scan, not merely an exemption. Consumed allowances are
floor-qualified and retained through persistent IDs and 60%-smaller-region overlap;
disjoint rooms/floors do not inherit them. Configuring more than one attempt is refused.

`peek_stairs.py` clips seen connector polylines to the current elevation and
excludes them from candidates and peek-only route maps, leaving normal stair maps
untouched. Active transitions preempt peeks without restoring a stale room route.
The floor guard now releases vetoed room transits instead of repeatedly replanning
them. Converter hysteresis is optional (generic default 0 degrees, ObjectNav 1),
path-only and bounded; blocked recovery, pitch, final facing and STOP retain priority.

Validation: **1,206 passed**, 22 existing deprecation warnings, across runtime and
all `core/planning/objnav` tests (including Python 3.8/import contract tests).
CPU native-NavMesh Ranchester probes: **up 37 actions; down 38 actions; zero
blocked steps, recoveries and retreats**. The probes use the production 1-degree
margin and create no renderer. This does not claim a successful end-to-end Sofa
episode; none was launched during implementation.

## Authorized Sofa re-verification — 2026-10-01 14:54
Exactly one episode ran on the active branch from the previous exact spawn,
using `runs/sofa_same_spawn_20261001/episodes.json` (lower-floor Sofa/couch goals).
Frozen preflight confirmed `max_attempts=1` and 1-degree converter hysteresis.
Actual first-frame position, yaw and camera pitch match the previous Sofa run.

- Official result: **SR=0, SPL=0, DTG=12.286849 m**, 51 actions, runtime
  **30.680358 s**, average FPS 1.6623, `termination=agent_error`.
- A sofa candidate at step 38 (confidence 0.5387) triggered VERIFY. Subsequent
  views fell below the 0.50 threshold or contained no sofa detection. Qualifying
  views at 42/46/50 were nonconsecutive; the 12-action confirmation allowance expired.
- Error: `candidate never received consecutive depth-consistent confirmations`.
  The evaluator's final STOP was forced on failure, not a policy success STOP.
- The first room scan completed; two other peek attempts ended immediately on
  initial classification. Each record has at most one attempt, and final peek
  eligibility had no pending rooms. No stairs were seen, no transition occurred,
  and height remained 2.639427 m. Stair descent and lower-floor closing were not exercised.

Full HUD video:
`runs/sofa_reverify_20261001T145403/episode/recordings/e7c4f2ad5402/video.mp4`.
Official row: `episode/episodes.jsonl` under that run directory; detailed audit:
`runs/sofa_reverify_20261001T145403/verification.json`.
Video validated: H.264, 1600x900, 6 FPS, 52 frames (50 policy decisions plus
forced-terminal and metrics frames). Completion alert printed. Dedicated detector
and LLM service groups stopped automatically. No retries or runtime tuning.

