# 017 - Same-spawn lower-floor sofa verification

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** complete; verification failed at the floor-exit coverage gate
**Roadmap item:** ObjectNav robustness; follows 016

## Goal
Run exactly one Sofa episode from the identical Ranchester spawn and orientation
used in the previous toilet replay. Verify voluntary RPT* stair selection, clean
lower-floor arrival, persistent sofa closing and explicit policy STOP.

## Setup
- Output: `runs/sofa_same_spawn_20261001`.
- Derived frozen manifest copies the prior toilet start position and quaternion
  exactly, while using the existing `couch` category (Sofa alias) and unchanged
  lower-floor couch goal-region samples and source asset hashes.
- Native spawn: `[2.9731760025024414, 2.6394269466400146, -11.489082336425781]`.
- Quaternion wxyz: `[-0.9979774865099824, 0, 0.06356836020551236, 0]`.
- Goal-floor height: 0.0394269973 m; spawn height: 2.6394269466 m.
- Initial sofa geodesic: 14.2276315689 m, recomputed on the evaluator NavMesh.
- Seed 17, current frontier/RPT* policy, 500-action limit, full 6 FPS recording.
- No policy changes, stair forcing, target-coordinate injection or retry.

## Steps
- [x] Validate identical prior spawn and lower-floor sofa annotations without sim execution.
- [x] Freeze preflight configuration and run exactly one episode.
- [x] Audit room exploration, stair selection source, descent/floor arrival and loops.
- [x] Audit lower-floor target lock, occluded path continuity and actual policy STOP.
- [x] Validate video, report official SR/SPL/DTG and exact path; stop dedicated services.

## Result
One episode completed normally at the 500-action limit, with no agent exception:
SR=0, SPL=0, DTG=12.58522892 m, runtime=398.304279 s, average FPS=1.2553.
The actual initial ENU pose (position, yaw and camera pitch) equals the previous
toilet recording exactly; method configuration and Python source fingerprints match.

The traversable down-stair connector was seen at action 135 and a portal placed.
However, no RPT* stair selection, committed transition or height change occurred:
height stayed 2.6394269466 m throughout. The coverage gate vetoed floor departure
62 times; final pending rooms were `[3, 6, 7, 8, 10]` out of ten known rooms.
The trace recorded 324 doorway-peek decisions, 171 room-transit decisions and five
search decisions; 23 peeks started, 11 completed and 12 were cancelled (nine
`room_disappeared`, three `floor_coverage_guard`).

Descent, stair-rotation-loop behavior, lower-floor sofa closing and explicit STOP
were **not exercised**. Target lock never activated and no STOP was issued. This
does not establish a fault in the stair traversal controller: it was never entered.
No retry, forced descent or policy change was performed.

Full HUD video (H.264 1600x900, 6 FPS, all 500 decisions plus terminal/metrics frames):
`runs/sofa_same_spawn_20261001/episode/recordings/e7c4f2ad5402/video.mp4`.
Audit: `runs/sofa_same_spawn_20261001/verification.json`.
Spawn/goal provenance: `runs/sofa_same_spawn_20261001/spawn_goal_verification.json`.
Video decode succeeded and dedicated detector/LLM workers were stopped afterward.


