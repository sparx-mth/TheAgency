# 016 - Persistent target lock and continuous approach

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** complete; closing-control verification passed, benchmark SR=0
**Roadmap item:** ObjectNav robustness; follows 015

## Goal
Keep the confirmed observed 3D target and NavMesh standoff goal through occlusion;
continue collision-qualified A* until terminal proximity, refine on re-observation,
then stop moving, inspect pitch/yaw and explicitly STOP on fresh close confirmation.

## Steps
- [x] Trace the previous failure: missing detections replaced paths with scan holds;
  bbox centering competed with path turns; infrastructure exception lost metrics.
- [x] Persist the goal/path, refine only meaningful target changes, retain collision checks.
- [x] Stationary target-bearing-referenced terminal inspection at 1.0 m; fresh visual STOP.
- [x] Record expected method failures without crediting forced STOP as a successful close.
- [x] Regression tests including sustained occlusion, corner routing and terminal loss.
- [x] Exactly one recorded frozen toilet episode; audit trace, report SR/SPL/DTG and video.

## Constraints
Do not resume RPT*/LLM after takeover. Goal range and evaluator DTG remain separate.
Occlusion is not collision permission: depth mapping, route validation and native
collision checks remain enabled. No goal annotations are passed to the policy.
Use the same Ranchester/000000 start for comparison with the previous failure;
the manifest only scores downstairs reference-floor toilets, unlike all detections.

## Result
645 runtime/converter tests passed. The synthetic converter rollout covers occluded
forward motion past the old visibility timeout and fresh terminal LOOK_DOWN/STOP.
Refinement, path preservation, corner heading ownership and forced-failure STOP
distinction are regression-tested. Inspection uses a 0.05 m measurement tolerance
around the 1.0 m terminal radius; both filtered and fresh measured range must qualify.

Exactly one live replay: Ranchester/000000, seed 17, target toilet, recorded at
`runs/toilet_persistent_lock_20261001`. No runtime tuning or repeat episode.
- Takeover 169, confirmation 170; LLM=11, classifier=8, RPT*=11 remained frozen.
- Occluded path decisions 177/178/180 retained the same NavMesh goal and committed
  route. MOVE_FORWARD at 177 proved motion continued without a current detection.
- Fresh observations resumed and refined target XYZ; XY change stayed below the
  0.15 m endpoint-replan threshold in this episode.
- Stationary inspection: LOOK_DOWN at 183, TURN_RIGHT at 184, explicit policy STOP
  at 185; observed range 0.874049 m, yaw error 12.85 degrees within the discrete
  half-turn tolerance. No forward movement during inspection; no agent error.
- Official completed benchmark row: 186 actions, SR=0, SPL=0, DTG=10.892387 m,
  episode runtime=102.291134 s (process runtime 104.18 s). Upper-floor detection
  does not satisfy the downstairs reference-floor goal annotation.

Video: `runs/toilet_persistent_lock_20261001/episode/recordings/e7c4f2ad5402/video.mp4`.
Audit: `runs/toilet_persistent_lock_20261001/verification.json`. H.264 decode verified;
all decisions, terminal and final-metrics frames retained. Dedicated GPU services
were stopped afterward. This verifies the closing-control fix, not overall ObjectNav success.
