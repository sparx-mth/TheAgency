# 001 - FALCON click-to-fly working end-to-end for Rooster/Sphera

**Branch:** `fix/falcon-rooster-clickfly` (currently on `create_devcontainer_daphna` — not yet split out, see Notes)
**Status:** in-progress (core bugs fixed and individually verified; map still shows some noisy/speckled voxels under investigation)
**Roadmap item:** → `ROADMAP.md` "Now"

## Goal
Click a goal in FALCON's BEV view and have the Sphera-simulated Rooster (R1) actually
fly there, with a correctly-oriented occupancy map building live in RViz — not just
takeoff/hover, which already worked before this entry.

## Why
The Rooster/Sphera bring-up (takeoff, video streaming, depth `.npy` output, altitude
hold) was already confirmed working. The next real milestone is FALCON driving the
drone autonomously from a BEV click, which is the actual point of bridging FALCON in
as the mission planner. It didn't work at all going in — the drone just hovered in
place regardless of where you clicked.

## Steps
- [x] Diagnose why a BEV click produced no drone motion at all
- [x] Fix the click-to-fly deadlock (default `roll_assist` controller's demo-mode
      handshake never confirms for Rooster; switched to `multi_axis` + renamed the
      demo-mode topic to `/R1/...` convention)
- [x] Fix the real motion-blocking bug: `waypoint_follower_node.py` had
      `_publish_twist_multi` defined twice — the second (older) definition silently
      shadowed the first, and its signature didn't accept the `vz` kwarg the caller
      passed, so every tick threw a `TypeError` before ever calling `.publish()`.
      `/cmd_vel` was never actually emitted despite FALCON's internal state showing
      `nav=RUN`.
- [x] Fix the world-frame Y-axis handedness bug: `rooster_ground_truth_localization.py`
      only negated yaw for the Sphera/Unreal (left-handed) → ROS (right-handed)
      conversion, never position.y — verified live via a pure `forward` move
      (`dy/dx` only matched `tan(yaw)`'s sign after flipping `dy`).
- [x] Fix the consequential map/BEV bounds break: every Y bound in
      `maps/sphera_jail.yaml` and the `sphera_drone.launch`/skill-documented launch
      args was tuned against the old (unflipped) Y sign — the drone's real,
      unchanged spawn point now reports the opposite Y sign, which took it outside
      its own configured map. Negated + min/max-swapped every Y bound.
- [x] Document the always-crashes-shortly-after-start `exploration_node` vendor bug
      (Eigen assertion in `ExplorationFSM::visualize()` indexing `averages_` — a
      much smaller "top viewpoints" vector — by `frontiers_.size()`) as a known,
      restart-to-recover issue, not something in our code.
- [ ] **Open**: RViz's 3D voxel view still shows a noisy/speckled, non-room-like
      blob rather than clean planar walls, even after both fixes above and a fully
      clean `falcon`/`exploration_node` restart. Vendor CDR corruption
      (`invalid data size` / `string data is not null-terminated` in
      `rmw_cyclonedds_cpp`) was confirmed recurring on `R1`-side nodes
      (`video_handler`, `fcu_driver`, `rooster_manager`,
      `sphera_physical_rooster_backend_node`) during the flight window that produced
      this map, and neither `mapping_sync` nor `exploration_node` do any content
      sanity-checking on depth values — only pose/timing gating — so corrupted
      frames pass straight into the map. A quick spot-check of the 30 most recent
      `/tmp/rooster_depth/*.npy` frames showed no NaNs and sane min/max/mean values,
      so the corruption (if it's the cause) isn't leaving an obvious NaN trail in
      the saved depth arrays — needs more investigation before concluding it's
      fully explained.

## Open questions
- Is the noisy voxel map fully explained by the vendor CDR corruption, or is there
  a second, still-unidentified issue (e.g. in how corrupted frames specifically
  translate into bad 3D points, given the saved `.npy` depth values look sane)?
- Which of the two conflicting battery-capacity config files
  (`~/rqs7-private-parameters/developer.params.yaml` at `10000.0` vs.
  `~/rooster-private-parameters/developer.params.yaml` at `1000.0`) does Sphera's
  simulator core actually read? Not yet resolved — raising the lower one to match
  was proposed but not yet applied pending user decision.

## Notes
- 2026-07-27: All of today's fixes ended up on `create_devcontainer_daphna` (the
  branch already checked out from earlier devcontainer/model-registry work), not a
  fresh `fix/` branch, since the session was live debugging against a running
  simulator rather than a clean planned task. Should be split onto its own
  `fix/falcon-rooster-clickfly` branch before merging, per the branch-per-item
  policy — flagged here rather than done silently mid-session.
- 2026-07-27: Multiple full Sphera restarts were needed during this session,
  each requiring a specific re-bring-up sequence (`ros1_bridge`,
  `rooster_command_unit.py`, video trigger, and — new discovery this session —
  a full `falcon` container recreation whenever the map itself needs to be clean,
  not just a node restart). This sequence is now documented in the
  `fly-rooster-sphera` skill's "After a Sphera restart" section.

## Result
Core click-to-fly motion path confirmed working (drone visibly moved toward a
clicked goal, `cmd_vel_gate`'s passed-command counter climbed continuously instead
of freezing). Y-axis handedness and map-bounds fixes verified with quantitative
ground-truth tests. Map-quality issue (noisy voxels, likely vendor corruption)
remains open for next session. See `CHANGELOG.md` and `LESSONS.md` for the
individual fixes and debugging detail.
