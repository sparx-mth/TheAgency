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
- [x] Root-caused the noisy/speckled voxel map from 2026-07-27: the bottom ~25%
      of every RGB frame showed a near-constant `~0.17-0.35m` depth, completely
      stable regardless of drone motion — the camera rig/mount itself, visible
      in its own FOV, not real environment. `cam_min_depth` was `0.1`, below
      that artifact's range, so it passed straight through and got fused as a
      permanent phantom wall directly ahead of the drone on every frame. Raised
      to `0.45` (comfortably above the observed ceiling, similar margin to
      `astar_planner`'s own `inflate_radius_m=0.4m`). This is very likely the
      real cause of "boxed in, no A* route" seen the same day, not primarily
      the turning-freeze timing issue first suspected.
- [x] Added `rooster_demo_mode_manager.py` (new, ROS1, `falcon_adapter`
      package): a minimal demo-mode arbiter for Rooster, since only XTEND had
      one (`xtend_drone_demo_manager.py`). Without it `/R1/demo_mode` never
      actually reported `"turning"`, so `mapping_sync`'s authoritative rotation
      freeze never engaged at all. Deliberately not a copy of the XTEND
      version (that one also auto-sends stop/land/disarm on FINISH, which
      would make this arbiter autonomously land the drone — landing must stay
      a deliberate pilot action for Rooster).
- [x] **New bug found once the arbiter worked**: `waypoint_follower_node.py`'s
      own rotation supervisor gets stuck permanently requesting `"turning"` —
      confirmed live, continuously, tick after tick, while its `RUNNING`
      navigation loop targets the default startup goal with no real flight
      dynamics ever confirming a turn is complete (e.g. while the drone is
      flown manually, bypassing FALCON's own `/cmd_vel`, or sitting disarmed
      on the floor). This permanently froze `mapping_sync` — confirmed the
      freeze does NOT clear on `reset_freeze()`/warm-up cycling, nor on
      manually publishing `fly_straight` to the request topic. Pragmatic fix
      for now: `freeze_on_turning_mode:=false` on `mapping_sync` (reverting to
      the freeze-less baseline everything was verified against before the
      arbiter existed). The supervisor's stuck-freeze bug itself is
      unresolved — flagged, not guess-patched.
- [x] Raised the artificial low "ceiling" (`box_max_z`/`vbox_max_z`: `1.8` →
      `4.0`, above the real ~3.39m ceiling) so RViz shows real room geometry
      instead of truncating early. This immediately crashed `exploration_node`
      live (`voxel_mapping::ESDF::getDistance` → glog FATAL "Address out of
      range") because `vbox_max_z` was left exactly equal to `map_max_z`,
      removing the margin ESDF's neighbor-cell distance/gradient queries need
      near a boundary. Fixed by raising `map_max_z` to `5.0`, restoring a 1.0m
      margin (the original config had a 2.2m gap for the same reason).
- [x] Fixed a stale saved RViz camera position in `sphera_jail.rviz` (`Focal
      Point Y: 14.66`, from before the Y-axis fix) that pointed the camera at
      empty space once Y flipped sign — RViz looked completely empty with no
      error, not just missing voxels. Updated to `Y: -14.66`.
- [x] Verified orientation conclusively with a clean, longer (15-18s) pure
      `forward` command (bypassing the planner/twist-adapter entirely):
      `dx≈-0.1 to -0.7, dy≈+3.5 to +3.8` each time — confidently and
      repeatably forward in world `+Y`, matching the spawn heading (~94°)
      almost exactly. Also confirmed the depth-filter fix's effect: occupied
      cell counts dropped substantially (889 → 315 in one comparable window)
      after raising `cam_min_depth`.
- [ ] **Open**: left/right (lateral) axis reported mirrored during one BEV
      flight test (drone next to the real left wall in Sphera; map showed it
      next to the right wall) — a DIFFERENT bug from the Y/forward-back
      handedness fix above, since forward/back and altitude are both
      confirmed correct and the drone's own tracked position is accurate.
      Most likely in how the camera's local lateral axis gets projected into
      world-frame points (not the drone's own pose/localization). Added a
      physical landmark (a person, then a wooden box) near one hallway wall
      as an asymmetric visual reference to test this conclusively. Not yet
      definitively confirmed either way as of end of session — last visual
      check was reported as "looks good" but wasn't cross-checked
      side-by-side against the landmark's known real-world side.
- [ ] Confirmed unrelated to our own code: a genuine Sphera vendor rendering
      bug — an actor (person) visible in Sphera's live scene view and its own
      "Cameras View" preview was completely absent from the actual
      saved/exported frame file at the same moment. Reproduced and documented
      with a side-by-side frame comparison; drafted (not sent) a vendor bug
      report. A later fresh capture of a different added object (a wooden
      box) did NOT reproduce this, suggesting it may be a transient/timing
      issue (e.g. only affecting the first captures right after an actor
      spawns) rather than a permanent per-actor-type exclusion — not
      conclusively isolated.

## Open questions
- Is the left/right (lateral) mirroring real, or was the "looks good" check
  at the end of the session actually conclusive? Needs a proper side-by-side
  check against the landmark before considering this closed.
- Is the `waypoint_follower` rotation-supervisor stuck-freeze bug going to
  resurface once `freeze_on_turning_mode` is re-enabled (it must be
  re-enabled eventually — the turning-smear protection it exists for is
  real, confirmed by the original ring-artifact map from 2026-07-27)?
- Is the vendor's missing-actor-in-saved-frame bug transient (spawn-timing)
  or does it still need a vendor report? One counter-example (the box
  rendering fine) isn't enough to rule it out completely.
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
- 2026-07-28: Most of today's testing deliberately bypassed FALCON's own
  planner (direct `cmd_nav` "forward"/"stop" commands to `rooster_command_unit`,
  twist-adapter killed) specifically to isolate whether bugs were in low-level
  flight control/mapping versus the planner layer. That isolation is exactly
  what surfaced the `waypoint_follower` rotation-supervisor stuck-freeze bug
  (its own navigation loop runs independently of whatever the manual test is
  doing) — a real, useful side effect of the test methodology, not a
  distraction from it.
- 2026-07-28: A real, one-time regression was introduced by today's own
  Z-bound fix (`vbox_max_z` raised to equal `map_max_z` exactly) — caught
  live via `exploration_node`'s crash, not by review. Worth remembering when
  touching `map`/`box`/`vbox` bounds again: they need a real margin between
  each other, not just the documented `map ⊇ vbox ⊇ box` ordering.

## Result
Core click-to-fly motion path confirmed working (drone visibly moved toward a
clicked goal, `cmd_vel_gate`'s passed-command counter climbed continuously instead
of freezing). Y-axis handedness and map-bounds fixes verified with quantitative
ground-truth tests. As of 2026-07-28: the camera-rig phantom-obstacle bug (likely
the real cause of the original noisy/speckled map) is fixed and verified reduced;
RViz's blank-screen bug (stale saved camera position) is fixed; the ceiling-height
and its follow-on `exploration_node` crash are both fixed. Two things remain open:
a possible left/right (lateral) mirroring bug, not yet conclusively confirmed
either way, and `waypoint_follower`'s rotation-supervisor stuck-freeze bug, worked
around (freeze disabled) rather than fixed. See `CHANGELOG.md` and `LESSONS.md`
for the individual fixes and debugging detail.
