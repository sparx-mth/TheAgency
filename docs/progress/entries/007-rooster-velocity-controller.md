# 007 - Rooster velocity controller calibration

**Branch:** `create_devcontainer_daphna` (current)
**Status:** in progress
**Roadmap item:** "Calibrate/tune the FALCON->Rooster velocity controller against real drone
behavior, using logged manual-flight data" (new, add to ROADMAP.md)

## Goal
Get `rooster_twist_control_adapter.py` (the Twist -> `/R1/cmd_nav` move-axes translator that
sits between FALCON's `waypoint_follower_node.py` and the drone) calibrated against real
Rooster/Sphera behavior instead of never-validated assumed max speeds, so click-to-fly
navigation drives at the speed/turn-rate FALCON's planner actually intends.

## Why
`rooster_twist_control_adapter.py` scales an incoming Twist to an axis value as
`axis = twist_component / max_component * 1000`, where `max_linear_x`/`max_linear_y`/
`max_yaw_rate` are supposed to be "real-world rate at full axis deflection." These were
guessed defaults (0.25 m/s / 0.25 m/s / 0.5 rad/s), never live-validated. A mismatch here
means FALCON's planner (which reasons in real m/s and rad/s) actually commands a different
real-world speed than it thinks it does.

## Steps
- [x] Build a read-only command+pose logger (`manual_flight_logger.py`, subscribes to
      `/R1/cmd_nav` and `/R1/localization`, never publishes) to capture a command->motion
      dataset without disturbing manual flight.
- [x] Log a full manual flight (arm -> takeoff -> forward/left/right/turn_left/turn_right
      mix -> land -> disarm) via the Tkinter `ui.py`, confirmed by the user to look visually
      correct (left/right and turn_left/turn_right all matched expected direction in Sphera).
- [x] Segment the log by command boundaries, compute body-frame displacement (forward/right)
      and yaw rate per segment.
- [x] Found: turn_right/turn_left rates are consistent and robust (8 turn_right segments,
      axis r=500 -> ~55 deg/s ~0.96 rad/s; 3 turn_left segments -> ~42 deg/s ~0.74 rad/s),
      extrapolating to axis 1000 gives ~1.9 rad/s (right) / ~1.5 rad/s (left) - the adapter's
      assumed 0.5 rad/s max was ~4x too low.
- [x] Found: forward/lateral segments are too short and interleaved with adjacent turns in
      this flight (leftover momentum contaminates each segment - several "forward" segments
      even show net negative displacement) to extract a trustworthy linear-speed number. The
      one clean isolated sample (first forward segment, axis 600, right after takeoff) gives
      ~0.36 m/s but is a single point, not a calibration.
- [x] Recalibrated `max_yaw_rate` default 0.5 -> 1.8 rad/s (rounded, between the left/right
      estimates) in `rooster_twist_control_adapter.py`, documented derivation inline and in
      `LESSONS.md`. Left `max_linear_x`/`max_linear_y` unchanged (within noise of the current
      assumption, not confidently re-derivable from this flight).
- [ ] Dedicated calibration flight: isolated single-axis moves (e.g. 5s pure forward, stop,
      5s pure right, stop, 5s pure turn_right, stop, repeat at 2-3 axis values each) with no
      interleaving, to get a trustworthy forward/lateral gain and cross-validate the yaw
      asymmetry seen here (real, or just a 3-sample noise artifact).
- [ ] Re-test click-to-fly (BEV click goal) with the corrected yaw-rate gain and confirm
      turning behavior looks less aggressive/erratic than before.

## Open questions
- Is the ~25% left/right yaw-rate asymmetry (1.9 vs 1.5 rad/s) real drone/FCU behavior, or
  noise from only having 3 turn_left samples? The dedicated calibration flight should settle
  this before picking anything more precise than a single symmetric `max_yaw_rate`.
- Should `max_linear_x`/`max_linear_y` get their own calibration flight pass, or are they
  close enough as-is? Current single clean sample (~0.36 m/s at axis 600, i.e. ~0.6 m/s
  extrapolated to full scale) suggests they might also be underestimated, but it's one noisy
  point, not evidence at the same confidence level as the yaw finding.
- Original, still-unresolved thread from earlier BEV-click testing: a "clicked point to my
  right, drone turned left" report was investigated and found FALCON's own internal
  commanded-wz-vs-goal-bearing math to be self-consistent and correct at the time - but that
  investigation used the *old* 0.5 rad/s yaw-rate assumption to reason about behavior. Worth
  revisiting once the corrected gain is live, in case the previous "self-consistent" finding
  was self-consistent with a wrong *gain* even though the *sign* was right.

## Notes
- 2026-07-30: `manual_flight_logger.py` is subscribe-only by design specifically so it could
  be dropped into an already-flying pipeline without any risk of interfering (unlike the
  duplicate-`/R1/cmd_nav`-publisher class of bug documented in `LESSONS.md`'s twist-control
  adapter entry).
- 2026-07-30: Considered adding a second, new controller node instead of editing
  `rooster_twist_control_adapter.py` in place - rejected: it already owns exactly this job
  (Twist in, cmd_nav move-axes out), and a second node with the same responsibility would
  recreate the same "who owns this topic" ambiguity that `rooster_command_unit.py`'s own
  docstring calls out as the reason it's the sole owner of `/manual_control`.

## Result
`max_yaw_rate` recalibrated from a guessed 0.5 rad/s to a data-derived 1.8 rad/s - in progress,
pending the follow-up calibration flight and a re-test of click-to-fly behavior with the new
gain.
