# 010 - nav_debug: make the control pipeline transparent

**Branch:** `claude/debug-viz-telemetry-refactor-af3794` (worktree
`.claude/worktrees/shpera-map-calibration-f54b4d`)
**Status:** in progress
**Roadmap item:** "Make the FALCON->Rooster control chain readable end to end in the
nav_debug replay: the velocity-loop input, the reference vs the aircraft's actual state,
and the joystick counts actually sent" (new, added to ROADMAP.md)
**Driving run:** `~/.cache/sparx_agency/falcon_nav_logs/sphera/nav_debug_20260906_235809`

## Goal
Make one frame of the nav_debug replay answer, without leaving the screen: what velocity
the velocity-closing block was handed, where the aircraft actually was and how fast it was
actually going against the setpoint it was chasing, what `OURS (cmd_vel)` even is, and what
the drone was really told on each stick axis.

## Why
Replaying `nav_debug_20260906_235809` showed four gaps, three of them in data that was
already being recorded and simply never read:

- The **velocity-loop input vector** was drawn only as gauges plus two number lines, and
  the tracker's own `command` / `command_requested` records — present on 5750 of 5800
  control rows — reached nothing. Neither did the `gate` block.
- **Nothing on screen was a measurement.** The drone's position and yaw appeared only as
  map geometry; its velocity appeared nowhere at all on a run without the ROS2 half. Worse,
  `telemetry.jsonl`'s `vx`/`vy`/`vz`/`wz` are the *command* (latched off `/cmd_vel` at each
  pose tick), and `_build_series` substituted them for achieved speed when ground truth was
  missing — so the "speed" strip plotted the command against itself and read as perfect
  tracking on exactly the runs where nothing had been measured.
- **`OURS (cmd_vel)` was a bare title.** "Ours" is ambiguous the moment more than one node
  can publish a velocity, and this one is specifically the *post-GO-gate* topic, one hop
  downstream of what the follower published.
- **`TO DRONE (cmd_nav)` was structurally unreachable-as-populated on Sphera.** It read
  `frame.drone_cmd`, which only the XTEND certainty CSV fills; Sphera writes no CSV. And
  the branch that drew it was taken only when the ROS2 half was *absent* — i.e. exactly
  when there were no counts to draw. With the ROS2 half present the whole block was
  replaced by the servo-internals section.

## Steps
- [x] Establish the real chain by reading the launch files rather than the topic names:
      the follower publishes `cmd_vel_raw`, `cmd_vel_gate_node` is the only publisher of
      `<drone_ns>/cmd_vel`, and the recorder's `cmd_topic` is the post-gate one.
      (`nav_stack.launch:1695-1715, 1843-1847`)
- [x] `frame.VelocityTarget` + `frame.DroneState`; `Truth` gains the recorded-but-dropped
      `yaw` / `yaw_rate`.
- [x] Read `command` / `command_requested` into the frame (`records.velocity_target`,
      `records.velocity_requested`, `session._lanes_at`).
- [x] New `state_source.py`: the measured state, from the follower's odometry, else Sphera
      ground truth, else a centred difference of the pose spine. Named on screen.
- [x] The follower records its own `state` section in the control trace — additive, after
      the command is sent, inside the existing swallow-everything guard.
- [x] Panel: `OURS (cmd_vel)` keeps its gauges and gains an explicit `lin`/`ang` vector,
      its frame, its producer and its consumer, plus `SHAPER CLIPPED` / `GO GATE BLOCKED`
      when an upstream stage changed the command.
- [x] Lane: `REFERENCE vs ACTUAL` — a target/actual/error table over position, velocity
      and heading, world frame, with the heading error wrapped.
- [x] `drone_cmd` filled from the actuator lane's `ManualControl` when there is no CSV; the
      TO DRONE block always drawn, as a per-axis count table; the servo internals moved to
      their own `AXES` section instead of replacing it.
- [x] Absent states made honest: missing counts name `run_nav_debug_recorder.sh`;
      `NavSession` warns on a missing *or* non-overlapping `ros2/` half; the player prints
      `join_report()` (it was dead code).
- [x] `resolve_scales` counts FALCON's exploration lanes as Rooster evidence — the study
      run was being drawn on the XTEND envelope, mis-scaling every gauge by 3.5x.
- [x] `why` narrates commanded-but-not-moving, which no commanded-vs-commanded comparison
      could ever detect.
- [x] Also read `axis_trace`'s recorded `twist` -- the adapter's own copy of the twist it
      acted on -- so the command can be compared across the ROS1->ROS2 bridge
      (`BRIDGE MISMATCH`).
- [x] Draw the counts as a **transmitter**: two Mode-2 sticks
      (`hud.gauges.draw_stick`), replacing three needle gauges that never showed the
      throttle axis at all. Sign conventions pinned by test and mutation-checked.
- [x] Split `render_panel.py` (464 lines) into the command chain and a new
      `render_counts.py`; the shared three-gauge stack moved to `render_widgets.gauge_set`.
      All render modules are now under 300 lines.
- [x] Tests: 57 in `nav_debug/tests/` (was 26), including a `state_source` unit suite, plus
      2 in the recorder's own suite pinning the writer/reader key contract.
- [x] Mutation-tested the new guards: seven deliberate breaks, six killed on the first
      pass. The survivor -- deleting the heading-error wrap -- now has its own test.
- [ ] Fly a run with **both** recorders started, and confirm the TO DRONE axis table and
      the `odom` state source against live data. Everything below the ROS1 boundary is so
      far verified only against a synthetic run and a ROS2 recording from another flight.
- [ ] Collect the `ros2/` directory into the run folder automatically, so `--ros2` and the
      "did you start the second recorder" failure mode both go away.

## Notes
- **The measured velocity is derived on every existing run.** No recording on disk has a
  measured velocity in its ROS1 half, so `pose_diff` is what the reference/actual table
  will use for all of them (6415 of 6420 frames on the study run; the other 5 are the run's
  ends, which report position with an unknown velocity rather than zero). It is honest but
  noisier than either recorded source, and the lane says which one it used.
- **First real number out of it:** across the study run the aircraft achieves a mean of
  0.64x its commanded ground speed, and 15.5% of frames are commanded >0.15 m/s while
  measuring <0.05 m/s. That is consistent with the open contact-hold/wall-trap defect and
  was not previously observable from a recording.
- The `--ros2` example directory available on this machine is from a flight 2.35 days after
  the study run, so it cannot be joined to it. That is now *reported* rather than silently
  producing a blank panel.
- Running the whole `sparx_agency/tasks/planning/` tree under pytest crashes during
  collection (a `cv2` stub in `object_approach_webcam` plus a heap corruption). This
  predates the change — HEAD shows the same, 45 errors — and is worth its own entry. The
  `falcon` suite also segfaults at interpreter *teardown* after reporting 283 passed; that
  too reproduces at HEAD.
- The `AXES` section and the count table below `TO DRONE` are still only exercised against
  a synthetic run and a ROS2 recording from a different flight. That is the gap the
  remaining unchecked step closes.
