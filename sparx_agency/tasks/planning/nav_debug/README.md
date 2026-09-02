# nav_debug — per-frame visual debugging for the FALCON nav stack

The certainty CSV and the campaign metrics tell you *what* happened. They do not
tell you **why**. This package records a flight and replays it as one screen that
answers, for any instant: what the planner asked for, what the controller decided,
what the drone was actually told, what it actually did, and what the map looked
like while it decided — together, and scrubbable.

It serves two different stacks:

| | XTEND (A\* / click-to-fly) | **Sphera (FALCON exploration)** |
|---|---|---|
| plan | `/path/waypoints_astar` → `_safe` → `/path/waypoints` | `/planning/bspline` → `/falcon/planned_path` |
| target | pure-pursuit `/path/lookahead` | **`/planning/pos_cmd`** — a 100 Hz reference |
| controller | `DriftPidFollower` | `ReferenceTracker3D` |
| actuation | `cmd_nav` axis counts | expo curve + PI servo → `/R1/manual_control` |
| spine | certainty CSV | `telemetry.jsonl` (no CSV is written) |

## Why this was rebuilt (read this first)

The recorder had been running on **every** Sphera flight since it was written —
`nav_debug_record` defaults true — and had produced 1111 run folders that were
almost entirely empty:

```
events.jsonl        0 bytes in 1056 of 1111 runs
routes/             1 file, {"astar": null, "safe": null, "final": null}, in ~95%
certainty CSV       never written on this path at all
```

Everything it subscribed to belongs to the XTEND A\* chain, and on Sphera the
stack runs `nav_mode:=exploration`, where those nodes are **gated off**. Pose,
`/cmd_vel` and the BEV map recorded fine; the route, the target and every
"why" signal did not. This package now records FALCON's own chain alongside the
XTEND one, each lane behind its own rosparam, so one recorder serves both stacks
and an XTEND run is byte-for-byte unchanged.

## Two recorders, because one cannot see the whole loop

`bridge.yaml` carries `/cmd_vel` and the frame paths across the ROS1↔ROS2 bridge
and **nothing else** — not `/R1/manual_control`, not `/R1/velocity_truth`, not
`/R1/sphera/state`. So the ROS1 recorder is structurally blind to what the drone
was told and what it actually did, and the ROS2 recorder is blind to the plan.

```
falcon container (Noetic)                 it container (Foxy)
  nav_debug_recorder_node.py                nav_debug_ros2_recorder.py
  plan · reference · tracker · map          cmd_nav · ManualControl · truth
            │                                          │
            └──────── joined offline on the HOST WALL CLOCK ────────┘
```

Every row from either side carries **both** its own ROS `t` and the host
`wall` clock (`schema.row`). The two containers share the host kernel clock, so
`wall` is the join key and needs no offset negotiation — and because the ROS
epochs differ wildly between containers, joining on `t` would silently blank
every cross-recorder panel. `NavSession.join_report()` prints the estimated
offset and its spread so a bad join is reportable instead of invisible.

## Recording a run

**The ROS1 half is automatic.** `run_falcon_sphera.sh` now exports
`FALCON_RUN_DIR`, so the recording, the thought journal and the certainty log
share **one folder and one stamp** (previously the manifest pointed at a
certainty CSV that was never written):

```
~/.cache/sparx_agency/falcon_nav_logs/sphera/nav_debug_<stamp>/    # + `current_run` symlink
```

**The ROS2 half is one extra command**, because `it` is started separately and
has no mount in common with the FALCON log directory:

```bash
sparx_agency/robots/ROBOTICAN/run_nav_debug_recorder.sh
```

It is subscribe-only: it publishes nothing, commands nothing, and starts no
flight node. It writes under `it`'s workspace bind (host-visible immediately),
so nothing needs `docker cp`. To turn the whole thing off for a clean run:

```bash
roslaunch falcon_adapter sphera_drone.launch nav_debug_record:=false
```

`nav_debug_trace:=false` keeps the recording but drops the three diagnostic
publishers described below.

## Replaying it

```bash
python -m sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug \
    --run ~/.cache/sparx_agency/falcon_nav_logs/sphera/current_run \
    --ros2 ~/rqs_iai_ws/nav_debug_logs/<stamp>/ros2
```

`--ros2` is only needed while the two halves live in separate trees; drop it once
the ROS2 directory has been collected into the run folder. Gauge full-scales are
picked automatically (`--scales auto`) — Rooster's envelope is 3–4× XTEND's, so
forcing the wrong one mis-scales every command gauge.

Keys: **n/→** next · **p/←** prev · **SPACE** play/pause · **z/x** zoom ·
**+/−** speed · **s** save PNG · **q** quit. Headless? `--export out/` writes
frames plus an mp4.

## What the screen tells you

The map pane draws the BEV occupancy, the planned and **executed** paths, the
pose trail, and the reference being chased with the gap to it drawn as a line —
so position error is visible as geometry, not just a number. The lane column:

- **REFERENCE** — the `pos_cmd` this instant, with `MOVING` / `FROZEN ENDPOINT` /
  `STALE` badges. `traj_server` republishes a frozen endpoint with *fresh
  stamps* at a trajectory's end, so "fresh" must never be read as "moving".
- **TRACKING** — `position_error_m` split into **lag** (benign: late) and
  **cross-track** (not benign: this is the one that flies into walls), plus
  `diverged` / `holding`. `ReferenceTracker3D` computed all four every tick and
  the flight node kept one of them, for a 0.5 Hz log line.
- **CONTROL** — the command decomposed: `ff + damp + cor → cmd → clamp → out`.
  Recording only the sum makes an over-aggressive gain indistinguishable from a
  large reference velocity; this shows which term, or which *limit*, chose the tick.
- **TO DRONE** — per axis: requested vs achieved speed, and
  `ff + cor → pre-slew → sent` counts, flagged `saturated` / `slew_limited` /
  `capped` / `feedback_stale`, then the `cmd_nav` request beside the
  `ManualControl` actually published. A `MANUAL != CMD_NAV` flag catches the
  altitude loop or a second publisher writing the stick underneath you.
- **ALTITUDE** — `target` vs `ranger`, `wanted_z` vs `sent_z`, `AT CEILING`, and
  a red `GUARD ×n`. The rangefinder plausibility gate used to reject a sample and
  return **silently**, so a gate firing every tick looked exactly like a healthy
  hold — the blind spot behind the open "14% of flights climb past 2 m" defect.
- **TRUTH** — achieved speed vs commanded, roll/pitch, battery, arm/mode, from
  Sphera's own state. PX4's estimate is not authoritative here: Sphera's physics
  runs off the vendor ManualControl pipeline.
- **MAP** — pose age and the occupied/free/unknown census behind the grid the
  planner actually used.

Yaw on Rooster is **open loop** — it has no rate feedback in the adapter — so its
`measured`/`error`/`correction` read zero by construction, not by success.

## Absent is never drawn as zero

A debug screen that invents a plausible number is worse than one that admits it
does not know, so every unrecorded value renders as `-`, not `0`:

- `"tracking": null` (the follower returned early — muted demo mode, no
  reference, a tilt cut) blanks the TRACKING lane instead of showing a
  zero-error "perfect tracking" panel.
- An altitude tick the hold loop skipped names its reason (`not_holding`,
  `no_ranger`, `no_new_sample`, `guard_rejected`) and blanks
  `err`/`wanted_z`/`sent_z`, which the loop never computed.
- The MAP counters with no recorder yet — `depth_frames`, `emitted`,
  `drop_reason`, `gate_state`, `outside_bbox_frac`, `depth_age_s`, `tilt_deg` —
  read `-`. Wiring them to `mapping_sync`'s per-second heartbeat (which already
  counts seven distinct drop reasons and the gate state) is the obvious next step.
- Rooster's yaw axis is **open loop** — there is no rate feedback in the adapter —
  so its `measured`/`error`/`correction` are zero by construction, and the lane
  labels it rather than letting it read as perfect tracking.

## Recording cost

FALCON's executed path is a marker that grows for the whole flight and is
republished whole, so it is decimated to `~executed_max_points` (600) at ingest
and route snapshots are rate-limited to `~routes_min_interval_s` (0.5 s);
without both, a snapshot cost O(flight length) and the folder grew as its square.
The control lane is held to `~record_hz` rather than the follower's publish rate.

## Layout of the code

| file | role |
|------|------|
| `schema.py` | the on-disk contract + `row()`; pure stdlib, imported by both recorders |
| `frame.py` | `NavFrame` and the dataclasses one moment is made of |
| `session.py` | assemble a run into a lazy frame timeline |
| `timeline.py` | the spine (certainty CSV, else `telemetry.jsonl`) |
| `sources.py` | jsonl lanes, as-of joins, the cross-recorder clock estimate |
| `records.py` | recorded row → `frame` dataclass |
| `bev_source.py` / `route_source.py` / `event_source.py` | the map, route and event lanes |
| `bev_image.py` | occupancy grid → BGR + the world→pixel mapping |
| `render.py` | compose the screen (`render_map`, `render_panel`, `render_lanes`, `render_widgets`) |
| `why.py` | the one-line narration under the screen |
| `run_folder_nav_debug.py` | the offline player / exporter CLI |
| `../hud/` | shared gauges + panel primitives (also used by object-approach) |

Writers, outside this package:

| file | role |
|------|------|
| `../falcon/adapter/scripts/nav_debug_recorder_node.py` | the ROS1 recorder (+ `nav_debug_sources`, `nav_debug_run_folder`) |
| `../../../robots/ROBOTICAN/nav_debug_ros2_recorder.py` | the ROS2 recorder |
| `../falcon/adapter/scripts/falcon_exploration_follower_node.py` | publishes the control trace |
| `../../../robots/ROBOTICAN/adapters/rooster_twist_control_adapter.py` | publishes the axis trace |
| `../../../robots/ROBOTICAN/helpers/rooster_unit.py` | publishes the altitude trace |

The three trace publishers are additive by construction: they run after the
command is sent, change no gain and no timing, swallow their own failures, and
are removed entirely by `nav_debug_trace:=false`.

The renderer is ROS-free and Python 3.8 compatible on purpose, so the same
`render()` can drive a live viewer node inside the Noetic container. Note that
no single container sees the whole loop live — a ROS1-side live view can show the
plan, the reference and the tracker, but not the actuator or ground-truth lanes.
