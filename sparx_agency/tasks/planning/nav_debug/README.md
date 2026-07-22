# nav_debug — offline (and, next, live) visual debugging for the FALCON nav stack

The certainty CSV alone doesn't tell you *why* the drone did what it did. This
package records a flight and replays it as one screen that answers, for any
instant: **what A\* planned, what the drone wanted to do, and why** — the map,
the routes, the two command channels, the drift, the localization and the replan
reasons, together and scrubbable.

It reuses the exact ROLL/PITCH/YAW gauge vocabulary of the object-approach
target-lock HUD (now the shared primitives in `tasks/planning/hud/`).

## The debug screen (layout: map + telemetry panel)

```
┌─ NAV DEBUG ───────────────────────┬─ TELEMETRY ──────┐
│ MODE / REPLAN reason / LOC state  │ OURS (cmd_vel)   │
│   BEV map (world, +y up):         │  ROLL PITCH YAW  │
│    · raw A*   ═ corrected  ━ FINAL│  + vz, numbers   │
│    ◍ target wp   ➤ pose + trail   ├──────────────────┤
│    ↳ drift   ✱ goal   • lookahead │ TO DRONE(cmd_nav)│
│   scale bar · WHY strip           │  ROLL PITCH YAW  │
│                                   │  conf · why · ▁▂▄ │
└───────────────────────────────────┴──────────────────┘
```

- **Two gauge stacks.** *OURS* is `cmd_vel` (`vx/vy/vz/wz`, m/s & rad/s, green);
  *TO DRONE* is the `cmd_nav` axis counts the converter sends (`fwd/lat/vert/yaw`,
  ±1000, cyan). The converter inverts the lateral and yaw signs, so the drone
  counts are negated before their gauges — **the same physical motion reads the
  same way in both stacks**.
- **Three route layers** so you can see the correction: raw A\*
  (`/path/waypoints_astar`), BEV-corrected (`/path/waypoints_safe`), and the final
  route the drone flies (`/path/waypoints`).
- **Replan banner** names the reason from `/path/astar_event`:
  `time` (periodic), `rotation`, `obstacle` / `blockage` / `boxed_in`.
- **Localization** current pose + a fading trail, tinted by confidence; `coast`
  when dead-reckoning. **Drift** is the orange vector the controller is fighting.

## Record a run (on the drone)

**On by default.** `nav_debug_record` and `certainty_log` default to `true` on
`nav_stack.launch`, `real_drone.launch` and `object_mission.launch`, so a plain
run already records. Nothing flight-critical changes — the recorder is a
subscribe-only sink. To turn it off for a clean run:

```
roslaunch falcon_adapter real_drone.launch nav_debug_record:=false certainty_log:=false
```

`run_falcon.sh` makes **one folder per run**, under `$FALCON_LOG_DIR` (the host
bind-mount `/tmp/falcon` by default, so it survives the `--rm` container). The
thought journal, the certainty CSV **and** the recording all land inside it,
sharing **one timestamp** (`FALCON_RUN_DIR` points every writer at the same dir):

```
nav_debug_YYYYmmdd_HHMMSS/                ← one folder per run, one shared stamp
  certainty_YYYYmmdd_HHMMSS.csv    per-tick pose, both commands, drift, quality
  thoughts_YYYYmmdd_HHMMSS.log     the thought journal (why, in prose)
  manifest.json                    run metadata
  telemetry.jsonl                  pose + OUR command, per tick
  bev/<ms>.npy(+.json)             occupancy grid + geometry (saved on change)
  bev_conf/<ms>.npy                per-cell confidence, co-registered with bev/<ms>
  routes/<ms>.json                 raw / corrected / final routes + goal + aim point
  events.jsonl                     replan reasons + blockages
```

The `<ms>` in the map/route filenames is each snapshot's own ROS-time
millisecond — those *are* meant to differ (one file per moment). Keep
`certainty_log:=true`: the CSV carries the per-tick pose, both command sets, drift
and quality; the recorder adds the map, routes and events; the offline tool joins
them by timestamp (as-of, matching the drone's latched view).

## Replay it (on the dev PC)

```
python -m sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug \
    --run $FALCON_LOG_DIR/nav_debug_YYYYmmdd_HHMMSS
```

The window is resizable — drag any corner to enlarge (a quick zoom); **z/x**
re-render the map larger/smaller for a crisp one. Keys: **n/→** next · **p/←**
prev · **SPACE** play/pause · **z/x** zoom · **+/−** speed · **s** save PNG ·
**q** quit. Headless (no display)? Export instead:

```
python -m ...run_folder_nav_debug --run <folder> --export out/   # writes frames/ + replay.mp4
```

`--csv` overrides the certainty CSV; without a CSV the run still replays from
`telemetry.jsonl` (map + routes + OUR command), just without the drift/quality
panel fields.

## Layout of the code

| file | role |
|------|------|
| `frame.py` | `NavFrame` + the small dataclasses one moment is made of |
| `bev_image.py` | occupancy grid → BGR + the world→pixel mapping |
| `session.py` | load a run folder + CSV → lazy `NavFrame` timeline (as-of joins) |
| `render.py` | draw the debug screen (map pane + telemetry panel) |
| `run_folder_nav_debug.py` | the offline player / exporter CLI |
| `../hud/` | shared gauges + panel primitives (also used by object-approach) |
| `../falcon/adapter/scripts/nav_debug_recorder_node.py` | the ROS1 recorder |

The renderer is ROS-free and Python 3.8-compatible on purpose: the same
`render()` will drive a live viewer node inside the Noetic container (next step),
exactly as `target_lock_viewer_node` shows the object-approach HUD.
