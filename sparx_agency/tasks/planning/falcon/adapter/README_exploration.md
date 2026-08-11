# FALCON exploration mode (`nav_mode:=exploration`)

Flies FALCON's own autonomous frontier-exploration planner instead of following a
BEV-clicked goal — no click needed, it picks where to go on its own.

**Requires the drone to already be airborne and hovering before you start it** — it plans
viewpoints in free 3D space and cannot path to them from the ground. If you skip this it
looks alive (busy computing) but never moves.

## Programs to run, in order

**1. Sphera** — the simulator app itself. Started by hand, not from this repo.

**2. `it`** (vendor ROBOTICAN/Sphera backend — FCU driver, `rooster_manager`, video handler):
```bash
cd ~/rqs_iai_ws/src && docker compose up -d it
```

**3. `robotican_dev`** (runs Frame Capture / Depth Processor / Twist Control Adapter below —
without this up first, those three fail with `container ... is not running` even though
mission_control looks fine):
```bash
cd /home/user1/GIT/TheAgency && docker compose -f docker-compose.robotican.yml up -d
```
If `/tmp` was freshly cleared (e.g. a reboot today), also run this once — otherwise Frame
Capture crashes on its first frame with `PermissionError` (see `LESSONS.md`):
```bash
sudo chmod 777 /tmp/rooster_frames /tmp/rooster_depth
```

**4. mission_control** (the dashboard everything else runs from):
```bash
cd /home/user1/GIT/TheAgency && venv/bin/streamlit run sparx_agency/tools/mission_control.py
```
Opens at `http://localhost:8501`.

**5. In mission_control, start these services, in this exact order** (wait for each to show
running before starting the next):

1. Rooster Ground Truth Localization (R1)
2. Rooster Video Trigger (R1)
3. Rooster Command Unit (R1)
4. Rooster Frame Capture (R1) — needs `robotican_dev` (step 3) already up
5. Rooster Depth Processor (R1) — needs `robotican_dev` (step 3) already up
6. Rooster Twist Control Adapter (R1) — needs `robotican_dev` (step 3) already up
7. Rooster Falcon Container
8. Rooster ROS1<->ROS2 Bridge

**6. Arm and take off.** Start **Rooster Manual UI (R1)** from mission_control and use it
(or however you normally fly manually) to arm + take off. Confirm it's hovering before
the next step.

**7. Start exploration.** Start **Rooster Falcon Adapter** from mission_control — it's
already configured with `nav_mode:=exploration`. This launches `exploration_node` →
`traj_server` → `falcon_exploration_follower_node.py`, which takes over `/cmd_vel_raw` and
flies the plan.

To fly a normal click-to-fly mission instead, edit `nav_mode:=exploration` back to
`nav_mode:=astar` in `tools/mission_control.py`'s "Rooster Falcon Adapter" service — it's a
launch-time switch, not something you can flip live.

## Health checks (copy-paste)

```bash
# Is it actually driving the drone?
docker exec falcon bash -lc "source /opt/ros/noetic/setup.bash && rostopic hz /cmd_vel_raw"

# Follower's own status — the fastest single signal
docker exec falcon tail -n 5 /root/.ros/log/latest/falcon_exploration_follower*.log
#   demo=exploring                    -> mode handoff granted, it owns cmd_vel
#   ref_ready=True                    -> traj_server is publishing a READY trajectory
#   holding=True                      -> no valid/fresh trajectory right now (safe
#                                         default, not a crash) - check exploration_node's
#                                         log next; most likely cause: not airborne yet
#   pos_err=<changing> holding=False  -> actively tracking - this is "it's working"

# Is the map actually building? depth=0 stuck at gate=WARMUP means no video reaching
# the pipeline (not a planner bug) - see the fly-rooster-sphera skill's Gotchas
docker exec falcon tail -n 3 /root/.ros/log/latest/mapping_sync-3.log
```

## Where the logs are

| What | Command |
|---|---|
| Everything (all nodes' console output, incl. `exploration_node`'s FSM/HGrid/TSP lines) | `tail -f /tmp/rooster_planner_adapter.log` |
| `falcon_exploration_follower`'s heartbeat | `docker exec falcon tail -f /root/.ros/log/latest/falcon_exploration_follower*.log` |
| Map fusion health (pose/depth flowing?) | `docker exec falcon tail -f /root/.ros/log/latest/mapping_sync-3.log` |

## What's actually running

```
exploration_node (always on, every mode) -> /planning/bspline
  -> traj_server -> /planning/pos_cmd (50Hz)
  -> falcon_exploration_follower_node.py -> /cmd_vel_raw
  -> ros1_bridge -> rooster_twist_control_adapter.py -> rooster_command_unit.py -> FCU
```
`waypoint_follower_node.py` and the A*/NavDP/combination planner nodes are all disabled
in this mode — there's no discrete path, only a continuous trajectory.

For a normal click-to-fly mission (or object approach) instead, see `README_click_to_fly.md`.

See `LESSONS.md` (2026-08-10 entries) and the `fly-rooster-sphera` skill for the
debugging history behind these.
