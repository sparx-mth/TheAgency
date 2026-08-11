# Click-to-fly + object approach (`nav_mode:=astar`)

Normal mission: BEV-click (or any goal) → A* flies there. Optionally layer object-approach
on top: while the target is unconfirmed the drone flies its normal route; once YOLO
confirms it, object-approach takes over `/cmd_vel` and flies to/lands on it.

## Programs to run, in order

**1. Sphera** — the simulator app itself. Started by hand, not from this repo.

**2. `it`** (vendor ROBOTICAN/Sphera backend):
```bash
cd ~/rqs_iai_ws/src && docker compose up -d it
```

**3. `robotican_dev`** (runs Frame Capture / Depth Processor / Twist Control Adapter):
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

**5. In mission_control, start these services, in this exact order:**

1. Rooster Ground Truth Localization (R1)
2. Rooster Video Trigger (R1)
3. Rooster Command Unit (R1)
4. Rooster Frame Capture (R1) — needs `robotican_dev` (step 3) already up
5. Rooster Depth Processor (R1) — needs `robotican_dev` (step 3) already up
6. Rooster Twist Control Adapter (R1) — needs `robotican_dev` (step 3) already up
7. Rooster Falcon Container
8. Rooster ROS1<->ROS2 Bridge
9. Rooster Falcon Adapter — check `tools/mission_control.py`'s "Rooster Falcon Adapter"
   `cmd` is `nav_mode:=astar`, not `nav_mode:=exploration` (see `README_exploration.md`
   if you need to flip it back — it's a launch-time switch, not live)

**6. Arm and take off.** Start **Rooster Manual UI (R1)** and use it to arm + take off.

**7. Give it a goal.** Start **Rooster RViz** and **Rooster BEV Click Goal**, then click a
point on the BEV view — `waypoint_follower_node.py` will fly there via A*.

## Adding object approach (optional, on top of the above)

**8. `detector_dev`** — a separate, ephemeral container (`run --rm`, not `up -d` — it does
not persist across a restart the way `robotican_dev`/`it` do, so this needs re-running each
session):
```bash
docker compose -f /home/user1/GIT/TheAgency/docker-compose.detector.yml \
  run -d --rm --name detector_dev detector tail -f /dev/null
```
mission_control's "Rooster YOLO Detector" only `docker exec`s into this container — it will
fail with `container detector_dev is not running` if you skip this step.

**9. Start Rooster YOLO Detector**, then **Rooster Object Approach**, both from
mission_control.

Once the detector confirms the target for a few consecutive frames, `waypoint_follower`
should go passive (`demo=visual_servoing` on `/R1/demo_mode`) and object-approach takes
over, flies to the target, and lands.

## Health checks (copy-paste)

```bash
# Who currently owns cmd_vel?
docker exec falcon bash -lc "source /opt/ros/noetic/setup.bash && rostopic echo -n1 /R1/demo_mode"
#   fly_straight     -> waypoint_follower is driving
#   visual_servoing  -> object_approach has taken over
#   finish           -> object_approach confirmed landing; rooster_demo_mode_manager
#                        should now run stop -> land -> disarm

# Is object_approach actually seeing detections?
docker exec falcon bash -lc "source /opt/ros/noetic/setup.bash && rostopic hz /object_approach/detections"

# Everything's console output (waypoint_follower, object_approach, etc.)
tail -f /tmp/rooster_planner_adapter.log
tail -f /tmp/rooster_planner_object_approach.log
```

## Where the logs are

| What | Command |
|---|---|
| Falcon Adapter (waypoint_follower, path planning, mapping) | `tail -f /tmp/rooster_planner_adapter.log` |
| Object Approach state machine | `tail -f /tmp/rooster_planner_object_approach.log` |
| YOLO Detector | `tail -f /tmp/rooster_planner_detector.log` |

See `LESSONS.md` (2026-08-09/10 entries) and the `fly-rooster-sphera` skill for the
demo_mode-topic bug this fix was for, and other debugging history.
