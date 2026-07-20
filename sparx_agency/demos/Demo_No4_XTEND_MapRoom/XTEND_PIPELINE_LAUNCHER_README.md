# XTEND Pipeline Launcher UI

This document describes the XTEND pipeline launcher UI, the commands it starts, and the expected ROS 2 topic flow for drone communication, depth, localization, and planning.

The launcher is a helper/dashboard. It does **not** directly fly the drone. It starts the required ROS 2 / WebSocket / depth / localization nodes in Jetson `tmux` sessions and provides copyable commands for manual debugging.

---

## High-Level Command Flow

```text
UI / planner / replay
        ↓
/cmd_vel
        ↓
xtend_twist_to_cmd_nav.py
        ↓
/xtend/cmd_nav
        ↓
online_nav_bridge_publisher.py
        ↓
XTEND WebSocket
        ↓
Drone
```

The manual control UI can also publish direct commands such as:

```json
{"action": "arm", "value": 0}
{"action": "takeoff", "value": 0}
{"action": "land", "value": 0}
{"action": "stop", "value": 0}
```

to:

```text
/xtend/cmd_nav
```

---

## Required SSH Setup

The launcher starts Jetson commands over SSH.

Recommended SSH alias on the PC:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_xtend -C "xtend-jetson"
ssh-copy-id -i ~/.ssh/id_ed25519_xtend.pub user@192.0.0.89
```

Add this to:

```bash
~/.ssh/config
```

```sshconfig
Host xtend-jetson
    HostName 192.0.0.89
    User user
    IdentityFile ~/.ssh/id_ed25519_xtend
```

Test:

```bash
ssh xtend-jetson
```

In the launcher, set:

```text
Jetson SSH: xtend-jetson
```

---

## Jetson Environment

The Jetson commands use:

```bash
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/theagency_venv/bin/activate

export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:$PYTHONPATH
```

---

## PC Environment

The PC UI uses:

```bash
cd /home/user1/GIT/TheAgency
source /opt/ros/jazzy/setup.bash
source /home/user1/GIT/TheAgency/venv/bin/activate

export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/jazzy/lib
export PYTHONPATH=/usr/lib/python3.12/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/home/user1/GIT/TheAgency:$PYTHONPATH
```

---

## Parameters

Selecting a command shows **every parameter it accepts** on the Parameters tab —
not only the handful its command line spells out. The localization node declares
37 parameters and its command names 6; the object mission has some 290 and names 4.

They are read from whatever *declares* them, so the list can never drift from the
code:

| Command | Parameters come from |
| --- | --- |
| A ROS2 node | its `declare_parameter(...)` calls |
| An argparse script | its `add_argument(...)` calls |
| The object mission | `config/mission.yaml` + `adapter/launch/object_mission.launch` |

The comments the author wrote next to each one come too, so every knob arrives
explained, grouped under the same headings its source file uses.

**Defaults and reset.** Each parameter is shown against the value a plain start
would use — which is what the command spells out where it spells one out, and the
underlying declaration otherwise. A value you move is marked with a dot and shown
in blue; `↺` puts one back, **Reset all to defaults** puts the whole screen back.

**Only what you changed is sent.** The command is rebuilt from the parameters, and
carries the ones the command already named plus the ones you actually moved. A
mission with 290 available knobs still starts as:

```bash
NAV_MODE=hybrid ./run_object_mission.sh --falcon-only office gui dp_cruise_speed:=0.22
```

**Finding things.** The filter box searches names, documentation and sections;
**Changed only** narrows to what you have moved — useful before flying, to see
exactly how this run differs from a default one.

**Saving.** *Save these as my defaults* writes the changed parameters to
`~/.config/sparx_agency/launcher_params.json` and restores them next session.
Only changed values are stored, so an improvement to a node's own default is not
overridden by a stale copy of yesterday's. *Forget saved* drops them again.

**Run on.** Overrides where this command starts — `jetson` (SSH + tmux), `pc`
(a local terminal), or `manual` (copy to the clipboard).

The Command tab always shows what Start will run, and is editable for a one-off;
changing any parameter rewrites it.

---

## Launcher UI Buttons

### Start selected

Starts the selected item with the parameters currently on screen.

If the item is set to run on the Jetson, it starts over SSH inside a named `tmux`
session. On the PC, it opens a local terminal. Manual items are copied to the
clipboard.

---

### Start checked

Starts every ticked Jetson command in its own `tmux` session, each with its own
parameters.

Typical core sessions:

```text
xtend_bridge
xtend_depth
xtend_demo_manager
xtend_apriltag
xtend_static_tf
xtend_pose_to_tf
xtend_octomap
```

Check active sessions on Jetson:

```bash
tmux ls
```

Attach to a session:

```bash
tmux attach -t xtend_bridge
```

Detach without stopping:

```text
Ctrl+B then D
```

---

### Stop selected tmux

Stops the selected Jetson `tmux` session.

Recommended stop behavior is:

```text
send Ctrl+C -> wait -> kill session if needed
```

This is safer than directly killing the session, because Python nodes can run their cleanup logic.

For the command converter, this matters because it can publish a final stop command before shutdown.

Manual equivalent:

```bash
tmux send-keys -t xtend_twist_converter C-c
```

If needed:

```bash
tmux kill-session -t xtend_twist_converter
```

---

### Stop all known tmux

Stops all known Jetson pipeline sessions.

Manual equivalent:

```bash
tmux send-keys -t xtend_twist_converter C-c
tmux send-keys -t xtend_bridge C-c
tmux send-keys -t xtend_depth C-c
tmux send-keys -t xtend_flow_depth C-c
tmux send-keys -t xtend_velocity_integrator C-c
tmux send-keys -t xtend_static_tf C-c
```

Hard kill:

```bash
tmux kill-session -t xtend_twist_converter
tmux kill-session -t xtend_bridge
tmux kill-session -t xtend_depth
tmux kill-session -t xtend_flow_depth
tmux kill-session -t xtend_velocity_integrator
tmux kill-session -t xtend_static_tf
```

---

## Pipeline Commands

### 1. XTEND Online Bridge and RGB Publisher

Runs on Jetson.

This node:

- connects to the XTEND WebSocket
- sends drone commands
- receives drone telemetry
- publishes RGB frames
- publishes bearing/local telemetry
- subscribes to `/xtend/cmd_nav`

Command:

```bash
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/online_nav_bridge_publisher.py \
  --camera-info-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_crop_504_280.yaml
```

Expected important topics:

```text
/xtend/rgb
/xtend/camera_info
/xtend/bearing
/xtend/local_telemetry
/xtend/cmd_nav
```

Check:

```bash
ros2 topic list | grep xtend
ros2 topic info /xtend/rgb -v
ros2 topic echo /xtend/bearing --qos-reliability best_effort
```

---

### 2. DA3 Small Depth Processor

Runs on Jetson.

This node:

- subscribes to `/xtend/rgb`
- subscribes to `/xtend/camera_info`
- publishes `/xtend/depth_m`

Command:

```bash
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p camera_info_topic:=/xtend/camera_info \
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-392x504.engine
```

Important timestamp rule:

```text
/xtend/depth_m.header.stamp should be copied from the RGB input frame.
```

This is correct because the depth image is derived from that RGB frame.

Check:

```bash
ros2 topic info /xtend/depth_m -v
ros2 topic hz /xtend/depth_m
```

There should be exactly one publisher:

```text
Publisher count: 1
```

---

### 3. Twist to XTEND Command Converter

Runs on Jetson.

This node:

- subscribes to `/cmd_vel`
- converts `geometry_msgs/Twist` into XTEND JSON commands
- publishes to `/xtend/cmd_nav`

Command:

```bash
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/adapters/xtend_twist_to_cmd_nav.py \
  --cmd-vel-topic /cmd_vel \
  --cmd-nav-topic /xtend/cmd_nav \
  --timeout-sec 1.0 \
  --allow-multi-axes
```

`--allow-multi-axes` honours `linear.z` (up/down) and `linear.y` (left/right) as
well as `linear.x` and `angular.z`. Pass it if you run **this** script and want the
lost-localization recovery's climb rungs to work — its ladder climbs to see over
whatever is hiding the AprilTag. Without the flag `linear.z` falls through to
`{"action": "stop"}` and the climb rungs silently do nothing: no error, the drone
just sits there. If you cannot pass it, set `recovery_climb_enabled:=false` so the
ladder skips those rungs rather than wasting time on a no-op.

> **Which converter am I actually running?** There are two, and only one of them
> needs the flag:
> * **this standalone script** (`xtend_twist_to_cmd_nav.py`, the recipe above) —
>   vertical/lateral **OFF** unless you pass `--allow-multi-axes`;
> * **the in-process converter** inside `online_nav_bridge*` (which subscribes
>   `/cmd_vel` itself and converts via `TwistToCmdNavConverter`) — vertical/lateral
>   already **ON**, nothing to pass.
>
> Check with `ros2 node list`. Running **both** at once converts `/cmd_vel` twice
> and races two command streams into `cmd_queue` — pick one.

One action at a time: the converter picks a single axis per Twist, in priority
order `angular.z` → `linear.x` → `linear.z` → `linear.y`, and the XTEND bridge
zeroes every other axis when it applies one. So a climb cannot be combined with a
turn or a forward — which is why the recovery's rungs each drive exactly one axis.

Calibration:

```text
linear.x = 0.3 m/s  -> forward thrust 400
forward max thrust  -> 600
turn thrust default -> 1000
```

The vertical axis reuses the **translation** calibration above (`|v| / 0.3 * 400`);
there is no vertical-specific constant, and there is no altitude feedback anywhere
in the stack, so a commanded climb is open-loop thrust-for-a-duration.

Expected output:

```json
{"action": "forward", "value": 400}
```

Check:

```bash
ros2 topic info /cmd_vel -v
ros2 topic info /xtend/cmd_nav -v
```

Expected:

```text
/cmd_vel subscription:
  xtend_twist_to_cmd_nav

/xtend/cmd_nav publisher:
  xtend_twist_to_cmd_nav

/xtend/cmd_nav subscriber:
  xtend_online_bridge
```

Manual test:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Expected converter log:

```text
Published: {"action": "forward", "value": 400}
```

Stop:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

### 4. Optional Twist Replayer

Runs on Jetson.

This node replays a recorded `.jsonl` command file onto `/cmd_vel`.

Command:

```bash
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/planning/twist_replayer.py \
  --ros-args \
  -p log_path:=/home/user/GIT/TheAgency/cmd_log.jsonl \
  -p topic:=/cmd_vel \
  -p speed:=1.0 \
  -p loop:=false
```

Use `--timeout-sec 1.0` on the converter when replaying, because recorded logs may have command gaps.

---

### 5. Optical-Flow Depth Velocity Node

Runs on Jetson.

This node estimates motion from RGB + depth.

Command:

```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node_separted \
  --ros-args \
  -p use_sim_time:=false \
  -p show_debug:=false \
  -p csv_filename:="/home/user/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv" \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p depth_scale:=0.8
```

Check subscriptions:

```bash
ros2 topic info /xtend/rgb -v
ros2 topic info /xtend/depth_m -v
```

---

### 6. Velocity Integrator

Runs on Jetson.

Command:

```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.velocity_integrator \
  --ros-args \
  -p use_sim_time:=false \
  -p target_frame:=odom \
  -p init_from_gt:=false
```

---

### 7. Static Transform

Runs on Jetson.

Command:

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom xtend_camera
```

---

### 8. PC Manual UI

Runs on PC.

Command:

```bash
python3 /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/ui.py
```

The UI publishes:

- direct commands to `/xtend/cmd_nav`
- movement Twist commands to `/cmd_vel`

Direct commands:

```text
ARM
TAKEOFF
LAND
DISARM
STOP
```

Movement buttons:

```text
FORWARD
TURN LEFT
TURN RIGHT
STOP
```

Important safety behavior:

```text
STOP should publish zero Twist and {"action": "stop", "value": 0}
```

---

## Planner / FALCON — the object mission

The planner side is the **select-then-go object mission**: pick an object from the
room map's catalog, and the drone flies to it and lands. It is driven by
`tasks/planning/falcon/run_object_mission.sh`, and the launcher items below start
exactly what that script documents.

### The two-terminal workflow (items 12 and 13)

Loading the YOLO-World TensorRT engines is the slow part of a start; FALCON is the
part worth iterating on. So they are two separate commands:

```bash
# terminal A — start the detector once and leave it up
./run_object_mission.sh --detector-only

# terminal B — relaunch the mission as often as you like (seconds, no engine reload)
./run_object_mission.sh --falcon-only
```

- **`--detector-only`** runs *only* the YOLO-World detector, a ROS2 sidecar on the
  host (the FALCON container has no CUDA/TensorRT/pycuda). Nothing plans, nothing
  flies. It starts on a placeholder prompt and is re-prompted by the mission
  director the moment you select an object.
- **`--falcon-only`** runs the ros1↔ros2 bridge and the FALCON container (nav +
  A*/NavDP + object-approach + the mission director), reusing the sidecar already
  running — it neither starts nor, on exit, stops it. It **refuses to start** when
  no sidecar is running: nothing would ever publish a detection, so the mission
  could only ever land by A* alone while looking perfectly healthy.

The bridge is restarted with FALCON every time and cannot be kept: it is a ROS1
node against the roscore that roslaunch starts *inside* the container, so that
master dies with the container. A fresh roscore is wanted anyway — it is what
stops a stale latched goal from pre-arming the planners.

Item **14** runs all three in one session, for a one-shot run.

### NavDP inference server (item 11)

Every `nav_mode` except `astar` calls the NavDP point-goal policy, so start this
**before** the mission:

```bash
cd /home/user/agency_ws
export NAVDP_REPO=/home/user/GIT/NavDP/baselines/navdp
PYTHONPATH=$PWD python3 \
  -m sparx_agency.tasks.planning.navdp.server.navdp_trt_server \
  --port 8888 --engine-dir sparx_agency/tasks/planning/navdp/engines/orin_sm87
```

It runs on the FALCON **host**, not in the container: the Noetic image ships no
TensorRT, and FALCON reaches it over `--network host` loopback. Engines are built
per device and are not portable, so `--engine-dir` must name the tag of the machine
running it. Start it in the same power mode the engines were built in (MAXN +
`jetson_clocks`).

### Viewers (items 15 and 16)

```bash
# RViz: BEV map, planned path and odometry, already wired up
docker exec -it falcon bash -lc 'export DISPLAY=:0 && source /catkin_ws/devel/setup.bash && roslaunch exploration_manager rviz.launch'

# The interactive 2D BEV map with click-to-goal and the status HUD
docker exec -it falcon bash -lc 'export DISPLAY=:0 && source /catkin_ws/devel/setup.bash && rosrun falcon_adapter bev_click_goal_node.py'
```

The mission normally starts the BEV viewer itself (`bev_viewer` defaults to true),
so item 16 is only for a mission launched headless with `bev_viewer:=false`.

### Start-up order

```text
item 11  NavDP server        (skip only if nav_mode is astar)
item 12  FALCON A: detector  (start once, leave it up)
item 13  FALCON B: mission   (restart this as often as you like)
item 15  RViz                (optional)
```

---

## ROS1 Bridge Notes

The ROS1 bridge container must first be able to receive ROS2 topics.

Inside the container:

```bash
export ROS_DOMAIN_ID=5
timeout 5 ros2 topic echo /xtend/bearing --qos-reliability best_effort
```

If this does not print data, the ROS1 bridge will not work yet.

The bridge container should run with:

```text
--net=host
--ipc=host
```

If the container can discover topics but cannot receive data, use a FastDDS no-shared-memory profile.

Example file:

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_transport</transport_id>
      <type>UDPv4</type>
    </transport_descriptor>
  </transport_descriptors>

  <participant profile_name="participant_profile" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udp_transport</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
```

Then run the container with:

```bash
-e FASTRTPS_DEFAULT_PROFILES_FILE="/fastdds_no_shm.xml" \
-v "${SCRIPT_DIR}/fastdds_no_shm.xml:/fastdds_no_shm.xml:ro" \
```

---

## Debug Commands

### Check Active tmux Sessions

```bash
tmux ls
```

### Attach to a Node

```bash
tmux attach -t xtend_bridge
```

Detach:

```text
Ctrl+B then D
```

### Stop Converter Safely

```bash
ros2 topic pub --once /xtend/cmd_nav std_msgs/msg/String \
"{data: '{\"action\":\"stop\", \"value\":0}'}"

tmux send-keys -t xtend_twist_converter C-c
```

### Stop Bridge Safely

```bash
ros2 topic pub --once /xtend/cmd_nav std_msgs/msg/String \
"{data: '{\"action\":\"stop\", \"value\":0}'}"

tmux send-keys -t xtend_bridge C-c
```

### Check Duplicate Publishers

```bash
ros2 topic info /xtend/rgb -v
ros2 topic info /xtend/camera_info -v
ros2 topic info /xtend/depth_m -v
```

Expected:

```text
/xtend/rgb          Publisher count: 1
/xtend/camera_info  Publisher count: 1
/xtend/depth_m      Publisher count: 1
```

### Check Command Path

```bash
ros2 topic info /cmd_vel -v
ros2 topic info /xtend/cmd_nav -v
```

Expected:

```text
/cmd_vel:
  publisher: UI/planner/replayer
  subscriber: xtend_twist_to_cmd_nav

/xtend/cmd_nav:
  publisher: xtend_twist_to_cmd_nav or UI
  subscriber: xtend_online_bridge
```

---

## Safety Notes

- Only `online_nav_bridge_publisher.py` should own the XTEND WebSocket.
- Do not run multiple bridge instances.
- Do not run multiple depth processors publishing `/xtend/depth_m`.
- Use the UI for manual ARM / TAKEOFF / LAND / DISARM.
- Before stopping nodes, publish STOP if the drone may be moving.
- Closing the pipeline launcher does not stop tmux sessions. Nodes keep running until their sessions are stopped.
