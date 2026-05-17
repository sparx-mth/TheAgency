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

## Launcher UI Buttons

### Start selected

Starts the selected item.

If the item is a Jetson command, it starts it over SSH inside a named `tmux` session.

If the item is a PC command, it opens a local terminal.

Manual/container commands are copied to the clipboard.

---

### Start checked Jetson core

Starts all checked Jetson pipeline nodes in separate `tmux` sessions.

Typical core sessions:

```text
xtend_bridge
xtend_depth
xtend_twist_converter
xtend_flow_depth
xtend_velocity_integrator
xtend_static_tf
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
  --timeout-sec 1.0
```

Calibration:

```text
linear.x = 0.3 m/s  -> forward thrust 400
forward max thrust  -> 600
turn thrust default -> 1000
```

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

## Planner / FALCON

### Start Hospital World

Runs manually on Jetson:

```bash
cd /home/user/GIT/TheAgency
./run_hospital.sh office
```

### Inside Planner Container

```bash
roslaunch falcon_adapter real_drone.launch map_name:=office
```

### ROS Bridge Docker

```bash
cd /home/user/GIT/sjtu_project/ros_bridge_docker
./run_bridge.sh
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
