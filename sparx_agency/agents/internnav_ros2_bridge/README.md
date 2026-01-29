# InternNav ROS2 Bridge (Gazebo + Sphera)

This repository provides a ROS2 bridge that connects camera + state topics to the InternVLA-N1 (InternNav) model server, and executes model actions in either:
1) Gazebo (Twist/cmd_vel), or
2) Rooster (ManualControl + KeepAlive)

---

## 1) Start the InternVLA-N1 Model Server (HOST, conda)

Run on the host (not in Docker):

```bash
cd ~/GIT/InternNav
conda activate internnav_server
python scripts/eval/start_server.py \
  --config scripts/eval/configs/h1_internvla_n1_async_cfg.py \
  --host 0.0.0.0 \
  --port 8087
````

---

## Option A: Gazebo

### A1) Run your Gazebo simulation (HOST or Docker)

Start your Gazebo world as you normally do, and make sure it publishes an RGB topic (e.g. `/camera/image_raw`) and accepts velocity commands (e.g. `/cmd_vel`).

### A2) Run the bridge (ROS2 environment)

Run inside the ROS2 environment that can see Gazebo topics (host or container):

```bash
source /opt/ros/foxy/setup.bash
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.bridge_node --config config/gazebo_internnav.yaml
```

### A3) Send an instruction

In another terminal (same ROS2 environment):

```bash
ros2 topic pub -1 /navigation/instruction std_msgs/msg/String "{data: 'Explore forward and avoid obstacles'}"
```

### A4) Monitor

```bash
ros2 topic echo /navigation/action
ros2 topic hz /cmd_vel
```

---

## Option B: Sphera

### B1) Start the camera stream → ROS2 Image (CONTAINER)

This publishes `/R1/camera/image_raw`:

```bash
source /opt/ros/foxy/setup.bash
cd ~/workspace/src/examples/src
python3 video_stream.py
```

### B2) Run the bridge (CONTAINER)

This connects to the model server and publishes control:

```bash
source /opt/ros/foxy/setup.bash
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.bridge_node --config config/rooster_r1_internnav.yaml
```

### B3) Run the interactive instruction console (CONTAINER)

Type an ID (1..N) to publish a predefined instruction from YAML, or type `0` for free-text:

```bash
source /opt/ros/foxy/setup.bash
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.instruction_console_node --ros-args \
  -p yaml_path:=config/prison_instructions.yaml \
  -p instruction_topic:=/R1/navigation/instruction
```

### B4) Monitor

```bash
ros2 topic echo /R1/navigation/action
ros2 topic hz /R1/manual_control
ros2 topic hz /R1/keep_alive
ros2 topic echo /R1/state
```

```
::contentReference[oaicite:0]{index=0}
```
