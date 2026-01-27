# InternNav ROS2 Bridge

A highly configurable ROS2 bridge that connects **any simulation** to the [InternNav](https://github.com/InternRobotics/InternNav) navigation foundation model. This bridge handles topic mapping, data transformation, model inference, and action execution—all through a simple YAML configuration file.

## 🚀 Quick Start (Simple Drone Simulation)

### 1. Start the InternNav Server (on HOST)

```bash
cd ~/GIT/InternNav
conda activate internnav_server
python scripts/eval/start_server.py \
  --config scripts/eval/configs/h1_internvla_n1_async_cfg.py \
  --host 0.0.0.0 \
  --port 8087
```

> **Note:** The config file sets port 8023 by default. Use `--port 8087` to override, or update `simple_drone_config.yaml` to use port 8023.

### 2. Run the Bridge (inside Docker)

```bash
cd /root/sjtu_project
source install/setup.bash

ros2 run internnav_bridge bridge_node --ros-args \
  -p config_path:=/root/sjtu_project/src/internnav_ros2_bridge/config/simple_drone_config.yaml
```

### 3. Send Navigation Commands (in another Docker terminal)

```bash
# Send a single instruction
ros2 topic pub /simple_drone/navigation/instruction std_msgs/String "data: 'Fly forward to the door'" --once

# Or continuous publishing
ros2 topic pub /simple_drone/navigation/instruction std_msgs/String "data: 'Turn left and explore'"
```

### 4. Monitor the Drone

```bash
# Watch velocity commands being sent to drone
ros2 topic echo /simple_drone/cmd_vel

# Watch discrete actions
ros2 topic echo /simple_drone/navigation/action

# Check camera is publishing
ros2 topic hz /simple_drone/front/image_raw
```

---

## 🔧 Development: Rebuild After Changes

When you modify the bridge code on the host, rebuild inside Docker:

```bash
cd /root/sjtu_project

# Clean previous build
rm -rf build/internnav_bridge install/internnav_bridge

# Force clean pip cache (optional, helps with stubborn issues)
pip cache purge 2>/dev/null || true

# Rebuild
colcon build --packages-select internnav_bridge

# Source the updated workspace
source install/setup.bash

# Run the bridge
ros2 run internnav_bridge bridge_node --ros-args \
  -p config_path:=/root/sjtu_project/src/internnav_ros2_bridge/config/simple_drone_config.yaml
```

---

## 🎯 Features

- **Universal Compatibility**: Works with any ROS2 simulation (Isaac Sim, Gazebo, Habitat, Unity, custom)
- **Flexible Configuration**: Single YAML file defines all topic mappings and transformations
- **Built-in Presets**: Quick-start configurations for popular simulators
- **Multiple Output Modes**: Discrete actions, continuous velocities, or both
- **Action Executor**: Optional node to convert discrete actions to timed velocity commands
- **Mock Server**: Test the bridge without the actual model loaded

## 📁 Package Structure

```
internnav_bridge/
├── config/
│   └── bridge_config.yaml      # Main configuration file
├── launch/
│   ├── bridge.launch.py        # Launch bridge node only
│   └── full_system.launch.py   # Launch both server and bridge
├── internnav_bridge/
│   ├── __init__.py
│   ├── bridge_node.py          # Main ROS2 bridge node
│   ├── model_server.py         # FastAPI model server wrapper
│   ├── action_executor.py      # Discrete action → velocity converter
│   └── utils.py                # Utility functions
└── README.md
```

## 🔧 Installation

### Prerequisites

- ROS2 Humble or later
- Python 3.9+
- The InternNav model (optional for testing - mock mode available)

### Install the Package

```bash
# Clone or copy the package to your workspace
cd ~/your_ros2_ws/src
# Copy internnav_bridge here

# Install Python dependencies
pip install numpy opencv-python pyyaml requests fastapi uvicorn

# Build the workspace
cd ~/your_ros2_ws
colcon build --packages-select internnav_bridge
source install/setup.bash
```

## ⚙️ Configuration

The bridge is configured through a single YAML file. Edit `config/bridge_config.yaml` to match your simulation:

### Quick Setup: Use a Preset

```yaml
# At the bottom of bridge_config.yaml
use_preset: "gazebo"  # Options: isaac_sim, gazebo, habitat, unity, null
```

### Custom Configuration

#### 1. Configure the Model Server Connection

```yaml
bridge:
  server:
    host: "localhost"
    port: 8000
    protocol: "http"
```

#### 2. Map Your Input Topics

```yaml
inputs:
  rgb:
    enabled: true
    topic: "/your_camera/rgb/image_raw"  # ← Your RGB topic
    msg_type: "sensor_msgs/Image"
    preprocessing:
      target_size: [224, 224]
      color_convert: "bgr_to_rgb"
      normalize:
        enabled: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  
  instruction:
    enabled: true
    topic: "/navigation/instruction"
    default: "Navigate to the goal"
```

#### 3. Configure Your Output Topics

```yaml
outputs:
  mode: "discrete"  # or "continuous" or "both"
  
  discrete:
    enabled: true
    topic: "/your_robot/action"  # ← Your action topic
    action_mapping:
      "MOVE_FORWARD": "forward"   # Model action → Your action
      "TURN_LEFT": "left"
      "TURN_RIGHT": "right"
      "STOP": "stop"
  
  continuous:
    enabled: false
    topic: "/cmd_vel"
    action_to_velocity:
      "MOVE_FORWARD":
        linear_x: 0.25
        angular_z: 0.0
      "TURN_LEFT":
        linear_x: 0.0
        angular_z: 0.5236
```

## 🏃 Usage

### Option 1: Bridge Only (Connect to Existing Model Server)

If you already have the InternNav model server running:

```bash
# Start the bridge
ros2 launch internnav_bridge bridge.launch.py

# With custom config
ros2 launch internnav_bridge bridge.launch.py config_path:=/path/to/your/config.yaml

# With a preset
ros2 launch internnav_bridge bridge.launch.py use_preset:=gazebo
```

### Option 2: Full System (Start Server + Bridge)

Start both the model server and bridge:

```bash
ros2 launch internnav_bridge full_system.launch.py

# Specify device
ros2 launch internnav_bridge full_system.launch.py model_device:=cuda
```

### Option 3: Manual Start (Recommended for Development)

```bash
# Terminal 1: Start the model server
python3 -m internnav_bridge.model_server --port 8000 --device cuda

# Terminal 2: Start the bridge
ros2 run internnav_bridge bridge_node --config /path/to/config.yaml

# Terminal 3 (Optional): Start action executor
ros2 run internnav_bridge action_executor
```

## 📤 Sending Navigation Instructions

Publish instructions to the configured topic:

```bash
# Send a navigation instruction
ros2 topic pub /navigation/instruction std_msgs/String "data: 'Go to the kitchen, turn left, and find the red chair'"

# One-shot publish
ros2 topic pub --once /navigation/instruction std_msgs/String "data: 'Stop'"
```

## 📊 Monitoring

```bash
# Watch actions being published
ros2 topic echo /navigation/action

# Watch feedback
ros2 topic echo /navigation/feedback

# Watch status
ros2 topic echo /navigation/status

# View all topics
ros2 topic list | grep navigation
```

## 🔌 Integration Examples

### Isaac Sim Integration

```yaml
# config/bridge_config.yaml
use_preset: null

inputs:
  rgb:
    topic: "/front_stereo_camera/left_rgb/image_raw"
  depth:
    enabled: true
    topic: "/front_stereo_camera/left_depth/image_raw"
  instruction:
    topic: "/isaac_nav/instruction"

outputs:
  continuous:
    enabled: true
    topic: "/cmd_vel"
```

### Gazebo Classic Integration

```yaml
inputs:
  rgb:
    topic: "/camera/image_raw"
  instruction:
    topic: "/move_base_simple/instruction"

outputs:
  continuous:
    enabled: true
    topic: "/mobile_base/commands/velocity"
```

### Custom Simulation

```python
# In your simulation, publish RGB images:
# /your_sim/camera/rgb  (sensor_msgs/Image)

# Publish instructions when ready:
# /your_sim/nav/instruction  (std_msgs/String)

# Subscribe to actions:
# /your_sim/nav/action  (std_msgs/String)
```

## 🧪 Testing Without Model

The bridge includes a mock inference mode for testing:

```bash
# Start server in mock mode (no GPU needed)
python3 -m internnav_bridge.model_server --port 8000 --device cpu

# The mock server will return random actions based on instruction keywords
```

## 🔄 Action Executor (Optional)

If your simulation expects continuous velocity commands but you're using discrete actions:

```bash
# Start the action executor
ros2 run internnav_bridge action_executor

# Parameters
ros2 run internnav_bridge action_executor --ros-args \
    -p forward_velocity:=0.3 \
    -p forward_distance:=0.25 \
    -p turn_angle:=0.5236
```

The action executor:
1. Receives discrete actions ("forward", "left", "right", "stop")
2. Publishes velocity commands for the appropriate duration
3. Queues actions if they arrive during execution

## 📝 API Reference

### Topics Published by Bridge

| Topic | Type | Description |
|-------|------|-------------|
| `/navigation/action` | `std_msgs/String` | Discrete action output |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command (if enabled) |
| `/navigation/feedback` | `std_msgs/String` | JSON with action details |
| `/navigation/status` | `std_msgs/String` | Navigation status |

### Topics Subscribed by Bridge

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/rgb/image_raw` | `sensor_msgs/Image` | RGB camera input |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth camera (optional) |
| `/navigation/instruction` | `std_msgs/String` | Navigation instruction |
| `/odom` | `nav_msgs/Odometry` | Odometry (optional) |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal pose (optional) |

### Model Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/inference` | POST | Run inference |
| `/v1/batch_inference` | POST | Batch inference |

## 🐛 Troubleshooting

### Simple Drone Specific Issues

#### "Agent init failed: KeyError: 'InternVLA-N1'"
- The model name must be **lowercase**: `internvla_n1`
- Check `config/simple_drone_config.yaml` has `variant: "internvla_n1"`

#### "Connection refused to port 8087"
- Make sure the server is started with `--port 8087`
- Make sure server is bound to `0.0.0.0` not `localhost`
- Check server is running: `curl http://127.0.0.1:8087/openapi.json`

#### "ModuleNotFoundError: No module named 'requests'"
```bash
apt-get update && apt-get install -y python3-requests
# Or
/usr/bin/python3 -m pip install requests --break-system-packages
```

#### "No module named 'internnav_bridge'"
```bash
cd /root/sjtu_project
rm -rf build/internnav_bridge install/internnav_bridge
pip cache purge 2>/dev/null || true
colcon build --packages-select internnav_bridge
source install/setup.bash
```

#### "Config not found, using defaults"
- Check the config path is correct
- Verify the file exists: `ls -la /root/sjtu_project/src/internnav_ros2_bridge/config/`

### General Issues

### "Model server not responding"
- Check if the model server is running: `curl http://127.0.0.1:8087/openapi.json`
- Verify the host and port in your config match the server

### "No RGB images received"
- Verify your topic name: `ros2 topic list | grep image`
- Check the message type matches your config

### "Actions not being published"
- Check if instructions are being received: `ros2 topic echo /simple_drone/navigation/instruction`
- Verify the inference rate in config is reasonable

### "Slow inference"
- Use GPU: `--device cuda`
- Reduce image resolution in config
- Disable image history if not needed

