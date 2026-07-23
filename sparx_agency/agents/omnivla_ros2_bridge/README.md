# OmniVLA ROS2 Bridge

ROS2 bridge for [OmniVLA](https://github.com/NHirose/OmniVLA) omni-modal navigation on Sphera / Rooster R1.

## Architecture

Everything runs on the **HOST** (GPU + conda + ROS2). The Sphera **container** is untouched.

```
HOST (conda, GPU, ROS2)                     CONTAINER (Sphera)
┌────────────────────────────┐               ┌─────────────────────┐
│  omnivla_ros2_bridge/      │    ROS2 DDS   │                     │
│  bridge_node.py            │◄─────────────►│  /R1/camera/image   │
│    ├─ loads OmniVLA model  │               │  /R1/manual_control │
│    ├─ subscribes to camera │               │  /R1/keep_alive     │
│    ├─ subscribes to goals  │               │  /R1/state          │
│    ├─ runs inference (GPU) │               │                     │
│    └─ publishes velocity   │               └─────────────────────┘
└────────────────────────────┘
```

## Files

```
~/GIT/
├── OmniVLA/                          # cloned repo + checkpoints
│   ├── omnivla-original/             # HuggingFace checkpoints
│   └── prismatic/                    # model code
│
└── omnivla_ros2_bridge/              # this repo
    ├── config/
    │   └── rooster_r1_omnivla.yaml
    ├── omnivla_ros2_bridge/
    │   ├── __init__.py
    │   ├── bridge_node.py            # ROS2 node
    │   └── omnivla_model.py          # model wrapper (no ROS)
    └── README.md
```

## Setup (HOST — one time)

```bash
# 1. Clone OmniVLA + checkpoints
cd ~/GIT
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
git clone https://huggingface.co/NHirose/omnivla-original

# 2. Create conda env (follow OmniVLA SETUP.md), then verify:
conda activate omnivla
python -c "from prismatic.vla.action_tokenizer import ActionTokenizer; print('OK')"

# 3. Install bridge dependencies
pip install pyyaml opencv-python-headless

# 4. Place the bridge
cp -r omnivla_ros2_bridge ~/GIT/omnivla_ros2_bridge

# 5. Verify host sees container topics (with Sphera running)
source /opt/ros/humble/setup.bash
ros2 topic list | grep R1
```

## Running

**Terminal 1** — Sphera container (no changes):
Start Sphera as usual.

**Terminal 2** — Bridge (HOST):
```bash
conda activate omnivla
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PYTHONPATH:~/GIT/OmniVLA

cd ~/GIT/omnivla_ros2_bridge
python3 -m omnivla_ros2_bridge.bridge_node --config config/rooster_r1_omnivla.yaml
```

Wait for `OmniVLA model ready ✓`, then set goals from **Terminal 3** (HOST).

---

## Goal Modalities

OmniVLA supports multiple goal types. Set any combination — the bridge auto-detects which modality to use. Navigation starts when **any goal** is published.

### Supported Combinations

| ID | Goals active | Example use case |
|----|---|---|
| 7 | language only | "move toward the blue door" |
| 6 | image only | navigate to where the goal photo was taken |
| 4 | pose only | go to coordinates (x, y) with heading |
| 8 | language + pose | "go carefully" + target position |
| 5 | pose + image | target position + visual confirmation |

> **Note:** language + image (without pose) is NOT a trained OmniVLA modality. The bridge will warn and fall back to language-only. To use language + image, also add a goal pose.

### Language Instruction

Publish a text string. The robot starts navigating immediately.

```bash
# Set
ros2 topic pub -1 /R1/navigation/instruction std_msgs/msg/String \
  "{data: 'move toward the blue door'}"

# Clear (empty string or "clear")
ros2 topic pub -1 /R1/navigation/instruction std_msgs/msg/String \
  "{data: ''}"
```

### Goal Image

Publish an egocentric image of where you want the robot to go.

```bash
# From a file (using ros2 bag or image_publisher)
ros2 run image_publisher image_publisher_node /path/to/goal.jpg \
  --ros-args -r image_raw:=/R1/navigation/goal_image -p publish_rate:=1.0
```

Or set a static file in the YAML config (no topic needed):
```yaml
inputs:
  goal_image:
    enabled: false
    file_path: "/path/to/goal.jpg"
```

### Goal Pose

Publish 4 float values as a `Float32MultiArray`:

```
[rel_y / spacing,  -rel_x / spacing,  cos(heading_diff),  sin(heading_diff)]
```

Where:
- `rel_x`, `rel_y` — target position relative to robot in the robot's local frame (meters)
- `heading_diff` — difference between goal heading and robot heading (radians)
- `spacing` — waypoint spacing, default 0.1m (so values are in units of 0.1m)

```bash
# Example: target is 5m forward, 2m to the left, same heading
# rel_x=5, rel_y=2, heading_diff=0 → [2/0.1, -5/0.1, cos(0), sin(0)] = [20, -50, 1.0, 0.0]
ros2 topic pub -1 /R1/navigation/goal_pose std_msgs/msg/Float32MultiArray \
  "{data: [20.0, -50.0, 1.0, 0.0]}"
```

### Combining Modalities

Publish to multiple topics — the bridge combines them automatically:

```bash
# Language + pose (modality 8):
ros2 topic pub -1 /R1/navigation/instruction std_msgs/msg/String \
  "{data: 'go carefully and avoid obstacles'}"
ros2 topic pub -1 /R1/navigation/goal_pose std_msgs/msg/Float32MultiArray \
  "{data: [20.0, -50.0, 1.0, 0.0]}"
```

### Clearing Individual Goals

```bash
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'clear_language'}"
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'clear_image'}"
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'clear_pose'}"
```

---

## Navigation Control

```bash
# Stop and clear all goals
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'done'}"

# Pause (keep goals, stop moving)
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'pause'}"

# Resume
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'resume'}"

# Reset (clear all goals)
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'reset'}"
```

## Monitoring

```bash
ros2 topic echo /R1/cmd_vel               # Twist velocity
ros2 topic echo /R1/navigation/feedback    # JSON: linear, angular, modality, inference_ms
ros2 topic echo /R1/navigation/status      # navigating / idle / paused
```

Feedback example:
```json
{"linear": 0.21, "angular": -0.05, "modality": "language+pose", "inference_ms": 312}
```

## Trajectory Visualization

OmniVLA predicts 8 waypoints each cycle. The bridge can visualize them in real-time — this is adapted from the paper's own `save_robot_behavior()` code.

Enable in YAML:
```yaml
visualization:
  enabled: true
  topic: "/R1/navigation/trajectory_viz"
  save_dir: "/tmp/omnivla_viz"              # optional: also save as JPGs
```

View live:
```bash
ros2 run rqt_image_view rqt_image_view /R1/navigation/trajectory_viz
```

What you see:

```
┌──────────────────┬──────────────────────────────────────┐
│ Current camera   │                                      │
│                  │   Bird's-eye trajectory plot          │
│                  │                                      │
├──────────────────┤   ▲ = Robot at origin                │
│ Goal image       │   ● = 8 predicted waypoints (blue)   │
│ (if set)         │   ◆ = Tracked waypoint (orange)      │
│                  │   ★ = Goal pose (red, if set)        │
│                  │                                      │
│                  │   Modality: language+pose             │
│                  │   v=0.21 m/s  ω=-0.05 rad/s          │
└──────────────────┴──────────────────────────────────────┘
```

If `save_dir` is set, frames are saved as `000000.jpg`, `000001.jpg`, etc. — useful for post-analysis or making videos.

---

## ManualControl Mapping (Ground Roll)

| State | x (forward) | z (tilt) | r (yaw) |
|---|---|---|---|
| Moving forward | lin × 800 | lin × 1300 (cap 500) | ang × 2000 |
| Turning in place | 0 | 250 | ang × 2000 |
| Stopped | 0 | -1000 (brake) | 0 |

## Tuning

Edit `config/rooster_r1_omnivla.yaml`:

- **Robot too slow** → increase `linear_scale`, `tilt_scale`
- **Turns too weak** → increase `angular_scale`
- **Robot lifts off** → decrease `max_tilt`
- **Waypoint tracking** → `model.waypoint_index` (0=near, 7=far)