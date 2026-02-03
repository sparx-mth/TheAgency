# InternNav ROS2 Bridge

ROS2 bridge connecting camera topics to the InternVLA-N1 model server for vision-language navigation.

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│  InternNav      │◄────────────► │  ROS2 Bridge     │
│  Model Server   │   (port 8087) │  (bridge_node)   │
│  (HOST/conda)   │               │  (CONTAINER)     │
└─────────────────┘               └────────┬─────────┘
                                           │ ROS2 Topics
                                           ▼
                                  ┌──────────────────┐
                                  │  Gazebo / Sphera │
                                  │  Simulation      │
                                  └──────────────────┘
```

## Quick Start

### 1. Start Model Server (HOST)

```bash
cd ~/GIT/InternNav
source ~/miniconda3/etc/profile.d/conda.sh
conda activate internnav_server
python scripts/eval/start_server.py \
  --config scripts/eval/configs/h1_internvla_n1_async_cfg.py \
  --host 0.0.0.0 --port 8087
```

### 2. Run Bridge (CONTAINER)

**For Gazebo:**
```bash
python3 -m internnav_bridge.bridge_node --config config/gazebo_internnav.yaml
```

**For Sphera/Rooster:**
Before running in the Sphera docker container (one-time / as needed)
```bash
cd ~/sparx_agency/agents/internnav_ros2_bridge/scripts
chmod +x install_bridge_deps.sh
./install_bridge_deps.sh
```

```bash
# Terminal 1: Start camera stream (must run from this directory)
cd ~/workspace/src/examples/src
python3 video_stream.py
```
```bash
# Terminal 2: Run bridge (must run from this directory)
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.bridge_node --config config/rooster_r1_internnav.yaml
```

### 3. Send Navigation Instruction

```bash
ros2 topic pub -1 /R1/navigation/instruction std_msgs/msg/String "{data: 'Go forward to the door'}"
```
Optional: Interactive instruction console (YAML + CLI)
```bash
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.instruction_console_node --ros-args \
  -p yaml_path:=config/prison_instructions.yaml \
  -p instruction_topic:=/R1/navigation/instruction
```

### 4. Monitor

```bash
ros2 topic echo /R1/navigation/action      # See actions
ros2 topic echo /R1/navigation/feedback    # See feedback
```

### 5. Screen Recording
**Optional: Overlay screen recording (MP4)**
```bash
cd ~/sparx_agency/agents/internnav_ros2_bridge
python3 -m internnav_bridge.overlay_recorder_node --ros-args \
  -p rgb_topic:=/R1/camera/image_raw \
  -p rgb_type:=raw \
  -p instruction_topic:=/R1/navigation/instruction \
  -p action_topic:=/R1/navigation/action \
  -p status_topic:=/R1/navigation/status \
  -p output:=/tmp/internnav_overlay.mp4 \
  -p show_preview:=True
```
Type an instruction ID (e.g., 1) to publish the matching text from the YAML.

Type 0 to enter free-text instruction (typed manually).

Type q to quit.

The recording will be saved to: /tmp/internnav_overlay.mp4

**(Optional) quick preview:**
```bash
docker cp it:/tmp/internnav_overlay.mp4 ~/Downloads/
```

## Control Commands

Stop navigation explicitly:
```bash
ros2 topic pub -1 /R1/navigation/control std_msgs/msg/String "{data: 'done'}"
```

Commands: `done`, `pause`, `resume`, `reset`

## Files

```
internnav_bridge/
├── config/
│   └── rooster_r1_internnav.yaml   # Sphera config
├── internnav_bridge/
│   ├── bridge_node.py              # Main ROS2 node
│   ├── model_client.py             # HTTP client
│   ├── types.py                    # Data types
│   └── config.py                   # Config loader
└── README.md
```