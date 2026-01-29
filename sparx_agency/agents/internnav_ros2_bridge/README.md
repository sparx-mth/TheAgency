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
```bash
# Terminal 1: Start camera stream
python3 video_stream.py

# Terminal 2: Run bridge
python3 -m internnav_bridge.bridge_node --config config/rooster_r1_internnav.yaml
```

### 3. Send Navigation Instruction

```bash
ros2 topic pub -1 /R1/navigation/instruction std_msgs/msg/String "{data: 'Go forward to the door'}"
```

### 4. Monitor

```bash
ros2 topic echo /R1/navigation/action      # See actions
ros2 topic echo /R1/navigation/feedback    # See feedback
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