# NoMaD ROS2 Bridge

ROS2 bridge for [NoMaD](https://github.com/robodhruv/visualnav-transformer) visual navigation on Sphera / Rooster R1.

This bridge replaces the original `pd_controller.py` from the NoMaD repo.
NoMaD publishes waypoints, the bridge converts them to ManualControl commands.

## Architecture

```
HOST (conda, GPU, ROS2)                     CONTAINER (Sphera)
┌────────────────────────────────┐           ┌─────────────────────┐
│                                │  ROS2 DDS │                     │
│  NoMaD  (explore.py)           │◄─────────►│  /R1/camera/image   │
│    ├─ subscribes to camera     │           │                     │
│    └─ publishes /waypoint      │           │                     │
│                                │           │                     │
│  nomad_ros2_bridge/            │           │                     │
│  bridge_node.py                │──────────►│  /R1/manual_control │
│    ├─ subscribes to /waypoint  │           │  /R1/keep_alive     │
│    ├─ PD control               │◄──────────│  /R1/fcu/state      │
│    └─ publishes ManualControl  │           │                     │
└────────────────────────────────┘           └─────────────────────┘
```

## Files

```
~/GIT/
├── visualnav-transformer/            # cloned NoMaD repo
│   ├── train/                        # training code + vint_train package
│   └── deployment/
│       ├── model_weights/            # .pth files
│       ├── topomaps/                 # recorded maps
│       └── src/
│           ├── explore.py
│           ├── navigate.py
│           ├── create_topomap.py
│           ├── record_bag.sh
│           └── topic_names.py        # ← copy ours here
│
├── diffusion_policy/                 # cloned diffusion_policy repo
│
└── nomad_ros2_bridge/                # this repo
    ├── config/
    │   └── rooster_r1_nomad.yaml
    ├── nomad_ros2_bridge/
    │   ├── __init__.py
    │   └── bridge_node.py
    ├── topic_names.py                # copy into NoMaD repo
    └── README.md
```

## Setup (one time)

### 1. Install ROS2 Humble

Follow the official guide: https://docs.ros.org/en/humble/Installation.html

### 2. Clone NoMaD and create the conda environment

```bash
cd ~/GIT
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer

# Create conda env from the deployment environment file
conda env create -f deployment/deployment_environment.yaml
conda activate vint_deployment
```

### 3. Install the vint_train package

```bash
# Run from inside visualnav-transformer/
pip install -e train/
```

### 4. Install the diffusion_policy package

```bash
cd ~/GIT
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

### 5. Download model weights

Download the `.pth` files from [this Google Drive link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing) and place them in:

```
visualnav-transformer/deployment/model_weights/
```

### 6. Point NoMaD at the Sphera camera

The original NoMaD code subscribes to `/usb_cam/image_raw`.
Copy our override so it subscribes to the Sphera camera instead:

```bash
cp ~/GIT/nomad_ros2_bridge/topic_names.py \
   ~/GIT/visualnav-transformer/deployment/src/topic_names.py
```

### 7. Install bridge dependencies

```bash
conda activate vint_deployment
pip install pyyaml
```

### 8. Verify topics

With the Sphera container running:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep R1
```

You should see `/R1/camera/image_raw`, `/R1/manual_control`, etc.

## Running

### Terminal 1 — Sphera container

Start Sphera as usual (no changes needed).

### Terminal 2 — NoMaD model

```bash
conda activate vint_deployment
source /opt/ros/humble/setup.bash
cd ~/GIT/visualnav-transformer/deployment/src

# Exploration (no map needed):
python explore.py --model nomad

# Or navigation (with a recorded topomap):
python navigate.py --model nomad --dir ../topomaps/images/<topomap_name>
```

### Terminal 3 — Bridge

```bash
conda activate vint_deployment
source /opt/ros/humble/setup.bash
cd ~/GIT/nomad_ros2_bridge
python3 -m nomad_ros2_bridge.bridge_node --config config/rooster_r1_nomad.yaml
```

## Monitoring

```bash
ros2 topic echo /waypoint                 # raw waypoint from NoMaD [dx, dy]
ros2 topic echo /R1/cmd_vel               # Twist velocity
ros2 topic echo /R1/navigation/feedback    # JSON feedback
```

Feedback example:
```json
{"dx": 0.42, "dy": -0.08, "linear": 0.30, "angular": -0.16, "t": 1706889600.0}
```

## Topological Map (for navigation mode)

Run these inside `visualnav-transformer/deployment/src/`:

```bash
# 1. Record a demo trajectory
./record_bag.sh my_map

# 2. Create topomap from the recorded bag
./create_topomap.sh my_map my_map_*.bag

# 3. Navigate using the map
python navigate.py --model nomad --dir ../topomaps/images/my_map
```

## PD Controller

NoMaD publishes a waypoint `[dx, dy]` in the robot frame (dx=forward, dy=left).
The bridge computes:

| Condition | Linear vel | Angular vel |
|---|---|---|
| Distance < 0.08 m | 0 | 0 |
| Heading error > 57° | 0 (turn in place) | kp_ang × heading |
| Otherwise | kp_lin × dx | kp_ang × heading |

## ManualControl Mapping (Ground Roll)

| State | x (forward) | z (thrust) | r (yaw) |
|---|---|---|---|
| Moving forward | lin × 800 | cruise_thrust (400) | ang × 2000 |
| Turning in place | 0 | turn_thrust (300) | ang × 2000 |
| Stopped | 0 | 0 | 0 |

## Tuning

Edit `config/rooster_r1_nomad.yaml`:

- **Robot too slow** → increase `kp_linear`, `linear_scale`
- **Turns too weak** → increase `kp_angular`, `angular_scale`
- **Robot lifts off** → decrease `cruise_thrust`, `turn_thrust`
- **Not enough traction** → increase `cruise_thrust`, `turn_thrust`
- **Oscillating** → decrease `kp_angular`, increase `waypoint_reached_m`
- **Doesn't turn before moving** → decrease `turn_in_place_rad`