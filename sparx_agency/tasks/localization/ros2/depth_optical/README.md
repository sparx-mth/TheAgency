# Depth + Optical Flow Localization (ROS2)

This project demonstrates monocular localization using a combination of:

- Depth estimation (Depth Anything V2)
- Optical Flow
- Depth-scaled motion integration
- Evaluation against Ground Truth (GT Pose)

The system runs fully in ROS2, supports rosbag playback, and is designed for indoor drone scenarios.

---

## Overview

The pipeline is composed of three main stages:

1. Depth Estimation  
   Infer dense depth from monocular RGB images.

2. Motion Estimation (Flow + Depth)  
   Combine optical flow with depth to estimate metric velocity.

3. Pose Integration & Evaluation  
   Integrate velocity over time to estimate position and compare against GT pose.

---

## Architecture

RGB Image  
↓  
Depth Anything V2  
↓  
Depth Map + Visualization  
↓  
Optical Flow + Depth  
↓  
Metric Velocity  
↓  
Pose Integration  
↓  
GT Pose Error & RMS Drift

---

## ROS Topics

### Input
- /simple_drone/front/image_raw
- /simple_drone/front/camera_info
- /simple_drone/odom
- /clock

### Output
- /depth_anything/depth
- /depth_anything/depth_vis
- /flow_depth/velocity
- Console pose error logs

---

## Prerequisites

- ROS2 Humble
- Python 3.10
- DepthAnythingV2 installed
- sparx_agency workspace built
- Rosbag of sjtu drone containing RGB, CameraInfo, Odometry, TF 

---

## How to Run

### 1. Depth Estimation

```bash
python3 -m python3 -m sparx_agency.tasks.mapping.create_map_from_video \
  --ros-args \
  -p use_sim_time:=true \
  -r /debug/depth_raw:=/depth_anything/depth \
  -r /debug/depth_vis:=/depth_anything/depth_vis \
  -r /debug/cloud_global:=/depth_anything/cloud
```

### 2. Flow + Depth Velocity

```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node_ros \
  --ros-args -p use_sim_time:=true -p show_debug:=true
```

### 3. Pose Evaluation

```bash
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_pose_eval_node_ros \
  --ros-args \
  -p use_sim_time:=true \
  -p target_frame:=/simple_drone/odom
```

### 4. Play Rosbag

```bash
ros2 bag play rosbag2_2026_01_20-09_37_20/ --clock --rate 0.1
```

if you want to record
```bash

ros2 bag record   /clock   /simple_drone/front/image_raw   /simple_drone/front/camera_info   /simple_drone/gt_pose   /simple_drone/gt_vel   /simple_drone/odom   /simple_drone/imu   /tf   /tf_static


apt-get install -y \
  ros-humble-rosbag2 \
  ros-humble-rosbag2-storage-default-plugins \
  ros-humble-rosbag2-compression-zstd
```
---

## RViz Setup

- Fixed Frame: simple_drone/odom
- Enable Use Sim Time
- Add Image displays for:
  - /simple_drone/front/image_raw
  - /depth_anything/depth_vis
- Add PointCloud2 for:
  - /debug/cloud_global
- Add map for:
  - /occupancy_grid
---

## Evaluation Output

Example:

[PoseEval] err_last=0.86 m, err_rms(300)=0.73 m

- err_last: current position error
- err_rms: accumulated drift

---
