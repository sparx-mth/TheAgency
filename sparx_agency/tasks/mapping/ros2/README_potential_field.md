# Potential Mapper ROS 2 Node

This node runs the local potential-field mapper on ROS 2 data, using:
## Overview

The node builds a local vector field:

- repulsive component from obstacles (depth → point cloud → grid)
- attractive component from goal
- total field = weighted sum of both

The resulting vector field is used for local navigation.


- RGB image input
- odometry / TF
- DepthAnything model inference
- local potential / navigation field publishing
- debug image publishing

It can be used live or with a ROS 2 bag.

---

## File

```bash
TheAgency/sparx_agency/tasks/mapping/ros2/potential_mapper_node.py
```

---
## Requirements
### ROS 2

This setup assumes:

* ROS 2 Humble
* Python 3.10
* use_sim_time:=true when working with bag playback

You need the packages used by the node and the mapper code, including at least:

* rclpy
* numpy
* opencv-python
* cv_bridge
* message_filters
* sensor_msgs_py
* tf2_ros
* geometry_msgs
* nav_msgs
* std_msgs
* yaml

### Project path

The project root must be available in `PYTHONPATH`:
```bash
/home/$USER/GIT/TheAgency
```
### Shared libraries

ROS 2 shared libraries must be visible in `LD_LIBRARY_PATH`.

---
## Environment setup
Before running the node, set:
```bash
export LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages:/home/daphnaa/GIT/TheAgency
source /opt/ros/humble/setup.bash
```
If you are inside a workspace that also has an install space, source it too if needed.

---
## Model and camera files

The node requires:

* an ONNX / TensorRT model file for depth inference
* a camera YAML file with intrinsics

Make sure the paths configured inside the node or passed through parameters point to valid files.

### Camera YAML

The camera YAML should contain intrinsics such as:
* image width / height
* fx, fy
* cx, cy

### Depth model
The depth model must match the inference wrapper used by self.depth_model.infer_all(cv_image).
See `DA3_README.md` for model setup.

---
## Running the node

Run:
```bash
source /opt/ros/humble/setup.bash

export LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages:/home/$USER/GIT/TheAgency

python3 /home/$USER/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/potential_mapper_node.py \
  --ros-args \
  -p use_sim_time:=true \
  -p odom_frame:="simple_drone/odom" \
  -p base_frame:="simple_drone/base_link"

```


---
## Node parameters

These parameters are expected by the node:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `engine_path` | string | path inside the node | TensorRT engine used for depth inference |
| `config_yaml` | string | path inside the node | Camera intrinsics YAML file |
| `base_frame` | string | `base_link` | Robot base frame used for TF and published outputs |
| `odom_frame` | string | `odom` | Odometry frame used for motion delta estimation |
| `size_m` | float | `6.5` | Local map size in meters |
| `show_gui` | bool | `True` | Enables the OpenCV visualization window |

---

## Expected inputs

### Subscribed topics
The node expects these topics to be available:

* `/simple_drone/front/image_raw`
* `/simple_drone/front/camera_info`
* `/planner_target_pixel`
* `/clicked_point`

## Required inputs

The node will not work without:

- RGB image: `/simple_drone/front/image_raw`
- Camera info: `/simple_drone/front/camera_info`
- TF: `odom → base_link`

And mark optional ones:

Optional:
- `/clicked_point` (for manual goal)
- `/planner_target_pixel`

### TF frames
The node expects a valid TF chain for motion estimation and clicked-point transforms, typically:

* `map`
* `odom`
* `base_link`

If your robot uses different frame names, update the parameters accordingly.

---

## Published topics

The node publishes:

* `/map_local` — local grid (repulsive / navigation map used by the planner, not a global SLAM map)* `/local_nav_vector` — normalized local navigation vector
* `/local_nav_heading` — navigation heading in radians
* `/potential_field_debug` — visual debug view of the field
* `/depth_debug` — depth visualization

---

## Notes on configuration

### Depth inference
The node uses a TensorRT depth wrapper. If the engine file or camera intrinsics file is invalid, startup may fail before the ROS spin loop begins.

### GUI mode
When `show_gui:=true`, an OpenCV window is created for click interaction and live feedback.

### Bag playback
When replaying recorded data, make sure:

* `/use_sim_time` is enabled
* the bag contains synchronized image and camera info messages
* TF is available during playback

---
## RViz setup

Set:

- Fixed Frame: `simple_drone/base_link`

Add displays:
- OccupancyGrid → `/map_local`
- Image → `/potential_field_debug`
- Image → `/depth_debug`

If the map does not update:
- verify `/use_sim_time`
- verify TF tree

---

## Suggested next checks

If the node does not start correctly, verify:

1. the engine file exists
2. the YAML file exists
3. ROS topics are active
4. TF frames are being published
5. the Python environment can import the required packages
6. OpenCV GUI is available if `show_gui` is enabled

---