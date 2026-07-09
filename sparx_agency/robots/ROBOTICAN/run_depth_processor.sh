#!/bin/bash
# Run depth_processor_node for ROBOTICAN/Rooster.
# Reads JPEG paths from /R1/rgb_frame_path, runs DA3-TRT, publishes depth.
#
# Engine expected at:
#   ~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_trt10.engine
# Override via: --ros-args -p engine_path:=<path>
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
export PYTHONPATH=$PYTHONPATH:/home/$USER/GIT/TheAgency

exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p engine_path:="$HOME/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_trt10.engine" \
  -p config_yaml:="$HOME/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml" \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p camera_info_topic:=/R1/camera_info \
  -p camera_info_mode:=base \
  -p depth_topic:=/R1/depth_m \
  -p depth_path_topic:=/R1/depth_frame_path \
  -p depth_dir:=/tmp/rooster_depth \
  "$@"
