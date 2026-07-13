#!/bin/bash
# Run depth_processor_node for ROBOTICAN/Rooster.
# Reads JPEG paths from /R1/rgb_frame_path, runs DA3-TRT, publishes depth.
#
# ============== ROBOTICAN FIX ==============
# This whole script is ROBOTICAN/Rooster-only (not shared with XTEND/Jetson),
# so the two changes below can't regress anything on those platforms.
# - engine_path corrected from a nonexistent "*_fp16_trt10.engine" filename
#   to the actual file on disk, "*_fp16_546x364.engine" (matches
#   DOME_CAPTURE_README.md's Terminal 3 command).
# - max_depth_kept:=500 added: default (30) rotated .npy files out of
#   /tmp/rooster_depth faster than mapping_sync (ROS1 side, via ros1_bridge)
#   could read them, so every depth frame dropped with ENOENT.
#
# Engine expected at:
#   ~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine
# Override via: --ros-args -p engine_path:=<path>
# ============================================
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
export PYTHONPATH=$PYTHONPATH:/home/$USER/GIT/TheAgency

exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_depth_processor.py \
  --ros-args \
  -p engine_path:="$HOME/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine" \
  -p config_yaml:="$HOME/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml" \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p camera_info_topic:=/R1/camera_info \
  -p camera_info_mode:=base \
  -p depth_topic:=/R1/depth_m \
  -p depth_path_topic:=/R1/depth_frame_path \
  -p depth_dir:=/tmp/rooster_depth \
  -p max_depth_kept:=500 \
  "$@"
