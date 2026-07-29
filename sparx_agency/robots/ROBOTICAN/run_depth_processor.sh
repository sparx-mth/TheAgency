#!/bin/bash
# Runs rooster_depth_processor.py inside the `robotican_dev` container
# (docker-compose.robotican.yml / theagency:robotican), not the host venv.
#
# 2026-07-29: moved off the bare host venv (see docs/progress/entries/
# 002-rooster-full-containerize.md) -- matches run_rooster_frame_dir_publisher.sh's
# already-containerized pattern. robotican_dev already has TensorRT/pycuda and
# both bind mounts this needs: ~/depth_anything_ws (the engine) and
# ~/GIT/TheAgency (config_yaml, cage_static_mask.npy via bar_inpainter.py),
# confirmed live via `docker inspect robotican_dev`. /tmp/rooster_depth is
# also bind-mounted host<->container, so depth .npy output lands on the host
# exactly as before -- nothing downstream (mapping_sync via ros1_bridge, the
# new Jetson depth relay) needs to change.
#
# Reads JPEG paths from /R1/rgb_frame_path, runs DA3-TRT, publishes depth.
#
# ============== ROBOTICAN FIX (carried over from the pre-container script) ==
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
docker exec -it \
  -e ROS_DOMAIN_ID=9 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml" \
  robotican_dev bash -lc "
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=\$PYTHONPATH:/home/$USER/GIT/TheAgency
    python3 /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_depth_processor.py \
      --ros-args \
      -p engine_path:=\"/home/$USER/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine\" \
      -p config_yaml:=\"/home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml\" \
      -p frame_path_topic:=/R1/rgb_frame_path \
      -p camera_info_topic:=/R1/camera_info \
      -p camera_info_mode:=base \
      -p depth_topic:=/R1/depth_m \
      -p depth_path_topic:=/R1/depth_frame_path \
      -p depth_dir:=/tmp/rooster_depth \
      -p max_depth_kept:=500 \
      $*
  "
