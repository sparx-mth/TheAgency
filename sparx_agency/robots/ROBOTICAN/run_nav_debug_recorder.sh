#!/bin/bash
# Run nav_debug_ros2_recorder inside container `it` (ROS2 Foxy, domain 9).
#
# This is the ROS2 half of a nav-debug recording: the actuator lane
# (/R1/cmd_nav, /R1/manual_control) and the ground truth (/R1/velocity_truth,
# /R1/attitude_rpy, /R1/sphera/state, /R1/state, /R1/rooster_status), none of
# which the ROS1 recorder in `falcon` can see -- bridge.yaml carries none of
# them. The ROS1 half keeps running where it already does; this adds to it.
#
# Subscribe-only. It publishes nothing, commands nothing and starts no flight
# node; stopping it (Ctrl-C, or SIGTERM) only ends the recording.
#
# MUST run inside `it`, NOT on the host: it imports fcu_driver_interfaces,
# sphera_common_interfaces and rooster_manager_interfaces, which are only built
# for Foxy inside `it` (sphera-backend:rooster-with-sparx) -- same reason as
# run_ground_truth_localization.sh. sparx_agency is bind-mounted read-write into
# `it` at /home/rooster/sparx_agency, so host edits are live there immediately.
#
# WHERE THE FILES LAND: /home/rooster/workspace is a read-write bind of a host
# directory, so the default run folder (<workspace>/nav_debug_logs/nav_debug_
# <stamp>/ros2) is on the host as it is written -- no `docker cp` needed, and it
# survives `it` being recreated on a Sphera restart. Set NAV_DEBUG_RUN_DIR to
# point this recorder at the same run-folder stamp the ROS1 recorder is using;
# ROOSTER_WORKSPACE overrides the mount point. Both are forwarded below.
#
# Usage:
#   ./run_nav_debug_recorder.sh
#   ./run_nav_debug_recorder.sh -p record_hz:=20.0 -p duration_sec:=3600.0
#   NAV_DEBUG_RUN_DIR=/home/rooster/workspace/nav_debug_logs/run_42 \
#     ./run_nav_debug_recorder.sh
docker exec \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-9}" \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///etc/cyclonedds.xml}" \
  -e NAV_DEBUG_RUN_DIR="${NAV_DEBUG_RUN_DIR:-}" \
  -e ROOSTER_WORKSPACE="${ROOSTER_WORKSPACE:-}" \
  it bash -c '
    source /opt/ros/foxy/setup.bash
    source /home/rooster/workspace/install/setup.bash
    export PYTHONPATH=/home/rooster:$PYTHONPATH
    cd /home/rooster
    python3 -m sparx_agency.robots.ROBOTICAN.nav_debug_ros2_recorder \
      --ros-args -p rooster_id:=R1 "$@"
  ' _ "$@"
