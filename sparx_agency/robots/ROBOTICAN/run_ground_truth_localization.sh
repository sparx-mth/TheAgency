#!/bin/bash
# ============== ROBOTICAN FIX ==============
# New file, ROBOTICAN/Rooster-only. Not used by, and doesn't touch,
# XTEND or Jetson.
# ============================================
# Run rooster_ground_truth_localization for ROBOTICAN/Rooster (Sphera sim only).
# Republishes Sphera's ground-truth pawn pose as PoseStamped on /R1/localization
# (bridged to ROS1 by the falcon/bridge container's bridge.yaml).
#
# MUST run inside container `it`, NOT on the host: it imports
# sphera_common_interfaces/msg/SpheraPawnState, which is only built for ROS2
# Foxy inside `it` (sphera-backend:rooster-with-sparx) -- see
# DOME_CAPTURE_README.md's "Where things run" table. sparx_agency is
# bind-mounted read-write into `it` at /home/rooster/sparx_agency, so this
# script (and any host edits) are live there immediately.
#
# If you don't need the sim ground-truth shortcut, prefer the already-working
# AprilTag-based localization_node.py command in DOME_CAPTURE_README.md
# Terminal 3 instead -- it runs on the host, no `it` dependency.
docker exec it bash -c '
  source /opt/ros/foxy/setup.bash
  source /home/rooster/workspace/install/setup.bash
  export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
  cd /home/rooster
  python3 -m sparx_agency.robots.ROBOTICAN.rooster_ground_truth_localization \
    --ros-args -p rooster_id:=R1 '"$@"'
'
