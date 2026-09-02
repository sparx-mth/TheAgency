#!/bin/bash
# Runs video_trigger.py inside the 'it' container (needs Foxy + video_handler_interfaces).
# Overrides ROS_DOMAIN_ID to 6 so it matches the host ROBOTICAN pipeline.
# Copies video_trigger.py into the container each run — no volume mount needed.
#
# Usage: ./run_video_trigger.sh --drone-id R1 --host-ip 127.0.0.1 --port 5001
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker cp "${SCRIPT_DIR}/video_trigger.py" it:/tmp/video_trigger.py
docker exec \
  -e ROS_DOMAIN_ID=6 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml \
  it bash -lc \
  "source /opt/ros/foxy/setup.bash && \
   source /home/rooster/workspace/install/setup.bash && \
   python3 /tmp/video_trigger.py $*"
