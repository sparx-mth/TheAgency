#!/bin/bash
# Restart video_trigger.py when the camera stops producing NEW frames.
#
# video_trigger keeps writing files after its video session dies -- byte-identical
# ones -- so "the directory is being written to" proves nothing and a file-mtime
# check reads healthy through a total freeze. Hashing the newest frame is what
# actually distinguishes a live camera from a stuck one.
#
# It fires often: 1550 restarts logged over the campaign's first three days,
# against four cycles that still aborted on stale frames (~1% of cycles). So it
# is load-bearing, not belt-and-braces -- without it the abort rate would be far
# higher.
#
# This file was reconstructed into the repo on 2026-08-21 from the copy that had
# been running out of /tmp since 2026-08-18. bringup.start_video_watchdog()
# referenced this path, the path did not exist, and the only reason the campaign
# had a watchdog at all was that the /tmp instance never died: pgrep found it,
# so the spawn that would have failed was never attempted. Had it been killed,
# `bash <missing file>` would have failed silently and nothing checks the result.
set -u

FRAME_DIR="${FRAME_DIR:-/tmp/rooster_frames}"
STALE_THRESHOLD="${STALE_THRESHOLD:-3}"     # consecutive identical frames
CHECK_INTERVAL="${CHECK_INTERVAL:-3}"       # seconds between checks
REPO="${SPARX_REPO:-/home/user1/GIT/TheAgency}"

cd "$REPO" || exit 1

last_hash=""
stale_count=0
while true; do
  sleep "$CHECK_INTERVAL"
  latest=$(ls -t "$FRAME_DIR"/*.jpg 2>/dev/null | head -1)
  [ -z "$latest" ] && continue
  hash=$(md5sum "$latest" 2>/dev/null | cut -d' ' -f1)
  [ -z "$hash" ] && continue
  if [ "$hash" == "$last_hash" ]; then
    stale_count=$((stale_count + 1))
  else
    stale_count=0
  fi
  last_hash="$hash"
  if [ "$stale_count" -ge "$STALE_THRESHOLD" ]; then
    echo "$(date '+%H:%M:%S') STALE -> restarting video_trigger"
    docker exec it pkill -f video_trigger.py
    sleep 1
    docker cp sparx_agency/robots/ROBOTICAN/video_trigger.py it:/tmp/video_trigger.py
    docker exec -d -e ROS_DOMAIN_ID=9 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
      -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml it bash -lc "
      source /opt/ros/foxy/setup.bash && source /home/rooster/workspace/install/setup.bash
      python3 /tmp/video_trigger.py --drone-id R1 --host-ip 127.0.0.1 --port 5001 \
        --width 540 --height 360 > /tmp/video_trigger.log 2>&1
    "
    stale_count=0
    sleep 6
  fi
done
