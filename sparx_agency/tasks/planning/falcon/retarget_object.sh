#!/usr/bin/env bash
# ============================================================
# retarget_object.sh <object>  --  switch the hunted object at runtime.
#
# Publishes the new prompt on /object_approach/goal from the HOST (ROS2) side
# with the QoS the consumers REQUIRE: RELIABLE + TRANSIENT_LOCAL. This matters --
# the detector sidecar and the ros1<->ros2 bridge both subscribe /object_approach/goal
# as TRANSIENT_LOCAL, so a naive `ros2 topic pub` (VOLATILE by default) is
# durability-incompatible and gets dropped SILENTLY: the object never switches
# and nothing errors. A TRANSIENT_LOCAL publisher is compatible with both, so one
# publish fans out to BOTH the host detector (re-prompt, open-vocab, no rebuild)
# AND, across the bridge, the in-container closure (re-key its confirmation gate).
#
# Usage:
#   ./retarget_object.sh knife
#   ./retarget_object.sh "person"
#
# Equivalent from INSIDE the FALCON container (ROS1 -- no QoS flags needed, the
# bridge's ROS2 side is already reliable/transient_local):
#   rostopic pub -1 /object_approach/goal std_msgs/String "data: 'knife'"
# ============================================================
set -euo pipefail

OBJ="${1:?usage: retarget_object.sh <object>   e.g. ./retarget_object.sh knife}"

# ROS setup scripts reference unbound vars / return nonzero -- guard set -e/-u.
set +u +e; source /opt/ros/humble/setup.bash; set -u -e   # shellcheck disable=SC1091

echo "[retarget] switching hunted object -> '$OBJ'"
# Publish a few times at 2 Hz so discovery completes before the publisher exits;
# transient_local keeps the last sample latched for the publisher's lifetime.
exec ros2 topic pub -t 3 -r 2 /object_approach/goal std_msgs/msg/String \
  "{data: '$OBJ'}" --qos-reliability reliable --qos-durability transient_local
