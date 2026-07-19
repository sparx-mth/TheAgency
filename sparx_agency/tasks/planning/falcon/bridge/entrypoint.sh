#!/bin/bash
set -e

# Base settings
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-5}"
# ROS_HOSTNAME=localhost is critical when multiple containers run on
# --net=host. Without it, each container registers ROS nodes under
# its container hostname, which doesn't resolve from other containers
# even though they all share the host's network namespace. Result:
# rostopic list works (talks to master only), but pub/sub connections
# fail with "Failed to connect to 127.0.0.1:<random port>".
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"

LOGFILE="${LOGFILE:-/tmp/bridge.log}"
BRIDGE_YAML="${BRIDGE_YAML:-/bridge.yaml}"

# Redirect output to log file
: > "${LOGFILE}" 2>/dev/null || true
exec > >(tee -a "${LOGFILE}") 2> >(tee -a "${LOGFILE}" >&2)

source /opt/ros/noetic/setup.bash

echo "================================================"
echo "  ros1_bridge - parameter_bridge (QoS-aware)"
echo "  ROS1 master    : ${ROS_MASTER_URI}"
echo "  ROS_HOSTNAME   : ${ROS_HOSTNAME}"
echo "  ROS2 domain    : ${ROS_DOMAIN_ID}"
echo "  RMW (ROS2)     : ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp (default)}"
echo "  Bridge config  : ${BRIDGE_YAML}"
echo "  Log file       : ${LOGFILE}"
echo "================================================"

# Wait for roscore
echo "[bridge] Waiting for roscore..."
until timeout 2 rostopic list >/dev/null 2>&1; do
    sleep 1
done
echo "[bridge] roscore reachable."

# Sanity-check the yaml is mounted/copied where we expect.
if [ ! -f "${BRIDGE_YAML}" ]; then
    echo "[bridge] ERROR: ${BRIDGE_YAML} not found inside the container."
    echo "[bridge]        run_bridge.sh should mount it as /bridge.yaml"
    exit 1
fi

# Load the bridge config into the ROS1 parameter server.
# parameter_bridge reads its topic/qos list from rosparam, not
# directly from a yaml file — so we have to load it here first.
# Loading under a namespace (here: empty / root) means parameter_bridge
# finds it at the default location. If you want to load multiple
# bridge configs side-by-side, load each under its own namespace
# and pass the namespace name as parameter_bridge's first arg.
echo "[bridge] Loading ${BRIDGE_YAML} into rosparam..."
rosparam load "${BRIDGE_YAML}"

# Source ROS2 + bridge workspace
source /opt/ros/foxy/setup.bash
source /bridge_ws/install/setup.bash

# Run parameter_bridge in a restart loop.
# parameter_bridge only bridges what's in bridge.yaml — explicit,
# deterministic, with per-topic QoS. It does NOT auto-discover new
# topics like dynamic_bridge does, so if you add a topic on either
# side at runtime, you have to edit bridge.yaml and restart this
# container.
while true; do
    echo "[bridge] Starting parameter_bridge..."
    ros2 run ros1_bridge parameter_bridge
    echo "[bridge] parameter_bridge exited. Restarting in 3s..."
    sleep 3
done