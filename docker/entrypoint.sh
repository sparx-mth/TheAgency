#!/bin/bash
# ============================================================
# entrypoint.sh -- generic, shared by every robot image built on ros2_humble.
# Sources ROS, then any workspace overlay that happens to exist (vendor
# interfaces for ROBOTICAN, a bind-mounted /workspace for others), exports
# PYTHONPATH for the repo, then execs. Deliberately has no robot-specific
# logic -- a robot's own Dockerfile decides what's there to source.
#
# Does NOT set ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI -- those
# come from the compose file's environment, exactly like the existing
# run_*.sh wrappers set them on the host, so one image works against any
# ROS domain without an image rebuild.
# ============================================================
set -eo pipefail

# ROS 2's own setup.bash references variables (e.g. AMENT_TRACE_SETUP_FILES)
# without a default, so it isn't `set -u`-safe -- source with nounset off,
# then turn nounset on for everything else in this script.
source /opt/ros/"${ROS_DISTRO:-humble}"/setup.bash
if [ -f /opt/rooster_ws/install/setup.bash ]; then
  source /opt/rooster_ws/install/setup.bash
fi
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
set -u

export PYTHONPATH="${PYTHONPATH:-}:${HOME}/GIT/TheAgency"

echo "[entrypoint] ROS_DISTRO=${ROS_DISTRO:-humble} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} " \
     "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>} CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"

exec "$@"
