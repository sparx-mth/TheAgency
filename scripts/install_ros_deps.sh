#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/install_ros_deps.sh /absolute/path/to/ros2_ws
# If not provided, defaults to ./ros2_ws relative to repo root.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="${1:-$REPO_ROOT/ros2_ws}"

SRC_DIR="$ROS_WS/src"
APRILTAG_DIR="$SRC_DIR/apriltag_ros"
APRILTAG_GIT="https://github.com/christianrauch/apriltag_ros.git"
APRILTAG_BRANCH="humble"

echo "[install_ros_deps] Repo root: $REPO_ROOT"
echo "[install_ros_deps] ROS workspace: $ROS_WS"
echo "[install_ros_deps] Source dir: $SRC_DIR"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[install_ros_deps] Creating $SRC_DIR"
  mkdir -p "$SRC_DIR"
fi

# 1) Clone (or update) apriltag_ros
if [[ -d "$APRILTAG_DIR/.git" ]]; then
  echo "[install_ros_deps] apriltag_ros already exists -> pulling latest ($APRILTAG_BRANCH)"
  git -C "$APRILTAG_DIR" fetch --all
  git -C "$APRILTAG_DIR" checkout "$APRILTAG_BRANCH" || true
  git -C "$APRILTAG_DIR" pull
else
  echo "[install_ros_deps] Cloning apriltag_ros ($APRILTAG_BRANCH)"
  git clone --branch "$APRILTAG_BRANCH" "$APRILTAG_GIT" "$APRILTAG_DIR"
fi

# 2) Resolve deps + build
echo "[install_ros_deps] Running rosdep install"
cd "$ROS_WS"

# TWEAK 1: rosdep update should run as the normal user, but may need a quick sudo apt update first
# to ensure the package lists are fresh for the rosdep install step.
sudo apt-get update

rosdep update
# TWEAK 2: Added 'sudo' here so rosdep has permission to invoke apt-get install
sudo rosdep install --from-paths src --ignore-src -r -y

# NOTE: We leave 'colcon build' WITHOUT sudo. 
# This ensures build/install directories are owned by your non-root user!
echo "[install_ros_deps] Building workspace with colcon"
colcon build --symlink-install

echo "[install_ros_deps] Done."
echo "Next: source $ROS_WS/install/setup.bash"