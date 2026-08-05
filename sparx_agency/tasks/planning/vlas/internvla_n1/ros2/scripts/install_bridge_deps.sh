#!/usr/bin/env bash
set -e

echo "[1/3] Installing system packages (ROS Foxy + OpenCV)..."
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-opencv \
  ros-foxy-cv-bridge

echo "[2/3] Installing Python packages..."
pip3 install --user requests pyyaml

echo "[3/3] Done."
echo "Remember to source ROS before running:"
echo "  source /opt/ros/foxy/setup.bash"
