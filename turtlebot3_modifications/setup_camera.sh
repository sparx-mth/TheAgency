#!/bin/bash
# Setup script to replace turtlebot3_burger URDF with depth camera version
# This script should be run inside the devcontainer

echo "Setting up turtlebot3_burger with depth camera..."

TUTLEBOT3_MODIFICATIONS_DIR="/workspace/turtlebot3_modifications"

# Backup original file
# if [ -f "/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro" ]; then
#     cp /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro \
#        /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro.bak

#     cp /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf \
#        /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.bak


#     cp /opt/ros/humble/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py \
#        /opt/ros/humble/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py.bak

#     echo "Original file backed up to .bak"
# fi

# Copy the modified file
if [ -d "$TUTLEBOT3_MODIFICATIONS_DIR" ]; then
    
    cp /workspace/turtlebot3_modifications/spawn_turtlebot3.launch.py \
       /opt/ros/humble/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py

    # cp /workspace/turtlebot3_modifications/common_properties.xacro \
    #    /opt/ros/humble/share/turtlebot3_description/urdf/common_properties.xacro

    cp /workspace/turtlebot3_modifications/turtlebot3_burger.urdf.xacro \
       /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro

    # cd /opt/ros/humble/share/turtlebot3_description/urdf && source /usr/share/gazebo-11/setup.sh && source /opt/ros/humble/setup.bash && ros2 run xacro xacro turtlebot3_burger.urdf.xacro > turtlebot3_burger.urdf && cd "/workspace"

    echo "Modified URDF with depth camera installed successfully!"
else
    echo "Warning: $TUTLEBOT3_MODIFICATIONS_DIR directory not found"
    exit 1
fi

echo "Setup complete. The turtlebot3_burger now includes a depth camera."
