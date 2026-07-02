# TurtleBot3 Burger with Depth Camera

This directory contains modifications to add a depth camera (OpenNI/Kinect style sensor) to the TurtleBot3 Burger robot model.

## Files

- `turtlebot3_burger.urdf.xacro` - Modified URDF/Xacro file with depth camera configuration
- `setup_camera.sh` - Setup script to apply the modification in the devcontainer

## What's Added

The modification includes:
- **Gazebo Sensor Configuration**: Depth camera sensor with 30 Hz update rate
- **Camera Link**: Visual and collision geometry for the camera
- **Camera Joints**: Fixed joints connecting camera to robot base
- **Optical Frame**: Required for proper image orientation in ROS
- **ROS Gazebo Plugin**: libgazebo_ros_openni_kinect plugin for ROS integration

### Camera Specifications
- **Update Rate**: 30 Hz
- **Resolution**: 640x480
- **Format**: RGB (R8G8B8)
- **Field of View**: 60 degrees
- **Range**: 0.05m - 3.0m
- **Position**: Front of robot at height 0.15m above base

### Published Topics
- `camera/color/image_raw` - RGB color image
- `camera/color/camera_info` - Camera calibration info
- `camera/depth/image_raw` - Depth image
- `camera/depth/camera_info` - Depth camera info
- `camera/depth/points` - Point cloud data

## Installation Instructions

### Option 1: Using the Setup Script (Recommended)

Inside the devcontainer, run:

```bash
cd /workspace/turtlebot3_modifications
chmod +x setup_camera.sh
./setup_camera.sh
```

This will:
1. Backup the original URDF file
2. Copy the modified version to the ROS installation
3. Verify the installation

### Option 2: Manual Copy

Inside the devcontainer:

```bash
sudo cp /workspace/turtlebot3_modifications/turtlebot3_burger.urdf.xacro \
        /opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro
```

### Option 3: Symlink

For development purposes, create a symlink:

```bash
sudo ln -sf /workspace/turtlebot3_modifications/turtlebot3_burger.urdf.xacro \
            /opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro
```

## Using the Modified Robot

Launch the robot with Gazebo:

```bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

View the camera topic in RViz:

```bash
rviz
# Subscribe to:
# - /camera/color/image_raw (Image display)
# - /camera/depth/points (PointCloud2 display)
```

## Reverting Changes

To restore the original URDF file:

```bash
sudo cp /opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro.bak \
        /opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro
```

## Customization

To modify camera parameters, edit the `turtlebot3_burger.urdf.xacro` file:

- **Horizontal FOV**: `<horizontal_fov>1.047198</horizontal_fov>` (in radians)
- **Resolution**: `<width>640</width>` and `<height>480</height>`
- **Range**: `<near>0.05</near>` and `<far>3.0</far>`
- **Position**: `<origin xyz="0.05 0 0.15" rpy="0 0 0"/>` in the camera_joint

## Notes

- The camera is positioned at the front of the robot, 0.05m forward and 0.15m up from the base
- The plugin uses `libgazebo_ros_openni_kinect.so` which provides realistic depth sensing simulation
- Point cloud data is published at the specified update rate
- Camera info topic is automatically populated with intrinsic parameters
