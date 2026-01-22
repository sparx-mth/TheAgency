#!/usr/bin/env python3
"""
Launch file for InternNav Bridge.

This launch file starts the bridge node with configurable parameters.
You can specify a custom config file or use a preset for common simulators.

Usage:
    ros2 launch internnav_bridge bridge.launch.py
    ros2 launch internnav_bridge bridge.launch.py config_path:=/path/to/config.yaml
    ros2 launch internnav_bridge bridge.launch.py use_preset:=gazebo
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('internnav_bridge')
    
    # Default config path
    default_config = os.path.join(pkg_share, 'config', 'bridge_config.yaml')
    
    # Declare launch arguments
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=default_config,
        description='Path to the bridge configuration YAML file'
    )
    
    use_preset_arg = DeclareLaunchArgument(
        'use_preset',
        default_value='',
        description='Use a preset configuration (isaac_sim, gazebo, habitat, unity)'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error)'
    )
    
    # Bridge node
    bridge_node = Node(
        package='internnav_bridge',
        executable='bridge_node',
        name='internnav_bridge',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('config_path'),
            'use_preset': LaunchConfiguration('use_preset'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )
    
    return LaunchDescription([
        # Log startup info
        LogInfo(msg=['Starting InternNav Bridge...']),
        LogInfo(msg=['Config: ', LaunchConfiguration('config_path')]),
        
        # Arguments
        config_path_arg,
        use_preset_arg,
        log_level_arg,
        
        # Nodes
        bridge_node,
    ])
