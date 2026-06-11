#!/usr/bin/env python3
"""
Full System Launch File for InternNav Bridge.

This launch file starts both the model server (in a separate process)
and the ROS2 bridge node.

Usage:
    ros2 launch internnav_bridge full_system.launch.py
    ros2 launch internnav_bridge full_system.launch.py model_host:=192.168.1.100
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('internnav_bridge')
    
    # Default config path
    default_config = os.path.join(pkg_share, 'config', 'bridge_config.yaml')
    
    # Launch arguments
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
    
    model_host_arg = DeclareLaunchArgument(
        'model_host',
        default_value='localhost',
        description='Model server host'
    )
    
    model_port_arg = DeclareLaunchArgument(
        'model_port',
        default_value='8000',
        description='Model server port'
    )
    
    start_server_arg = DeclareLaunchArgument(
        'start_server',
        default_value='true',
        description='Whether to start the model server'
    )
    
    model_device_arg = DeclareLaunchArgument(
        'model_device',
        default_value='cuda',
        description='Device for model inference (cuda or cpu)'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level'
    )
    
    # Model server process (optional - starts if start_server is true)
    # Note: In practice, you might want to run this separately for better control
    model_server_cmd = ExecuteProcess(
        cmd=[
            'python3', '-m', 'internnav_bridge.model_server',
            '--host', '0.0.0.0',
            '--port', LaunchConfiguration('model_port'),
            '--device', LaunchConfiguration('model_device'),
        ],
        name='model_server',
        output='screen',
    )
    
    # Bridge node (delayed to allow server startup)
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
    
    # Delay bridge start to allow server startup
    delayed_bridge = TimerAction(
        period=5.0,  # Wait 5 seconds for server to start
        actions=[bridge_node],
    )
    
    return LaunchDescription([
        # Log startup
        LogInfo(msg=['Starting InternNav Full System...']),
        
        # Arguments
        config_path_arg,
        use_preset_arg,
        model_host_arg,
        model_port_arg,
        start_server_arg,
        model_device_arg,
        log_level_arg,
        
        # Start model server
        model_server_cmd,
        
        # Start bridge (delayed)
        delayed_bridge,
    ])
