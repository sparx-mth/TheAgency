#!/usr/bin/env python3
"""Launch file for InternNav Bridge."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('internnav_bridge')
    default_config = os.path.join(pkg_share, 'config', 'bridge_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=default_config),
        DeclareLaunchArgument('log_level', default_value='info'),

        Node(
            package='internnav_bridge',
            executable='bridge_node',
            name='internnav_bridge',
            output='screen',
            parameters=[{'config_path': LaunchConfiguration('config_path')}],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
    ])