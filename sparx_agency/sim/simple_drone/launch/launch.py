from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration

import os
from pathlib import Path

def generate_launch_description():
    pkg_name = 'simple_drone'
    share_dir = FindPackageShare(pkg_name).find(pkg_name)
    
    world_path = PathJoinSubstitution([share_dir, 'worlds', 'empty.world'])
    drone_model_path = PathJoinSubstitution([share_dir, 'models', 'simple_drone', 'model.sdf'])
    box_model_path = PathJoinSubstitution([share_dir, 'models', 'target_box', 'model.sdf'])
    rviz_config_path = os.path.join('rviz', 'simple_drone.rviz')


    # CRITICAL FIX: Use additional_env and INCLUDE LD_LIBRARY_PATH
    # This appends to the existing environment instead of replacing it
    extra_env = {
        'LD_LIBRARY_PATH': '/opt/ros/humble/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
        'GZ_SIM_RESOURCE_PATH': '/usr/share/gz/models:' + os.path.join(share_dir, 'models'),
        'HOME': '/home/user1',
        'XDG_CONFIG_HOME': '/home/user1/.config'
    }

    # 1. Launch Gazebo
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_path],
        output='screen',
        additional_env=extra_env  # <--- CHANGE 'env' TO 'additional_env'
    )

    # 2. Spawn Drone
    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-file', drone_model_path, '-name', 'simple_drone',
                      '-x', '0.0',  
                      '-y', '0.0', 
                      '-z', '0.5'   
        ],
        output='screen',
        additional_env=extra_env  # <--- CHANGE 'env' TO 'additional_env'
    )

    # 3. Spawn Box
    spawn_box = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', box_model_path, 
            '-name', 'target_box',
            '-x', '2.0',  # Explicitly set X
            '-y', '0.0', # Explicitly set Y
            '-z', '0.5'   # Explicitly set Z
        ],
        output='screen',
        additional_env=extra_env  # <--- CHANGE 'env' TO 'additional_env'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': open('sim/simple_drone/models/simple_drone/drone.urdf').read()}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    # 4. Bridge (Updated)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Use the topic names confirmed by 'ign topic -l'
            '/drone/depth/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/drone/depth/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/drone/depth/image/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked'
        ],
        remappings=[
            ('/drone/depth/image', '/camera/depth/image'),
            ('/drone/depth/camera_info', '/camera/depth/camera_info'),
            ('/drone/depth/image/points', '/camera/depth/points')
        ],
        output='screen',
        additional_env=extra_env
    )
    
    frame_fixer = ExecuteProcess(cmd=['python3', 'sim/simple_drone/simple_drone/frame_fixer.py'],output='screen')


    delayed_nodes = TimerAction(
        period=10.0,  # Wait 5 seconds for Gazebo to settle
        actions=[spawn_drone, spawn_box, robot_state_publisher, rviz_node, bridge, frame_fixer]
    )

    return LaunchDescription([
        DeclareLaunchArgument(name='use_sim_time', default_value='True', description='Flag to enable use_sim_time'),
        DeclareLaunchArgument(name='rvizconfig', default_value=rviz_config_path, description='Absolute path to rviz config file'),
        gazebo,
        delayed_nodes
    ])

