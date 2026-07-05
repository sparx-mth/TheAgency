from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_name = 'simple_drone'
    share_dir = get_package_share_directory(pkg_name)
    
    world_path = os.path.join(share_dir, 'worlds', 'empty.world')
    drone_model_path = os.path.join(share_dir, 'models', 'simple_drone', 'model.sdf')
    box_model_path = os.path.join(share_dir, 'models', 'target_box', 'model.sdf')

    # Set Resource Path for Gazebo to find models
    env_vars = {
        'GZ_SIM_RESOURCE_PATH': '/usr/share/gz/models:' + os.path.join(share_dir, 'models')
    }

    # 1. Launch Gazebo
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_path],
        output='screen',
        env=env_vars
    )

    # 2. Spawn Drone
    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-file', drone_model_path, '-name', 'simple_drone'],
        output='screen',
        env=env_vars
    )

    # 3. Spawn Box (Dynamic Entity)
    spawn_box = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-file', box_model_path, '-name', 'target_box'],
        output='screen',
        env=env_vars
    )

    # 4. Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/simple_drone/depth/image@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/simple_drone/depth/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo'
        ],
        remappings=[
            ('/simple_drone/depth/image', '/camera/depth/image'),
            ('/simple_drone/depth/camera_info', '/camera/depth/camera_info')
        ],
        output='screen',
        env=env_vars
    )

    return LaunchDescription([
        gazebo,
        spawn_drone,
        spawn_box,  # Spawn box dynamically
        bridge
    ])   