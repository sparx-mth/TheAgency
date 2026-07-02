import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # Get the model name (e.g., 'burger')
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    
    # 1. Point to the XACRO file instead of the static SDF
    # Ensure this path matches where you saved your edited xacro file
    # If editing the system file, use the path below. 
    # If using a workspace, update get_package_share_directory to point to your workspace package.
    urdf_xacro_path = os.path.join(
        get_package_share_directory('turtlebot3_description'),
        'urdf',
        f'{model_folder}.urdf.xacro'
    )

    # Launch configuration variables
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    # Declare launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Specify x position of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Specify y position of the robot')

    # 2. Robot State Publisher: Processes Xacro and publishes to 'robot_description'
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_xacro_path])
        }],
        output='screen'
    )

    # 3. Spawn Entity: Subscribes to 'robot_description' topic instead of loading a file
    start_gazebo_ros_spawner_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', TURTLEBOT3_MODEL,
            '-topic', 'robot_description',  # Changed from '-file' to '-topic'
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01'
        ],
        output='screen',
    )

    ld = LaunchDescription()

    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)
    
    # Add the robot_state_publisher FIRST so the description is ready before spawning
    ld.add_action(robot_state_publisher_node)
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld   