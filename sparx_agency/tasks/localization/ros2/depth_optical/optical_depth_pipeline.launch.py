import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    main_venv_python = os.path.expanduser("~/GIT/TheAgency/.venv/bin/python3")
    da3_venv_python = os.path.expanduser("~/depth_anything_ws/src/ros2-depth-anything-v3-trt/da3_venv/bin/python3")

    return LaunchDescription([
        # 1. Flow Depth Velocity Node
        Node(
            executable=main_venv_python,
            arguments=['-m', 'sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node'],
            parameters=[{
                'use_sim_time': False,
                'show_debug': True,
                'csv_filename': "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv",
                'image_topic': '/xtend/rgb',
                'depth_topic': '/xtend/depth_m',
                'camera_info_topic': '/xtend/camera_info',
                'depth_scale': 0.8
            }],
            output='screen'
        ),

        # 2. Velocity Integrator
        Node(
            executable=main_venv_python,
            arguments=['-m', 'sparx_agency.tasks.localization.ros2.depth_optical.velocity_integrator'],
            parameters=[{
                'use_sim_time': False,
                'target_frame': 'odom',
                'init_from_gt': False
            }],
            output='screen'
        ),

        # 3. Static Transform (TF2)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'odom',
                '--child-frame-id', 'xtend_camera'
            ]
        ),

# 4. Live Depth Processor (TensorRT)
        Node(
            executable=da3_venv_python,
            arguments=['-m', 'sparx_agency.tasks.mapping.ros2.depth_processor_node'],
            parameters=[{
                'use_sim_time': False,
                'engine_path': os.path.expanduser("~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine"),
                'config_yaml': os.path.expanduser("~/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"),
                'rgb_topic': '/xtend/rgb',
                'pub_depth_topic': '/xtend/depth_m',
                'pub_debug_topic': '/xtend/depth_vis'
            }],
            output='screen'
        ),

        # 5. RGB Data Publisher 
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        main_venv_python, '-m', 'sparx_agency.tasks.localization.common.publish_rgb_from_files',
                        '--rgb-dir', os.path.expanduser('~/Documents/xtend_da3_takes/xtend_rectified_depth_take_003_20260429_160647/rgb_rectified'),
                        '--publish-hz', '10.0'
                    ],
                    output='screen'
                )
            ]
        )
    ])