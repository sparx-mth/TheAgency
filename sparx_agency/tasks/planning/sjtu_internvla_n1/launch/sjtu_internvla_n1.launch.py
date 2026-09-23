#!/usr/bin/env python3
"""Bring up the two CPU-side ROS2 nodes that fly SJTU under InternVLA-N1.

The N1 policy node (camera + instruction -> committed world path) and the pure
pursuit follower (path -> ``/cmd_vel``). Both read the one binding YAML and both
are pinned off the GPU with ``CUDA_VISIBLE_DEVICES=""`` -- the card is the model
server's alone.

This does **not** start Gazebo or the model server; ``scripts/run_sjtu_n1.sh``
orchestrates those. Launched on its own it expects the SJTU world already up on
the matching ``ROS_DOMAIN_ID`` and RMW, and the server reachable over loopback.

The nodes run as ``python3 -m`` modules rather than installed entry points,
because this task package is imported from the repo root and is not colcon-built.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# repo root = .../TheAgency, six parents up from this launch file.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_CONFIG = (_REPO_ROOT / "sparx_agency" / "robots" / "SJTU" / "config"
                   / "vla" / "internvla_n1.yaml")

_POLICY_MODULE = "sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.n1_policy_node"
_FOLLOWER_MODULE = "sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.trajectory_follower_node"
_RECORDER_MODULE = "sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.n1_run_recorder_node"
_SUPERVISOR_MODULE = ("sparx_agency.tasks.planning.sjtu_internvla_n1.ros2"
                      ".exploration_supervisor_node")


def _node_env():
    """Process env for the nodes: on the CPU, and importable from the repo root."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # the GPU is the model server's
    env["PYTHONUNBUFFERED"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def generate_launch_description():
    config = LaunchConfiguration("config_file")
    record = LaunchConfiguration("record")
    record_output = LaunchConfiguration("record_output")
    record_seconds = LaunchConfiguration("record_seconds")
    supervise = LaunchConfiguration("supervise")
    include_location = LaunchConfiguration("include_location")
    goal_only = LaunchConfiguration("goal_only")
    env = _node_env()
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file", default_value=str(_DEFAULT_CONFIG),
            description="SJTU/InternVLA-N1 binding YAML both nodes read."),
        DeclareLaunchArgument(
            "record", default_value="false",
            description="Also start the run recorder (camera + route + FPS -> MP4)."),
        DeclareLaunchArgument(
            "record_output", default_value="/tmp/sjtu_n1/run.mp4",
            description="MP4 path the recorder writes when record:=true."),
        DeclareLaunchArgument(
            "supervise", default_value="false",
            description="Start the exploration supervisor, which rewrites the "
                        "instruction as each sub-mission completes."),
        DeclareLaunchArgument(
            "include_location", default_value="true",
            description="Supervisor briefing part two: which area you are in."),
        DeclareLaunchArgument(
            "goal_only", default_value="false",
            description="Drop both narrative parts; the control arm of the "
                        "with/without-narrative experiment."),
        DeclareLaunchArgument(
            "record_seconds", default_value="0.0",
            description="Seconds to record before the recorder closes its own "
                        "file and exits; 0 records until stopped."),
        # `on_exit=Shutdown()` on every one of them, deliberately. Without it a
        # node that dies at import leaves `ros2 launch` alive and healthy-looking
        # -- the caller's `kill -0` passes, the flight proceeds, and the failure
        # is one traceback buried in a redirected log. That is exactly how a run
        # produces a full rosbag and no video at all.
        ExecuteProcess(
            cmd=["python3", "-m", _POLICY_MODULE,
                 "--ros-args", "-p", ["config_file:=", config]],
            name="n1_policy_node", output="screen", additional_env=env,
            on_exit=Shutdown()),
        ExecuteProcess(
            cmd=["python3", "-m", _FOLLOWER_MODULE,
                 "--ros-args", "-p", ["config_file:=", config]],
            name="trajectory_follower_node", output="screen", additional_env=env,
            on_exit=Shutdown()),
        ExecuteProcess(
            condition=IfCondition(record),
            cmd=["python3", "-m", _RECORDER_MODULE,
                 "--ros-args", "-p", ["config_file:=", config],
                 "-p", ["output:=", record_output],
                 "-p", ["record_seconds:=", record_seconds]],
            name="n1_run_recorder_node", output="screen", additional_env=env,
            on_exit=Shutdown()),
        ExecuteProcess(
            condition=IfCondition(supervise),
            cmd=["python3", "-m", _SUPERVISOR_MODULE,
                 "--ros-args", "-p", ["config_file:=", config],
                 "-p", ["include_location:=", include_location],
                 "-p", ["goal_only:=", goal_only]],
            name="exploration_supervisor_node", output="screen",
            additional_env=env, on_exit=Shutdown()),
    ])

