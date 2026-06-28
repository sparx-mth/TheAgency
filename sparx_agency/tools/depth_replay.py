#!/usr/bin/env python3
"""
Generate missing depth NPY files for an existing capture session by replaying
the JPG paths through a running depth_processor_node.

Usage
-----
# Terminal 1 — start depth processor with depth_dir = session dir:
source /opt/ros/humble/setup.bash
source /home/user/depth_anything_ws/install/setup.bash
export ROS_DOMAIN_ID=5
python3 /home/user/agency_ws/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p frame_path_topic:=/xtend/rgb_frame_path \
  -p depth_topic:=/xtend/depth_m \
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine \
  -p config_yaml:=/home/user/agency_ws/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  -p camera_info_mode:=base -p model_type:=large_metric \
  -p apply_metric_focal_scaling:=true -p metric_scale_divisor:=300.0 \
  -p clip_min_m:=0.2 -p clip_max_m:=5.0 -p depth_encoding:=32FC1 \
  -p depth_path_topic:=/xtend/depth_frame_path \
  -p depth_dir:=/PATH/TO/SESSION \
  -p max_depth_kept:=9999

# Terminal 2 — replay frames:
source /opt/ros/humble/setup.bash
source /home/user/agency_ws/venv/bin/activate
export ROS_DOMAIN_ID=5
cd /home/user/agency_ws
python3 sparx_agency/tools/depth_replay.py /PATH/TO/SESSION
"""

import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(
        description="Replay capture JPGs through depth_processor_node to generate NPY files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session_dir", help="Capture session directory (contains *.jpg)")
    parser.add_argument("--topic", default="/xtend/rgb_frame_path",
                        help="RGB frame path topic the depth processor subscribes to")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between publishes — must be > depth processor latency")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip JPGs that already have a matching .npy")
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    if not session.is_dir():
        print(f"[replay] ERROR: {session} is not a directory")
        return

    jpgs = sorted(session.glob("*.jpg"))
    if not jpgs:
        print(f"[replay] no JPG files found in {session}")
        return

    if args.skip_existing:
        jpgs = [j for j in jpgs if not j.with_suffix(".npy").exists()]
        print(f"[replay] {len(jpgs)} frames missing depth")
    else:
        print(f"[replay] {len(jpgs)} frames total")

    if not jpgs:
        print("[replay] nothing to do")
        return

    rclpy.init()
    node = Node("depth_replay")
    pub = node.create_publisher(String, args.topic, 10)

    print(f"[replay] waiting 2 s for depth_processor_node to subscribe...")
    time.sleep(2.0)

    for i, jpg in enumerate(jpgs):
        msg = String()
        msg.data = str(jpg)
        pub.publish(msg)
        print(f"[replay] {i+1}/{len(jpgs)}  {jpg.name}")
        time.sleep(args.interval)

    print(f"[replay] done — NPY files should now be in {session}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
