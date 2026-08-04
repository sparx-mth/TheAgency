#!/usr/bin/env python3
"""
ROBOTICAN Rooster depth-processor node.

Plain DepthProcessorNode entry point — kept as its own script because
run_depth_processor.sh (and mission_control.py/rooster_turn_debug.py's
process-detection patterns) already name it directly.

Cage removal (both the static arcs and the moving crossbar) used to happen
here via BarInpainter, but moved upstream into rooster_frame_dir_publisher.py
2026-08-04: this node and the YOLO detector both read the same saved JPEG off
disk (frame_path transport), so cleaning once at the point the frame is
written means neither has to duplicate the logic. See bar_inpainter.py.
"""
import sys
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sparx_agency.tasks.mapping.ros2.depth_processor_node import DepthProcessorNode


def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
