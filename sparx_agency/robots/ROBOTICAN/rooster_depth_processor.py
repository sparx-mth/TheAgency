#!/usr/bin/env python3
"""
ROBOTICAN Rooster depth-processor node.

Extends DepthProcessorNode with static cage-mask inpainting:
  - Loads the precomputed static cage mask from config/cage_static_mask.npy
  - Inpaints permanent side arcs out of each RGB frame before DA3 inference
  - Moving horizontal bar is NOT handled here (use temporal logic at mission level)
"""
import sys
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sparx_agency.tasks.mapping.ros2.depth_processor_node import DepthProcessorNode
from sparx_agency.robots.ROBOTICAN.bar_inpainter import BarInpainter

_DEFAULT_MASK = Path(__file__).resolve().parent / "config" / "cage_static_mask.npy"


class RoosterDepthProcessorNode(DepthProcessorNode):

    def __init__(self):
        super().__init__()
        self.declare_parameter("cage_mask_path", str(_DEFAULT_MASK))
        mask_path = str(self.get_parameter("cage_mask_path").value)
        self._bar_inpainter = BarInpainter(mask_path)
        self.get_logger().info(f"BarInpainter loaded: {mask_path}")

    def _run_inference_and_publish(self, bgr: np.ndarray, header, rgb_stem: str = ""):
        bgr_clean = self._bar_inpainter.process(bgr)
        super()._run_inference_and_publish(bgr_clean, header, rgb_stem=rgb_stem)


def main(args=None):
    rclpy.init(args=args)
    node = RoosterDepthProcessorNode()
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
