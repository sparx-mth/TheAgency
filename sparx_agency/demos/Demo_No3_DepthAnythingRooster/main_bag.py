import argparse
import os
import rclpy

from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig

from sparx_agency.tasks.mapping.mapping_depth_anything_cb import MappingTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone-id", default="R1")
    parser.add_argument("--period", type=float, default=2, help="Run DepthAnything once every N seconds.")
    parser.add_argument("--no-caminfo", action="store_true", help="Allow running without camera_info (not recommended).")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--hfov-deg", type=float, default=130.0)
    parser.add_argument("--vfov-deg", type=float, default=90.0)
    args = parser.parse_args()

    rclpy.init()

    costmap = ProbabilisticGridCostmap(ProbabilisticGridConfig(points_to_occupied=30))
    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())

    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=PinholeCloudGenerator(),
    )

    node = MappingTask(
        pipeline=pipeline,
        drone_id=args.drone_id,
        process_period_sec=args.period,
        width=args.width,
        height=args.height,
        hfov_deg=args.hfov_deg,
        vfov_deg=args.vfov_deg,
    )

    print("Node:", node.get_name(), "ns:", node.get_namespace())
    print("ROS_DOMAIN_ID:", os.getenv("ROS_DOMAIN_ID"))
    print("RMW_IMPLEMENTATION:", os.getenv("RMW_IMPLEMENTATION"))
    print("ROS_LOCALHOST_ONLY:", os.getenv("ROS_LOCALHOST_ONLY"))
    print("CYCLONEDDS_URI:", os.getenv("CYCLONEDDS_URI"))
    print("Jetson Mapping Node started (BAG MODE).")

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
