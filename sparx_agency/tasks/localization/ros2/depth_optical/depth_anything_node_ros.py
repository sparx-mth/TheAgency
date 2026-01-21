#!/usr/bin/env python3
import rclpy

from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import (
    MappingPipeline,
    PinholeCloudGenerator,
)
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.SJTU.adapters.gazebo_ros2_ingest import GazeboRos2Ingestor


def main():
    rclpy.init()

    # --- Step 1: Configure components ---
    map_cfg = ProbabilisticGridConfig(points_to_occupied=30)
    costmap = ProbabilisticGridCostmap(map_cfg)

    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())  # vits
    cloud_gen = PinholeCloudGenerator(stride=4)  # downsample cloud for RViz performance

    # --- Step 2: Assemble the pipeline ---
    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=cloud_gen,
    )

    # --- Step 3: Plug into the Gazebo adapter ---
    node = GazeboRos2Ingestor(pipeline=pipeline, costmap=costmap)

    # ✅ Configure publishers via parameters:
    # - depth image:      /depth_anything/depth
    # - depth visualized: /depth_anything/depth_vis
    # - point cloud:      /depth_anything/cloud  (optional)
    #
    # NOTE:
    # This assumes you made the small change inside GazeboRos2Ingestor to publish
    # those topics (instead of /debug/*), OR you will remap the topics at runtime.

    try:
        print("Task Started: Depth (and optional cloud) from live ROS topics...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
