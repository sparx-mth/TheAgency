import rclpy
from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig

# Import the class we defined earlier
from sparx_agency.tasks.mapping.mapping_with_depth_anything import MappingTask


def main():
    rclpy.init()

    # 1. Setup the Costmap (Occupancy Grid)
    map_cfg = ProbabilisticGridConfig(points_to_occupied=30)
    costmap = ProbabilisticGridCostmap(map_cfg)

    # 2. Setup DepthAnythingV2
    # NOTE: Use 'vits' (small) encoder for Jetson performance
    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())

    # 3. Assemble the Pipeline
    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=PinholeCloudGenerator()
    )

    # 4. Initialize the Task Node
    # This node handles the Triggering/Request logic
    node = MappingTask(pipeline=pipeline)

    import os
    print("Node:", node.get_name(), "ns:", node.get_namespace())
    print("ROS_DOMAIN_ID:", os.getenv("ROS_DOMAIN_ID"))
    print("RMW_IMPLEMENTATION:", os.getenv("RMW_IMPLEMENTATION"))
    print("ROS_LOCALHOST_ONLY:", os.getenv("ROS_LOCALHOST_ONLY"))
    print("CYCLONEDDS_URI:", os.getenv("CYCLONEDDS_URI"))

    print("Jetson Mapping Node started.")
    print("Asking Rooster/Sphera for the first frame...")

    # Trigger the first frame request
    node.request_new_frame()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down Jetson Mapping...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()