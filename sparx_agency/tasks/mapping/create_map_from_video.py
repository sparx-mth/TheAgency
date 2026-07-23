import rclpy

from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.SJTU.adapters.gazebo_ros2_ingest import GazeboRos2Ingestor


def main():
    rclpy.init()

    # --- Step 1: Configure the Brain ---
    map_cfg = ProbabilisticGridConfig(points_to_occupied=30)
    costmap = ProbabilisticGridCostmap(map_cfg)

    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())  # vits
    cloud_gen = PinholeCloudGenerator()

    # --- Step 2: Assemble the Pipeline ---
    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=cloud_gen
    )

    # --- Step 3: Plug into the Drone Adapter ---
    # This node handles the Gazebo/ROS 'plumbing'
    node = GazeboRos2Ingestor(pipeline=pipeline, costmap=costmap)

    try:
        print("Task Started: Mapping environment from video stream...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Here you can add logic to save the map to a file before exiting
        print("Saving map...")
        # costmap.save("final_map.npy")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()