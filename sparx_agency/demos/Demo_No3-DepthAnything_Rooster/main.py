import rclpy
from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig

# New Sphera Ingestor
from sparx_agency.robots.ROBOTICAN.adapters.sphera_ros2_ingestor import SpheraRos2Ingestor

def main():
    rclpy.init()

    # 1. Configure the Brain (Same as SJTU)
    map_cfg = ProbabilisticGridConfig(points_to_occupied=30)
    costmap = ProbabilisticGridCostmap(map_cfg)
    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())
    cloud_gen = PinholeCloudGenerator()

    # 2. Assemble Pipeline
    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=cloud_gen
    )

    # 3. Initialize Ingestor (The "Plumbing" for Rooster)
    node = SpheraRos2Ingestor(pipeline=pipeline, costmap=costmap, drone_id="R2")

    # 4. START: Ask for video explicitly
    node.activate_video_hardware()

    try:
        print("Mapping Started: Receiving hardware stream and running DepthAnything...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down and saving map...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()