import rclpy
# 1. Update imports to use the DA3 TensorRT class
from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.SJTU.adapters.gazebo_ros2_ingest import GazeboRos2Ingestor


def main():
    rclpy.init()

    # --- Step 1: Configure the Brain ---
    map_cfg = ProbabilisticGridConfig(points_to_occupied=30)
    costmap = ProbabilisticGridCostmap(map_cfg)

    # 2. Initialize DA3 with your specific paths
    # Replace these paths with your actual Jetson environment paths
    ENGINE_PATH = "/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine"
    YAML_PATH = "/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_laptop.yaml"

    depth_model = DA3TensorRTModel(engine_path=ENGINE_PATH, yaml_path=YAML_PATH)

    cloud_gen = PinholeCloudGenerator()

    # --- Step 2: Assemble the Pipeline ---
    pipeline = MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cloud_generator=cloud_gen
    )

    # --- Step 3: Plug into the Drone Adapter ---
    node = GazeboRos2Ingestor(pipeline=pipeline, costmap=costmap)

    try:
        print(f"Task Started: Mapping environment using DepthAnything V3 (TRT)... ")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Saving map and exiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()