import rclpy
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.SJTU.adapters.gazebo_ros2_ingest import GazeboRos2Ingestor


def main():
    rclpy.init()

    node = GazeboRos2Ingestor()

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